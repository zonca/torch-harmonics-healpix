"""Train SpectralCNN for Test 4: joint estimation of r (tensor-to-scalar ratio) and τ (optical depth) from Q/U polarization maps. Extends Test 3 to 2-parameter estimation relevant to Simons Observatory."""

import argparse
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_harmonics_healpix.data_generation_test4 import (
    generate_r_tau_map,
    precompute_camb_spectra_r_tau,
    NSIDE, LMAX,
    R_MIN, R_MAX,
    TAU_MIN, TAU_MAX,
    N_CAMB_SPECTRA,
    R_LOG_EPSILON,
)
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


class RTauDataset(Dataset):
    """On-the-fly Q/U map generation for joint r/τ estimation.

    Pre-computes CAMB spectra for N_CAMB_SPECTRA (r, τ) pairs, then
    randomly picks one spectrum for each map. This avoids calling CAMB
    per map and is much faster.

    r is stored in log-space: r_target = log(r + ε) to handle the
    boundary at r = 0 and improve training dynamics across the wide
    dynamic range r ∈ [0, 0.01].
    """

    def __init__(self, n_maps, nside=NSIDE, lmax=LMAX,
                 noise_std=0.0, f_sky=1.0, seed=42, mask=None,
                 camb_spectra=None):
        """Initialize the RTauDataset.

        Args:
            n_maps: Number of maps to generate (dataset length).
            nside: HEALPix Nside parameter.
            lmax: Maximum multipole for power spectra.
            noise_std: White noise standard deviation in μK (per pixel).
            f_sky: Sky fraction observed (1.0 = full sky).
            seed: Random seed for reproducibility.
            mask: Pre-computed sky mask (1D HEALPix array); generated if None.
            camb_spectra: Optional tuple of (r_values, tau_values, cl_ee_array, cl_bb_array)
                to reuse pre-computed CAMB spectra across datasets.
        """
        self.n_maps = n_maps
        self.nside = nside
        self.lmax = lmax
        self.noise_std = noise_std
        self.f_sky = f_sky
        self.rng = np.random.default_rng(seed)

        # Use shared CAMB spectra if provided, otherwise pre-compute
        if camb_spectra is not None:
            r_values, tau_values, self.cl_ee_array, self.cl_bb_array = camb_spectra
        else:
            print(f"  Pre-computing {N_CAMB_SPECTRA} CAMB spectra for RTauDataset...")
            r_values, tau_values, self.cl_ee_array, self.cl_bb_array = \
                precompute_camb_spectra_r_tau(N_CAMB_SPECTRA, lmax, seed=seed + 100)

        # Randomly choose from pre-computed spectra for each map
        self.spectrum_indices = self.rng.integers(0, len(r_values), size=n_maps)
        self.r_values = r_values[self.spectrum_indices]
        self.tau_values = tau_values[self.spectrum_indices]

        # Log-transform r for boundary handling
        self.r_target = np.log(self.r_values + R_LOG_EPSILON).astype(np.float32)
        self.tau_target = self.tau_values.astype(np.float32)

        # Use provided mask or generate one (shared mask critical for SpectralCNN)
        from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
        if mask is not None:
            self.mask = mask
        else:
            self.mask = create_sky_mask(f_sky, nside, self.rng).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        """Generate and return a single (QU_map, [r_log, τ]) pair on-the-fly."""
        spec_idx = self.spectrum_indices[idx]
        q, u, _ = generate_r_tau_map(
            self.r_values[idx],
            self.tau_values[idx],
            self.nside, self.lmax, self.noise_std, self.f_sky, self.rng,
            cl_ee=self.cl_ee_array[spec_idx],
            cl_bb=self.cl_bb_array[spec_idx],
        )

        # Stack Q, U, mask as 3 channels: [3, npix]
        qu_map = np.stack([q, u, self.mask], axis=0).astype(np.float32)
        target = torch.tensor([self.r_target[idx], self.tau_target[idx]],
                              dtype=torch.float32)

        return torch.from_numpy(qu_map), target


def train_one_epoch(model, dataloader, optimizer, device):
    """Train model for one epoch. Returns mean loss.

    Args:
        model: SpectralCNN model to train.
        dataloader: DataLoader yielding (input_maps, targets) batches.
        optimizer: PyTorch optimizer.
        device: torch.device for computation.

    Returns:
        Mean MSE loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = nn.functional.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate joint r/τ estimation. Returns dict with r_pct_error, tau_pct_error, r_bias.

    Args:
        model: SpectralCNN model to evaluate.
        dataloader: DataLoader yielding (input_maps, targets) batches.
        device: torch.device for computation.

    Returns:
        Dict with keys:
            r_pct_error: mean |r_pred - r_true| / r_true × 100 for r_true > 0.001.
            tau_pct_error: mean |τ_pred - τ_true| / |τ_true| × 100 (denom clamped ≥ 0.01).
            r_bias: mean (r_pred - r_true) for r_true < 0.001 (near-zero r detection).
    """
    model.eval()
    r_pct_errors = []
    tau_pct_errors = []
    r_biases = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        # batch_y: [batch, 2] with [r_log, tau]
        r_log_true = batch_y[:, 0]
        tau_true = batch_y[:, 1]

        # Convert r_log back to r
        r_pred = torch.exp(pred[:, 0]) - R_LOG_EPSILON
        r_true = torch.exp(r_log_true) - R_LOG_EPSILON
        tau_pred = pred[:, 1]
        tau_true_f = tau_true

        # r percentage error for r_true > 0.001
        r_mask = r_true > 0.001
        if r_mask.any():
            r_pct = (r_pred[r_mask] - r_true[r_mask]).abs() / r_true[r_mask] * 100
            r_pct_errors.extend(r_pct.tolist())

        # r bias for r_true < 0.001 (near-zero r detection)
        r_nearzero_mask = r_true < 0.001
        if r_nearzero_mask.any():
            r_bias = (r_pred[r_nearzero_mask] - r_true[r_nearzero_mask]).mean()
            r_biases.append(r_bias.item())

        # τ percentage error
        tau_denom = tau_true_f.abs().clamp(min=0.01)
        tau_pct = (tau_pred - tau_true_f).abs() / tau_denom * 100
        tau_pct_errors.extend(tau_pct.tolist())

    results = {
        "r_pct_error": float(np.mean(r_pct_errors)) if r_pct_errors else float("inf"),
        "tau_pct_error": float(np.mean(tau_pct_errors)) if tau_pct_errors else float("inf"),
        "r_bias": float(np.mean(r_biases)) if r_biases else 0.0,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train SpectralCNN on Test 4 (joint r/τ estimation)"
    )
    parser.add_argument("--f_sky", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0,
                        help="Noise in μK-arcmin")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--camb_cache", type=str, default=None,
                        help="NPZ file to cache/load CAMB spectra (saves ~1.5h on repeat runs)")
    args = parser.parse_args()

    # μK-arcmin → μK/pixel: σ_pix = σ_arcmin / sqrt(Ω_pix [arcmin²])
    noise_arcmin = args.noise_std
    npix = 12 * NSIDE**2
    pixel_area_sr = 4.0 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_sr * (180.0 * 60.0 / np.pi)**2
    noise_uK = noise_arcmin / np.sqrt(pixel_area_arcmin2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {noise_arcmin} μK-arcmin = {noise_uK:.4f} μK, f_sky = {args.f_sky}")
    print(f"r range: [{R_MIN}, {R_MAX}], τ range: [{TAU_MIN}, {TAU_MAX}]")
    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LR schedule: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Early stopping: patience={args.patience}")

    # Create datasets
    print("\nGenerating datasets (CAMB spectra needed)...")
    # Shared mask for train/val/test (critical for SpectralCNN)
    from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
    shared_rng = np.random.default_rng(0)
    shared_mask = create_sky_mask(args.f_sky, NSIDE, shared_rng).astype(np.float32)

    # Pre-compute CAMB spectra ONCE (with optional FITS disk cache)
    # FITS cache layout: PrimaryHDU + ImageHDU "R_VALUES"/"TAU_VALUES" + ImageHDU "CL_EE"/"CL_BB"
    if args.camb_cache and os.path.exists(args.camb_cache):
        print(f"  Loading cached CAMB spectra from {args.camb_cache}")
        from astropy.io import fits as pf
        with pf.open(args.camb_cache) as hdul:
            # ImageHDU.data is a plain numpy array (no structured columns)
            r_values = np.array(hdul["R_VALUES"].data, dtype=np.float32)
            tau_values = np.array(hdul["TAU_VALUES"].data, dtype=np.float32)
            cl_ee_array = np.array(hdul["CL_EE"].data, dtype=np.float64)
            cl_bb_array = np.array(hdul["CL_BB"].data, dtype=np.float64)
        shared_camb = (r_values, tau_values, cl_ee_array, cl_bb_array)
        print(f"  Loaded {len(shared_camb[0])} spectra from cache")
    else:
        print(f"  Pre-computing {N_CAMB_SPECTRA} CAMB spectra (shared across all datasets)...")
        shared_camb = precompute_camb_spectra_r_tau(N_CAMB_SPECTRA, LMAX, seed=142)
        if args.camb_cache:
            print(f"  Saving CAMB spectra to {args.camb_cache}")
            from astropy.io import fits as pf
            # FITS cache: ImageHDU for all arrays (simpler than BinTableHDU)
            hdu_r = pf.ImageHDU(shared_camb[0])
            hdu_r.name = "R_VALUES"
            hdu_tau = pf.ImageHDU(shared_camb[1])
            hdu_tau.name = "TAU_VALUES"
            hdu_ee = pf.ImageHDU(shared_camb[2])
            hdu_ee.name = "CL_EE"
            hdu_bb = pf.ImageHDU(shared_camb[3])
            hdu_bb.name = "CL_BB"
            hdul = pf.HDUList([pf.PrimaryHDU(), hdu_r, hdu_tau, hdu_ee, hdu_bb])
            hdul.writeto(args.camb_cache, overwrite=True)
            print(f"  Cache saved ({os.path.getsize(args.camb_cache)/1e6:.1f} MB)")

    train_dataset = RTauDataset(
        args.n_train, NSIDE, LMAX,
        noise_uK, args.f_sky, seed=42,
        mask=shared_mask, camb_spectra=shared_camb,
    )
    val_dataset = RTauDataset(
        args.n_val, NSIDE, LMAX,
        noise_uK, args.f_sky, seed=1234,
        mask=shared_mask, camb_spectra=shared_camb,
    )
    test_dataset = RTauDataset(
        args.n_test, NSIDE, LMAX,
        noise_uK, args.f_sky, seed=9999,
        mask=shared_mask, camb_spectra=shared_camb,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Model: 3 input channels (Q, U, mask), 2 outputs (r_log, τ)
    # Enable inpainting for partial-sky observations
    model = SpectralCNN(
        in_channels=3,
        out_channels=2,
        nside=NSIDE,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        inpaint=(args.f_sky < 1.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_reached = 0

    print(f"\nEpoch | Train Loss | r %err | τ %err | LR | No Imp")
    print("-" * 70)

    train_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        epoch_time = time.time() - t0

        val_results = evaluate(model, val_loader, device)
        val_loss = val_results["r_pct_error"] + val_results["tau_pct_error"]  # Combined validation metric for early stopping (equal weight to both parameters)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_reached = epoch
        r_err = val_results["r_pct_error"]
        tau_err = val_results["tau_pct_error"]
        print(f"{epoch:4d} | {train_loss:10.6f} | {r_err:5.1f}% | {tau_err:5.1f}% | {lr:9.6f} | {epochs_no_improve:3d}  ({epoch_time:.1f}s)")

        if epochs_no_improve >= args.patience:
            print(f"\n*** Early stopping at epoch {epoch} ***")
            break

    train_time = time.time() - train_start

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps ===")
    results = evaluate(model, test_loader, device)

    print(f"\nCNN r mean % error: {results['r_pct_error']:.1f}%")
    print(f"CNN τ mean % error: {results['tau_pct_error']:.1f}%")
    print(f"CNN r bias (near-zero r): {results['r_bias']:.6f}")

    # Fisher forecast comparison (placeholder)
    print(f"\nFisher forecast comparison:")
    print(f"  (Fisher forecasts for joint r/τ not yet implemented — placeholder)")

    # GPU memory usage
    gpu_memory_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nPeak GPU memory: {gpu_memory_mb:.1f} MB")

    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "test": "test4_joint_r_tau",
            "r_pct_error": float(results["r_pct_error"]),
            "tau_pct_error": float(results["tau_pct_error"]),
            "r_bias": float(results["r_bias"]),
            "n_params": int(n_params),
            "noise_std_uK_arcmin": float(noise_arcmin),
            "noise_std_uK": float(noise_uK),
            "f_sky": float(args.f_sky),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "epochs_reached": epoch_reached,
            "best_val_loss": float(best_val_loss),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_schedule": "ReduceLROnPlateau",
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "early_stopping_patience": args.patience,
            "hidden_channels": args.hidden_channels,
            "num_blocks": args.num_blocks,
            "inpaint": bool(args.f_sky < 1.0),
            "train_time_s": train_time,
            "gpu_memory_mb": gpu_memory_mb,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save best model weights
    if best_state is not None:
        model_path = args.output.replace(".json", ".pt")
        torch.save(best_state, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
