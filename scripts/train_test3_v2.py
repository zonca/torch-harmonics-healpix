"""Training script for Test 3: τ estimation from Q/U polarization maps.

v2: Paper-matching hyperparameters:
  - 100k training maps (paper: 100k, generated from 5000 CAMB spectra)
  - 10k validation maps (paper: 10k)
  - 1k test maps (paper: 1k)
  - Batch size 32 (paper: 32)
  - ReduceLROnPlateau (patience=5, factor=0.1)
  - Early stopping (patience=20)

Paper results:
  NNhealpix: τ % error = 4.0%
  MCMC:      τ % error = 2.8%

Requires: camb package (pip install camb)
"""

import argparse
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_harmonics_healpix.data_generation_test3 import (
    generate_tau_map,
    TAU_MIN, TAU_MAX,
    NSIDE, LMAX,
)
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


class TauDataset(Dataset):
    """On-the-fly τ map generation using pre-computed CAMB spectra.

    Paper approach: pre-compute 5000 CAMB spectra, then randomly pick
    one spectrum for each map (much faster than calling CAMB per map).
    """

    def __init__(self, n_maps, nside=NSIDE, lmax=LMAX,
                 noise_std=0.0, f_sky=1.0, seed=42, mask=None,
                 camb_spectra=None):
        self.n_maps = n_maps
        self.nside = nside
        self.lmax = lmax
        self.noise_std = noise_std
        self.f_sky = f_sky
        self.rng = np.random.default_rng(seed)

        # Use shared CAMB spectra if provided, otherwise pre-compute
        from torch_harmonics_healpix.data_generation_test3 import precompute_camb_spectra, N_CAMB_SPECTRA
        if camb_spectra is not None:
            tau_spectra, self.cl_ee_array, self.cl_bb_array = camb_spectra
        else:
            print(f"  Pre-computing {N_CAMB_SPECTRA} CAMB spectra for TauDataset...")
            tau_spectra, self.cl_ee_array, self.cl_bb_array = precompute_camb_spectra(
                N_CAMB_SPECTRA, lmax, seed=seed + 100
            )

        # Randomly choose from pre-computed spectra for each map
        self.spectrum_indices = self.rng.integers(0, N_CAMB_SPECTRA, size=n_maps)
        self.tau_values = tau_spectra[self.spectrum_indices]

        # Use provided mask or generate one (shared mask critical for SpectralCNN)
        from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
        if mask is not None:
            self.mask = mask
        else:
            self.mask = create_sky_mask(f_sky, nside, self.rng).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        """Generate and return a single (QUMap, [τ]) pair on-the-fly."""
        spec_idx = self.spectrum_indices[idx]
        q, u, mask = generate_tau_map(
            self.tau_values[idx],
            self.nside, self.lmax, self.noise_std, self.f_sky, self.rng,
            cl_ee=self.cl_ee_array[spec_idx],
            cl_bb=self.cl_bb_array[spec_idx],
        )

        # Stack Q, U, mask as 3 channels: [3, npix]
        qu_map = np.stack([q, u, self.mask], axis=0).astype(np.float32)
        target = torch.tensor([self.tau_values[idx]], dtype=torch.float32)

        return torch.from_numpy(qu_map), target


def train_one_epoch(model, dataloader, optimizer, device):
    """Train model for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = nn.functional.mse_loss(pred, batch_y.squeeze(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate τ estimation. Returns dict with tau_pct_error."""
    model.eval()
    tau_errors = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        # Percentage error for comparison
        # Clamp denominator to avoid division by near-zero τ values
        tau_denom = batch_y.squeeze(-1).abs().clamp(min=0.01)
        tau_pct = (pred - batch_y.squeeze(-1)).abs() / tau_denom * 100
        tau_errors.extend(tau_pct.tolist())

    return {
        "tau_pct_error": np.mean(tau_errors),
        "tau_median_pct_error": float(np.median(tau_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralCNN on Test 3 (v2 paper-matching)")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise in μK")
    parser.add_argument("--f_sky", type=float, default=1.0)
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
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--output", type=str, default="results/test3_v2.json")
    parser.add_argument("--camb_cache", type=str, default=None,
                        help="FITS file to cache/load CAMB spectra (saves ~2h on repeat runs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {args.noise_std} μK, f_sky = {args.f_sky}")
    print(f"τ range: [{TAU_MIN}, {TAU_MAX}]")
    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LR schedule: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Early stopping: patience={args.patience}")

    # Create datasets
    print("\nGenerating datasets (CAMB spectra needed)...")
    # Shared mask for train/val/test (critical for SpectralCNN — see train_test2_v2.py)
    from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
    from torch_harmonics_healpix.data_generation_test3 import precompute_camb_spectra, N_CAMB_SPECTRA
    shared_rng = np.random.default_rng(0)
    shared_mask = create_sky_mask(args.f_sky, args.nside, shared_rng).astype(np.float32)

    # Pre-compute CAMB spectra ONCE (with optional FITS disk cache)
    if args.camb_cache and os.path.exists(args.camb_cache):
        print(f"  Loading cached CAMB spectra from {args.camb_cache}")
        from astropy.io import fits as pf
        with pf.open(args.camb_cache) as hdul:
            tau_spectra = np.array(hdul["TAU_VALUES"].data, dtype=np.float32)
            cl_ee_array = np.array(hdul["CL_EE"].data, dtype=np.float64)
            cl_bb_array = np.array(hdul["CL_BB"].data, dtype=np.float64)
        shared_camb = (tau_spectra, cl_ee_array, cl_bb_array)
        print(f"  Loaded {len(shared_camb[0])} spectra from cache")
    else:
        print(f"  Pre-computing {N_CAMB_SPECTRA} CAMB spectra (shared across all datasets)...")
        shared_camb = precompute_camb_spectra(N_CAMB_SPECTRA, LMAX, seed=142)
        if args.camb_cache:
            print(f"  Saving CAMB spectra to {args.camb_cache}")
            from astropy.io import fits as pf
            hdu_tau = pf.ImageHDU(shared_camb[0]); hdu_tau.name = "TAU_VALUES"
            hdu_ee = pf.ImageHDU(shared_camb[1]); hdu_ee.name = "CL_EE"
            hdu_bb = pf.ImageHDU(shared_camb[2]); hdu_bb.name = "CL_BB"
            hdul = pf.HDUList([pf.PrimaryHDU(), hdu_tau, hdu_ee, hdu_bb])
            hdul.writeto(args.camb_cache, overwrite=True)
            print(f"  Cache saved ({os.path.getsize(args.camb_cache)/1e6:.1f} MB)")

    train_dataset = TauDataset(
        args.n_train, args.nside, LMAX,
        args.noise_std, args.f_sky, seed=42,
        mask=shared_mask, camb_spectra=shared_camb,
    )
    val_dataset = TauDataset(
        args.n_val, args.nside, LMAX,
        args.noise_std, args.f_sky, seed=1234,
        mask=shared_mask, camb_spectra=shared_camb,
    )
    test_dataset = TauDataset(
        args.n_test, args.nside, LMAX,
        args.noise_std, args.f_sky, seed=9999,
        mask=shared_mask, camb_spectra=shared_camb,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Model: 3 input channels (Q, U, mask), 1 output (τ)
    # Enable inpainting for partial-sky observations
    model = SpectralCNN(
        in_channels=3,
        out_channels=1,
        nside=args.nside,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        inpaint=(args.f_sky < 1.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Paper-matching training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor
    )

    best_val_error = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_reached = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val τ %err':>10} | {'LR':>10} | {'No Imp':>6}")
    print("-" * 60)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        epoch_time = time.time() - t0

        val_results = evaluate(model, val_loader, device)
        val_error = val_results["tau_pct_error"]

        scheduler.step(val_error)

        if val_error < best_val_error:
            best_val_error = val_error
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_reached = epoch
        print(f"{epoch:5d} | {train_loss:10.6f} | {val_results['tau_pct_error']:10.1f}% | {lr:10.6f} | {epochs_no_improve:6d}  ({epoch_time:.1f}s)")

        if epochs_no_improve >= args.patience:
            print(f"\n*** Early stopping at epoch {epoch} ***")
            break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps ===")
    results = evaluate(model, test_loader, device)

    print(f"\nCNN τ mean % error: {results['tau_pct_error']:.1f}%")

    # Paper baselines
    print(f"\nPaper baselines (Krachmalnicoff & Tomasi 2019, Section 6.2):")
    print(f"  NNhealpix: τ % error ≈ 4.0%")
    print(f"  MCMC:      τ % error ≈ 2.8%")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "version": "v2_paper_matching",
            "noise_std": float(args.noise_std),
            "f_sky": float(args.f_sky),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "epochs_reached": epoch_reached,
            "best_val_error": float(best_val_error),
            "batch_size": args.batch_size,
            "lr_schedule": "ReduceLROnPlateau",
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "early_stopping_patience": args.patience,
            "tau_pct_error": float(results["tau_pct_error"]),
            "n_params": int(n_params),
            "inpaint": bool(args.f_sky < 1.0),
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save best model weights
    if args.output and best_state is not None:
        model_path = args.output.replace(".json", ".pt")
        torch.save(best_state, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
