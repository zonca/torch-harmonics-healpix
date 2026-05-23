"""Training script for Test 2: ℓ_Ep/ℓ_Bp estimation from Q/U polarization maps.

v2: Paper-matching hyperparameters (same as train_test1_v2.py):
  - 100k training maps (paper: 100k)
  - 10k validation maps (paper: 10k)
  - 1k test maps (paper: 1k)
  - Batch size 32 (paper: 32)
  - ReduceLROnPlateau (patience=5, factor=0.1) (paper: same)
  - Early stopping (patience=20) (paper: same)

Paper results (full sky, no noise):
  NNhealpix: ℓ_Ep=2.7%, ℓ_Bp=2.7%
  MCMC: ℓ_Ep=0.7%, ℓ_Bp=0.7%

Paper also tests partial sky: f_sky = 0.5, 0.2, 0.1, 0.05.
"""

import argparse
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_harmonics_healpix.data_generation_test2 import (
    generate_polarization_map,
    create_sky_mask,
    LEP_MIN, LEP_MAX, LBP_MIN, LBP_MAX,
    NSIDE, LMAX, SIGMA_P,
)
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


class PolarizationDataset(Dataset):
    """On-the-fly Q/U polarization map generation."""

    def __init__(self, n_maps, nside=NSIDE, lmax=LMAX, sigma_p=SIGMA_P,
                 noise_std=0.0, f_sky=1.0, seed=42, mask=None):
        self.n_maps = n_maps
        self.nside = nside
        self.lmax = lmax
        self.sigma_p = sigma_p
        self.noise_std = noise_std
        self.f_sky = f_sky
        self.rng = np.random.default_rng(seed)

        # Pre-generate target values
        self.ell_ep = self.rng.uniform(LEP_MIN, LEP_MAX, size=n_maps).astype(np.float32)
        self.ell_bp = self.rng.uniform(LBP_MIN, LBP_MAX, size=n_maps).astype(np.float32)

        # Use provided mask or generate one (shared mask is critical for
        # SpectralCNN generalization — see train_test2_v2.py comments)
        if mask is not None:
            self.mask = mask
        else:
            self.mask = create_sky_mask(f_sky, nside, self.rng).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        """Generate and return a single (QUMap, [ℓ_Ep, ℓ_Bp]) pair on-the-fly."""
        q, u = generate_polarization_map(
            self.ell_ep[idx], self.ell_bp[idx],
            self.nside, self.lmax, self.sigma_p,
            self.noise_std, self.rng
        )

        # Apply sky mask
        q = q * self.mask
        u = u * self.mask

        # Stack Q, U, mask as 3 channels: [3, npix]
        qu_map = np.stack([q, u, self.mask], axis=0).astype(np.float32)
        target = torch.tensor([self.ell_ep[idx], self.ell_bp[idx]], dtype=torch.float32)

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
        loss = nn.functional.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate polarization estimation. Returns dict with ep_pct_error, bp_pct_error."""
    model.eval()
    ep_errors = []
    bp_errors = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        # Percentage errors for E and B separately
        # Clamp denominator to avoid division by near-zero true values,
        # which produces astronomically large % errors for maps with
        # very small ℓ_Ep or ℓ_Bp. This is especially problematic at
        # low f_sky where many realizations have near-zero polarization.
        ep_denom = batch_y[:, 0].abs().clamp(min=1.0)
        bp_denom = batch_y[:, 1].abs().clamp(min=1.0)
        ep_pct = (pred[:, 0] - batch_y[:, 0]).abs() / ep_denom * 100
        bp_pct = (pred[:, 1] - batch_y[:, 1]).abs() / bp_denom * 100
        ep_errors.extend(ep_pct.tolist())
        bp_errors.extend(bp_pct.tolist())

    return {
        "ep_pct_error": np.mean(ep_errors),
        "bp_pct_error": np.mean(bp_errors),
        "ep_median_pct_error": float(np.median(ep_errors)),
        "bp_median_pct_error": float(np.median(bp_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralCNN on Test 2 (v2 paper-matching)")
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--f_sky", type=float, default=1.0, help="Sky fraction (1.0=full sky)")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--hidden_channels", type=int, default=32,
                        help="Hidden channels (paper: 32)")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Spectral conv blocks (paper: 4 NBBs)")
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--output", type=str, default="results/test2.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {args.noise_std}, f_sky = {args.f_sky}")
    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LR schedule: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Early stopping: patience={args.patience}")

    # Create a shared sky mask for all datasets (train/val/test).
    # This is critical for SpectralCNN because the SHT is a global operation —
    # the spectral coefficients encode the absolute position of the mask boundary.
    # If train and test use different mask centers, the model cannot generalize.
    # The paper (Krachmalnicoff & Tomasi 2019) uses a single fixed mask per f_sky.
    shared_rng = np.random.default_rng(0)
    shared_mask = create_sky_mask(args.f_sky, NSIDE, shared_rng).astype(np.float32)

    # Create datasets
    train_dataset = PolarizationDataset(
        args.n_train, args.nside, LMAX, SIGMA_P,
        args.noise_std, args.f_sky, seed=42,
        mask=shared_mask,
    )
    val_dataset = PolarizationDataset(
        args.n_val, args.nside, LMAX, SIGMA_P,
        args.noise_std, args.f_sky, seed=1234,
        mask=shared_mask,
    )
    test_dataset = PolarizationDataset(
        args.n_test, args.nside, LMAX, SIGMA_P,
        args.noise_std, args.f_sky, seed=9999,
        mask=shared_mask,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Model: 3 input channels (Q, U, mask), 2 output channels (ℓ_Ep, ℓ_Bp)
    # Enable inpainting for partial-sky observations to prevent zero-masked
    # pixels from corrupting the SHT spectral coefficients
    model = SpectralCNN(
        in_channels=3,
        out_channels=2,
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

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_reached = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val ℓ_Ep%':>9} | {'Val ℓ_Bp%':>9} | {'LR':>10} | {'No Imp':>6}")
    print("-" * 70)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        epoch_time = time.time() - t0

        val_results = evaluate(model, val_loader, device)
        val_loss = val_results["ep_pct_error"] + val_results["bp_pct_error"]

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_reached = epoch
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_results['ep_pct_error']:9.1f}% "
              f"| {val_results['bp_pct_error']:9.1f}% | {lr:10.6f} | {epochs_no_improve:6d}  ({epoch_time:.1f}s)")

        if epochs_no_improve >= args.patience:
            print(f"\n*** Early stopping at epoch {epoch} ***")
            break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps "
          f"(σ_n={args.noise_std}, f_sky={args.f_sky}) ===")
    results = evaluate(model, test_loader, device)

    print(f"\nCNN ℓ_Ep mean % error: {results['ep_pct_error']:.1f}%")
    print(f"CNN ℓ_Bp mean % error: {results['bp_pct_error']:.1f}%")

    # Paper baselines (from Krachmalnicoff & Tomasi 2019)
    print(f"\nPaper baselines (full sky, σ_n=0):")
    print(f"  NNhealpix: ℓ_Ep=2.7%, ℓ_Bp=2.7%")
    print(f"  MCMC:      ℓ_Ep=0.7%, ℓ_Bp=0.7%")

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
            "best_val_loss": float(best_val_loss),
            "batch_size": args.batch_size,
            "lr_schedule": "ReduceLROnPlateau",
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "early_stopping_patience": args.patience,
            "ep_pct_error": float(results["ep_pct_error"]),
            "bp_pct_error": float(results["bp_pct_error"]),
            "n_params": int(n_params),
            "inpaint": True,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save best model weights
    if args.output and best_state is not None:
        model_path = args.output.replace(".json", ".pt")
        torch.save(best_state, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
