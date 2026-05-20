"""Training script for Test 2: ℓ_Ep/ℓ_Bp estimation from Q/U polarization maps.

Trains SpectralCNN on HEALPix Q/U polarization maps with Gaussian-peaked
E and B power spectra, reproducing the benchmark from Krachmalnicoff &
Tomasi (2019) Section 6.1.2.

Architecture: Q and U are stacked as 2 input channels (+1 mask channel)
to SpectralCNN, which outputs 2 values (ℓ_Ep, ℓ_Bp).
"""

import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

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
                 noise_std=0.0, f_sky=1.0, seed=42):
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

        # Pre-generate sky mask (same for all maps)
        self.mask = create_sky_mask(f_sky, nside, self.rng).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
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
    model.eval()
    ep_errors = []
    bp_errors = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        # Percentage errors for E and B separately
        ep_pct = (pred[:, 0] - batch_y[:, 0]).abs() / batch_y[:, 0] * 100
        bp_pct = (pred[:, 1] - batch_y[:, 1]).abs() / batch_y[:, 1] * 100
        ep_errors.extend(ep_pct.tolist())
        bp_errors.extend(bp_pct.tolist())

    return {
        "ep_pct_error": np.mean(ep_errors),
        "bp_pct_error": np.mean(bp_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralCNN on Test 2 (polarization)")
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--f_sky", type=float, default=1.0, help="Sky fraction (1.0=full sky)")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--output", type=str, default="results/test2.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {args.noise_std}, f_sky = {args.f_sky}")

    # Create datasets
    train_dataset = PolarizationDataset(
        args.n_train, args.nside, LMAX, SIGMA_P,
        args.noise_std, args.f_sky, seed=42,
    )
    val_dataset = PolarizationDataset(
        args.n_val, args.nside, LMAX, SIGMA_P,
        args.noise_std, args.f_sky, seed=1234,
    )
    test_dataset = PolarizationDataset(
        args.n_test, args.nside, LMAX, SIGMA_P,
        args.noise_std, args.f_sky, seed=9999,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")

    # Model: 3 input channels (Q, U, mask), 2 output channels (ℓ_Ep, ℓ_Bp)
    model = SpectralCNN(
        in_channels=3,
        out_channels=2,
        nside=args.nside,
        hidden_channels=64,
        num_blocks=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\nEpoch | Train Loss | Val ℓ_Ep% | Val ℓ_Bp% |       LR")
    print("-" * 60)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()

        val_loss = val_results["ep_pct_error"] + val_results["bp_pct_error"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_results['ep_pct_error']:9.1f}% "
              f"| {val_results['bp_pct_error']:9.1f}% | {lr:.6f}")

    # Load best model and evaluate
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps "
          f"(σ_n={args.noise_std}, f_sky={args.f_sky}) ===")
    results = evaluate(model, test_loader, device)

    print(f"\nCNN ℓ_Ep mean % error: {results['ep_pct_error']:.1f}%")
    print(f"CNN ℓ_Bp mean % error: {results['bp_pct_error']:.1f}%")

    # Paper baselines (from Krachmalnicoff & Tomasi 2019, Table 2)
    print(f"\nPaper baselines (full sky, σ_n=0):")
    print(f"  NNhealpix: ℓ_Ep=2.2%, ℓ_Bp=2.8%")
    print(f"  MCMC:      ℓ_Ep=1.7%, ℓ_Bp=2.0%")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    import json
    with open(args.output, "w") as f:
        json.dump({
            "noise_std": float(args.noise_std),
            "f_sky": float(args.f_sky),
            "n_train": args.n_train,
            "n_test": args.n_test,
            "epochs": args.epochs,
            "ep_pct_error": float(results["ep_pct_error"]),
            "bp_pct_error": float(results["bp_pct_error"]),
            "n_params": int(n_params),
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
