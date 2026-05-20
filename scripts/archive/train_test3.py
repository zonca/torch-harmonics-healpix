"""Training script for Test 3: τ estimation from Q/U polarization maps.

Trains SpectralCNN on HEALPix Q/U polarization maps generated from
CAMB power spectra with varying optical depth τ ∈ [0.03, 0.08],
reproducing the benchmark from Krachmalnicoff & Tomasi (2019) Section 6.1.3.

Requires: camb package (pip install camb)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_harmonics_healpix.data_generation_test3 import (
    generate_tau_map,
    TAU_MIN, TAU_MAX,
    NSIDE, LMAX,
)
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


class TauDataset(Dataset):
    """On-the-fly τ map generation."""

    def __init__(self, n_maps, nside=NSIDE, lmax=LMAX,
                 noise_std=0.0, f_sky=1.0, seed=42):
        self.n_maps = n_maps
        self.nside = nside
        self.lmax = lmax
        self.noise_std = noise_std
        self.f_sky = f_sky
        self.rng = np.random.default_rng(seed)

        # Pre-generate τ values
        self.tau_values = self.rng.uniform(TAU_MIN, TAU_MAX, size=n_maps).astype(np.float32)

        # Pre-generate sky mask
        from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
        self.mask = create_sky_mask(f_sky, nside, self.rng).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        q, u, mask = generate_tau_map(
            self.tau_values[idx],
            self.nside, self.lmax, self.noise_std, self.f_sky, self.rng
        )

        # Stack Q, U, mask as 3 channels: [3, npix]
        qu_map = np.stack([q, u, self.mask], axis=0).astype(np.float32)
        target = torch.tensor([self.tau_values[idx]], dtype=torch.float32)

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
        loss = nn.functional.mse_loss(pred, batch_y.squeeze(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    tau_errors = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        # Absolute error in τ (not percentage — τ values are small)
        tau_err = (pred - batch_y.squeeze(-1)).abs()
        # Also compute percentage error for comparison
        tau_pct = tau_err / batch_y.squeeze(-1) * 100
        tau_errors.extend(tau_pct.tolist())

    return {
        "tau_pct_error": np.mean(tau_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralCNN on Test 3 (τ estimation)")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise in μK")
    parser.add_argument("--f_sky", type=float, default=1.0)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--output", type=str, default="results/test3.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {args.noise_std} μK, f_sky = {args.f_sky}")
    print(f"τ range: [{TAU_MIN}, {TAU_MAX}]")

    # Create datasets
    print("\nGenerating datasets (CAMB spectra needed)...")
    train_dataset = TauDataset(
        args.n_train, args.nside, LMAX,
        args.noise_std, args.f_sky, seed=42,
    )
    val_dataset = TauDataset(
        args.n_val, args.nside, LMAX,
        args.noise_std, args.f_sky, seed=1234,
    )
    test_dataset = TauDataset(
        args.n_test, args.nside, LMAX,
        args.noise_std, args.f_sky, seed=9999,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")

    # Model: 3 input channels (Q, U, mask), 1 output (τ)
    model = SpectralCNN(
        in_channels=3,
        out_channels=1,
        nside=args.nside,
        hidden_channels=64,
        num_blocks=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\nEpoch | Train Loss | Val τ %err |       LR")
    print("-" * 50)

    best_val_error = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()

        if val_results["tau_pct_error"] < best_val_error:
            best_val_error = val_results["tau_pct_error"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {train_loss:10.6f} | {val_results['tau_pct_error']:10.1f}% | {lr:.6f}")

    # Load best model and evaluate
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps ===")
    results = evaluate(model, test_loader, device)

    print(f"\nCNN τ mean % error: {results['tau_pct_error']:.1f}%")

    # Paper baselines
    print(f"\nPaper baselines (Krachmalnicoff & Tomasi 2019, Table 3):")
    print(f"  NNhealpix: σ(τ) = 0.005")
    print(f"  MCMC:      σ(τ) = 0.004")

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
            "tau_pct_error": float(results["tau_pct_error"]),
            "n_params": int(n_params),
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
