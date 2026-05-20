"""Training script for Test 1: ℓ_p estimation using MultiResSpectralCNN.

v3: Multi-resolution spectral convolution with decreasing ℓ_max per block,
mimicking NNhealpix's multi-resolution pooling approach.

Block 1: ℓ_max=47 (full resolution)
Block 2: ℓ_max=23 (half resolution)
Block 3: ℓ_max=11 (quarter resolution)
Block 4: ℓ_max=5  (1/8 resolution)

Uses paper-matching hyperparams (100k train, batch 32, ReduceLROnPlateau, early stop).
"""

import argparse
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_harmonics_healpix.data_generation import (
    generate_map,
    generate_dataset,
    NSIDE, LMAX, SIGMA_P, ELL_P_MIN, ELL_P_MAX,
)
from torch_harmonics_healpix.models.multires_spectral_cnn import MultiResSpectralCNN
from torch_harmonics_healpix.mcmc_baseline import mcmc_estimate_ell_p


class OnTheFlyDataset(Dataset):
    """On-the-fly map generation to avoid storing 100k maps in memory."""

    def __init__(self, n_maps, nside=NSIDE, lmax=LMAX, sigma_p=SIGMA_P,
                 noise_std=0.0, seed=42):
        self.n_maps = n_maps
        self.nside = nside
        self.lmax = lmax
        self.sigma_p = sigma_p
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Pre-generate target values (maps generated on-the-fly in __getitem__)
        self.ell_p_true = self.rng.uniform(ELL_P_MIN, ELL_P_MAX, size=n_maps).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        m, _ = generate_map(
            self.ell_p_true[idx], self.nside, self.lmax, self.sigma_p,
            self.noise_std, self.rng
        )
        return torch.from_numpy(m), torch.tensor(self.ell_p_true[idx])


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
    pct_errors = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()
        pct = (pred - batch_y).abs() / batch_y * 100
        pct_errors.extend(pct.tolist())
    return np.mean(pct_errors)


def main():
    parser = argparse.ArgumentParser(description="Train MultiResSpectralCNN on Test 1 (v3)")
    parser.add_argument("--noise_std", type=float, default=0.0)
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
    parser.add_argument("--output", type=str, default="results/test1_v3.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {args.noise_std}")
    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LR schedule: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Early stopping: patience={args.patience}")
    print(f"Model: MultiResSpectralCNN (4 blocks, ℓ_max = 47→23→11→5)")

    # Create datasets
    train_dataset = OnTheFlyDataset(args.n_train, NSIDE, LMAX, SIGMA_P, args.noise_std, seed=42)
    val_dataset = OnTheFlyDataset(args.n_val, NSIDE, LMAX, SIGMA_P, args.noise_std, seed=1234)
    test_dataset = OnTheFlyDataset(args.n_test, NSIDE, LMAX, SIGMA_P, args.noise_std, seed=9999)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Multi-resolution SpectralCNN
    model = MultiResSpectralCNN(
        nside=NSIDE,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        in_channels=1,
        out_channels=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_reached = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'No Imp':>6}")
    print("-" * 60)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        epoch_time = time.time() - t0

        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_reached = epoch
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f}% | {lr:10.6f} | {epochs_no_improve:6d}  ({epoch_time:.1f}s)")

        if epochs_no_improve >= args.patience:
            print(f"\n*** Early stopping at epoch {epoch} (no improvement for {args.patience} epochs) ***")
            break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps (noise σ_n={args.noise_std}) ===")

    cnn_pct = evaluate(model, test_loader, device)
    print(f"\nCNN  mean % error: {cnn_pct:.1f}%")

    # Also run MCMC baseline on test set
    print(f"\nRunning MCMC baseline on {args.n_test} test maps...")
    mcmc_errors = []
    t0 = time.time()
    for i in range(args.n_test):
        ell_p_true = test_dataset.ell_p_true[i]
        m = test_dataset[i][0].numpy()
        ell_p_est = mcmc_estimate_ell_p(m, args.noise_std, NSIDE, LMAX)
        mcmc_errors.append(abs(ell_p_est - ell_p_true) / ell_p_true * 100)
    mcmc_time = (time.time() - t0) / args.n_test
    mcmc_pct = np.mean(mcmc_errors)

    print(f"MCMC mean % error: {mcmc_pct:.1f}%")
    print(f"MCMC mean time:    {mcmc_time:.3f}s per map")

    # Paper baselines
    print(f"\nPaper baselines (Krachmalnicoff & Tomasi 2019, Table 1):")
    print(f"  NNhealpix: 1.3% (σ_n=0), 2.9% (σ_n=5), 5.2% (σ_n=10), 8.4% (σ_n=15)")
    print(f"  MCMC:      0.7% (σ_n=0), 2.5% (σ_n=5), 4.8% (σ_n=10), 7.8% (σ_n=15)")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "version": "v3_multires",
            "model": "MultiResSpectralCNN",
            "architecture": "4 blocks, lmax=47->23->11->5, 32 hidden channels",
            "noise_std": float(args.noise_std),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "epochs_reached": epoch_reached,
            "batch_size": args.batch_size,
            "lr_schedule": "ReduceLROnPlateau",
            "cnn_pct_error": float(cnn_pct),
            "mcmc_pct_error": float(mcmc_pct),
            "mcmc_time_per_map": float(mcmc_time),
            "n_params": int(n_params),
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
