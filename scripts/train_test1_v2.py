"""Training script for Test 1: ℓ_p estimation from scalar (T) maps.

Trains SpectralCNN on HEALPix temperature maps with Gaussian-peaked
power spectra, reproducing the benchmark from Krachmalnicoff & Tomasi 2019.

v2: Paper-matching hyperparameters:
  - 100k training maps (paper: 100k)
  - 10k validation maps (paper: 10k)
  - 1k test maps (paper: 1k)
  - Batch size 32 (paper: 32)
  - ReduceLROnPlateau (patience=5, factor=0.1) (paper: ÷10 after 5 epochs no improvement)
  - Early stopping: 20 epochs no improvement (paper: 20)
  - Max 150 epochs (paper: converged ~70)

Usage:
    python scripts/train_test1_v2.py --noise_std 0
    python scripts/train_test1_v2.py --noise_std 5 --n_train 100000
"""

import argparse
import json
import time
import numpy as np
import healpy as hp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_harmonics_healpix.data_generation import generate_map, generate_power_spectrum
from torch_harmonics_healpix.mcmc_baseline import mcmc_estimate_ell_p
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


NSIDE = 16
LMAX = 3 * NSIDE - 1  # 47
SIGMA_P = 5.0  # Paper uses σ_p=5


class OnTheFlyDataset(Dataset):
    """Generate HEALPix maps on-the-fly for training."""

    def __init__(self, n_maps, nside=NSIDE, noise_std=0.0, sigma_p=SIGMA_P,
                 lmax=LMAX, seed=42):
        self.n_maps = n_maps
        self.nside = nside
        self.noise_std = noise_std
        self.sigma_p = sigma_p
        self.lmax = lmax
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        # Random ℓ_p in [5, 20]
        ell_p = self.rng.uniform(5.0, 20.0)
        m = generate_map(
            ell_p=ell_p, nside=self.nside, noise_std=self.noise_std,
            sigma_p=self.sigma_p, lmax=self.lmax, rng=self.rng
        )
        return torch.from_numpy(m), torch.tensor(ell_p, dtype=torch.float32)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for maps, ell_p_true in dataloader:
        maps = maps.to(device)
        ell_p_true = ell_p_true.to(device)

        optimizer.zero_grad()
        pred = model(maps)
        loss = criterion(pred, ell_p_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, nside=NSIDE, sigma_p=SIGMA_P, lmax=LMAX,
             noise_std=0.0):
    """Evaluate model and MCMC baseline on the same test maps."""
    model.eval()
    cnn_errors = []
    mcmc_errors = []
    mcmc_times = []

    for maps, ell_p_true in dataloader:
        maps_np = maps.cpu().numpy()
        ell_p_true_np = ell_p_true.cpu().numpy()

        # CNN prediction
        maps_dev = maps.to(device)
        pred = model(maps_dev).cpu().numpy()

        for i in range(len(ell_p_true)):
            # CNN error
            err_cnn = abs(pred[i] - ell_p_true_np[i]) / ell_p_true_np[i] * 100
            cnn_errors.append(err_cnn)

            # MCMC baseline
            t0 = time.time()
            ell_p_mcmc = mcmc_estimate_ell_p(
                maps_np[i], nside=nside, sigma_p=sigma_p, lmax=lmax,
                noise_std=noise_std
            )
            dt = time.time() - t0
            mcmc_times.append(dt)

            err_mcmc = abs(ell_p_mcmc - ell_p_true_np[i]) / ell_p_true_np[i] * 100
            mcmc_errors.append(err_mcmc)

    return {
        "cnn_pct_error": np.mean(cnn_errors),
        "mcmc_pct_error": np.mean(mcmc_errors),
        "mcmc_time": np.mean(mcmc_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralCNN for Test 1 (v2 paper-matching)")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Noise standard deviation")
    parser.add_argument("--n_train", type=int, default=100000,
                        help="Number of training maps (paper: 100k)")
    parser.add_argument("--n_val", type=int, default=10000,
                        help="Number of validation maps (paper: 10k)")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Number of test maps (paper: 1k)")
    parser.add_argument("--max_epochs", type=int, default=150,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (paper: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (paper: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (paper: 1e-3)")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience (paper: 5)")
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help="ReduceLROnPlateau factor (paper: 0.1)")
    parser.add_argument("--hidden_channels", type=int, default=32,
                        help="Number of hidden channels in SpectralCNN")
    parser.add_argument("--num_blocks", type=int, default=3,
                        help="Number of spectral convolution blocks")
    parser.add_argument("--nside", type=int, default=NSIDE,
                        help="HEALPix Nside")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise σ_n = {args.noise_std}")
    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LR schedule: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Early stopping: patience={args.patience}")

    # Datasets
    train_ds = OnTheFlyDataset(args.n_train, nside=args.nside,
                               noise_std=args.noise_std, seed=42)
    val_ds = OnTheFlyDataset(args.n_val, nside=args.nside,
                             noise_std=args.noise_std, seed=123)
    test_ds = OnTheFlyDataset(args.n_test, nside=args.nside,
                              noise_std=args.noise_std, seed=999)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=0)

    # Model
    model = SpectralCNN(
        nside=args.nside,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training setup — paper-matching
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor
    )
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_reached = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'No Imp':>6}")
    print("-" * 55)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - t0

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for maps, ell_p_true in val_loader:
                maps = maps.to(device)
                ell_p_true = ell_p_true.to(device)
                pred = model(maps)
                val_loss += criterion(pred, ell_p_true).item() * len(ell_p_true)
                n_val += len(ell_p_true)
        val_loss /= max(n_val, 1)

        # ReduceLROnPlateau
        scheduler.step(val_loss)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_reached = epoch
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {lr:10.6f} | {epochs_no_improve:6d}  ({epoch_time:.1f}s)")

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\n*** Early stopping at epoch {epoch} (no improvement for {args.patience} epochs) ***")
            break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps (noise σ_n={args.noise_std}) ===")
    results = evaluate(model, test_loader, device, nside=args.nside,
                       noise_std=args.noise_std)

    print(f"\nCNN  mean % error: {results['cnn_pct_error']:.1f}%")
    print(f"MCMC mean % error: {results['mcmc_pct_error']:.1f}%")
    print(f"MCMC mean time:    {results['mcmc_time']:.3f}s per map")
    print(f"\nPaper baselines (Krachmalnicoff & Tomasi 2019, Table 1):")
    print(f"  NNhealpix: 1.3% (σ_n=0), 2.9% (σ_n=5), 5.2% (σ_n=10), 8.4% (σ_n=15)")
    print(f"  MCMC:      0.7% (σ_n=0), 2.5% (σ_n=5), 4.8% (σ_n=10), 7.8% (σ_n=15)")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "version": "v2_paper_matching",
                "noise_std": float(args.noise_std),
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
                "cnn_pct_error": float(results["cnn_pct_error"]),
                "mcmc_pct_error": float(results["mcmc_pct_error"]),
                "mcmc_time_per_map": float(results["mcmc_time"]),
                "n_params": int(n_params),
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
