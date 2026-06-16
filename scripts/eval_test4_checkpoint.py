#!/usr/bin/env python3
"""Evaluate a saved SpectralCNN checkpoint on the HDF5 test set.

Usage:
    python scripts/eval_test4_checkpoint.py \
        --checkpoint results_v3/test4_nside128_hc32_fsky1.0_noise6_nside128_best.pt \
        --hdf5_path hdf5_striped/test4_nside128_fsky1.0_noise6.h5 \
        --output results_v3/test4_nside128_hc32_fsky1.0_noise6.json \
        --nside 128 --f_sky 1.0 --noise_std 6 --hidden_channels 32
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.utils.data as data

from torch_harmonics_healpix.data_generation_test4 import R_LOG_EPSILON
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


class HDF5RTauDataset(data.Dataset):
    """Minimal HDF5 dataset for evaluation only."""

    def __init__(self, hdf5_path, split='test'):
        import h5py
        self.f = h5py.File(hdf5_path, 'r')
        self.maps = self.f[f'{split}/maps']
        self.targets = self.f[f'{split}/targets']

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return (self.maps[idx].astype(np.float32),
                self.targets[idx].astype(np.float32))

    def __del__(self):
        self.f.close()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate joint r/τ estimation (same as train_test4.py)."""
    model.eval()
    r_pct_errors = []
    tau_pct_errors = []
    r_biases = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        r_log_true = batch_y[:, 0]
        tau_true = batch_y[:, 1]

        r_pred = torch.exp(pred[:, 0]) - R_LOG_EPSILON
        r_true = torch.exp(r_log_true) - R_LOG_EPSILON
        tau_pred = pred[:, 1]
        tau_true_f = tau_true

        r_mask = r_true > 0.001
        if r_mask.any():
            r_pct = (r_pred[r_mask] - r_true[r_mask]).abs() / r_true[r_mask] * 100
            r_pct_errors.extend(r_pct.tolist())

        r_nearzero_mask = r_true < 0.001
        if r_nearzero_mask.any():
            r_bias = (r_pred[r_nearzero_mask] - r_true[r_nearzero_mask]).mean()
            r_biases.append(r_bias.item())

        tau_denom = tau_true_f.abs().clamp(min=0.01)
        tau_pct = (tau_pred - tau_true_f).abs() / tau_denom * 100
        tau_pct_errors.extend(tau_pct.tolist())

    return {
        "r_pct_error": float(np.mean(r_pct_errors)) if r_pct_errors else float("inf"),
        "tau_pct_error": float(np.mean(tau_pct_errors)) if tau_pct_errors else float("inf"),
        "r_bias": float(np.mean(r_biases)) if r_biases else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpectralCNN checkpoint on HDF5 test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to _best.pt checkpoint")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--f_sky", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inpaint = args.f_sky < 1.0

    # Load model
    model = SpectralCNN(
        in_channels=3, out_channels=2,
        nside=args.nside, num_blocks=args.num_blocks,
        hidden_channels=args.hidden_channels, inpaint=inpaint
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model params: {n_params:,}")

    # Load test set from HDF5
    test_dataset = HDF5RTauDataset(args.hdf5_path, split='test')
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_dataset)} maps")

    # Evaluate
    t0 = time.time()
    results = evaluate(model, test_loader, device)
    eval_time = time.time() - t0

    print(f"\n=== Results ===")
    print(f"  r mean % error: {results['r_pct_error']:.1f}%")
    print(f"  τ mean % error: {results['tau_pct_error']:.1f}%")
    print(f"  r bias: {results['r_bias']:.6f}")
    print(f"  Eval time: {eval_time:.1f}s")

    # Save JSON (same format as train_test4.py)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "test": "test4_joint_r_tau",
            "data_mode": "hdf5",
            "hdf5_path": args.hdf5_path,
            "r_pct_error": results["r_pct_error"],
            "tau_pct_error": results["tau_pct_error"],
            "r_bias": results["r_bias"],
            "n_params": n_params,
            "noise_std_uK_arcmin": args.noise_std,
            "f_sky": args.f_sky,
            "n_test": len(test_dataset),
            "epochs_reached": "eval_only",
            "hidden_channels": args.hidden_channels,
            "num_blocks": args.num_blocks,
            "batch_size": args.batch_size,
            "nside": args.nside,
            "checkpoint": args.checkpoint,
            "eval_time_s": eval_time,
        }, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
