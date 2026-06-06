#!/usr/bin/env python3
"""Evaluate best N128 checkpoints on test sets (jobs killed by walltime before this step)."""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_harmonics_healpix.data_generation_test4 import (
    generate_r_tau_map,
    precompute_camb_spectra_r_tau,
    R_MIN, R_MAX,
    TAU_MIN, TAU_MAX,
    N_CAMB_SPECTRA,
    R_LOG_EPSILON,
)
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN
from train_test4 import HDF5RTauDataset


def evaluate(model, dataloader, device):
    model.eval()
    r_preds, r_trues, tau_preds, tau_trues = [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            target_r_log = batch['target_r_log'].to(device)
            target_tau = batch['target_tau'].to(device)
            
            out = model(x)
            pred_r_log = out[:, 0:1]
            pred_tau = out[:, 1:2]
            
            r_pred = torch.exp(pred_r_log) - R_LOG_EPSILON
            r_true = torch.exp(target_r_log) - R_LOG_EPSILON
            
            r_preds.append(r_pred.cpu().numpy())
            r_trues.append(r_true.cpu().numpy())
            tau_preds.append(pred_tau.cpu().numpy())
            tau_trues.append(target_tau.cpu().numpy())
    
    r_preds = np.concatenate(r_preds)
    r_trues = np.concatenate(r_trues)
    tau_preds = np.concatenate(tau_preds)
    tau_trues = np.concatenate(tau_trues)
    
    r_err = np.abs(r_preds - r_trues) / (np.abs(r_trues) + 1e-10) * 100
    tau_err = np.abs(tau_preds - tau_trues) / (np.abs(tau_trues) + 1e-10) * 100
    
    r_bias = np.mean(r_preds - r_trues)
    
    # Percentile analysis
    def percentile_stats(pred, true, label):
        err = np.abs(pred - true) / (np.abs(true) + 1e-10) * 100
        return {
            f"{label}_p50": float(np.percentile(err, 50)),
            f"{label}_p68": float(np.percentile(err, 68)),
            f"{label}_p95": float(np.percentile(err, 95)),
            f"{label}_mean": float(np.mean(err)),
            f"{label}_median": float(np.median(err)),
        }
    
    stats = {
        "r_pct_error": float(np.mean(r_err)),
        "tau_pct_error": float(np.mean(tau_err)),
        "r_bias": float(r_bias),
        "r_std": float(np.std(r_preds - r_trues)),
        "tau_std": float(np.std(tau_preds - tau_trues)),
        **percentile_stats(r_preds, r_trues, "r"),
        **percentile_stats(tau_preds, tau_trues, "tau"),
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--hdf5_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nside', type=int, default=128)
    parser.add_argument('--lmax', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SpectralCNN(
        in_channels=3,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        nside=args.nside,
        lmax=args.lmax,
        num_blocks=args.num_blocks,
    )
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load test set
    ds = HDF5RTauDataset(args.hdf5_path, split='test')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Test set: {len(ds)} maps")
    
    # Evaluate
    t0 = time.time()
    results = evaluate(model, loader, device)
    elapsed = time.time() - t0
    
    print(f"\n=== Test set evaluation ({elapsed:.1f}s) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "hdf5_path": args.hdf5_path,
            "nside": args.nside,
            "n_params": n_params,
            "n_test": len(ds),
            **results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
