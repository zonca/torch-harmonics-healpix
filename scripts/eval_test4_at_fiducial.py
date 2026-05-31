#!/usr/bin/env python3
"""Evaluate SpectralCNN at the Fisher fiducial point for direct comparison.

Generates many noise realizations at (r=0.003, τ=0.054), runs the CNN
on each, and computes σ(r_pred) and σ(τ_pred). This gives the CNN's
uncertainty at the exact same point as the Fisher forecast, enabling
a fair apples-to-apples comparison.

Usage:
    python scripts/eval_test4_at_fiducial.py [--n_realizations 1000]
"""

import numpy as np
import torch
import json
import os
import time
import healpy as hp

from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau, generate_r_tau_map,
    R_MAX, R_LOG_EPSILON
)
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


def noise_arcmin_to_uK(noise_arcmin, nside):
    """Convert noise from μK-arcmin to μK per HEALPix pixel.

    σ_pix = σ_arcmin / sqrt(Ω_pix [arcmin²]), where Ω_pix = 4π/npix sr.
    """
    npix = hp.nside2npix(nside)
    pixel_area_rad2 = 4 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_rad2 * (180 * 60 / np.pi) ** 2
    return noise_arcmin / np.sqrt(pixel_area_arcmin2)


def eval_at_fiducial(model_path, r_fid=0.003, tau_fid=0.054,
                     f_sky=1.0, noise_arcmin=0, n_realizations=1000,
                     nside=16, lmax=47, seed=42):
    """Evaluate a trained SpectralCNN at a single fiducial (r, τ) point.

    Generates n_realizations maps with different noise seeds at the same
    (r_fid, tau_fid), runs the CNN, and returns the distribution of
    predictions. The standard deviation of predictions gives the CNN's
    σ(r) and σ(τ), directly comparable to the Fisher forecast.

    Args:
        model_path: Path to trained .pt model file.
        r_fid: Fiducial tensor-to-scalar ratio.
        tau_fid: Fiducial optical depth.
        f_sky: Sky fraction.
        noise_arcmin: Noise level in μK-arcmin.
        n_realizations: Number of noise realizations.
        nside: HEALPix NSIDE.
        lmax: Maximum multipole.
        seed: Random seed for reproducibility.

    Returns:
        dict with sigma_r, sigma_tau, r_pct_error, tau_pct_error, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    inpaint = f_sky < 1.0
    model = SpectralCNN(
        in_channels=3, out_channels=2,
        nside=nside, num_blocks=3,
        hidden_channels=32, inpaint=inpaint
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Compute noise per pixel
    noise_uK = noise_arcmin_to_uK(noise_arcmin, nside) if noise_arcmin > 0 else 0.0

    # Get CAMB spectra for fiducial point
    cl_ee, cl_bb = generate_camb_spectra_r_tau(r_fid, tau_fid, lmax)

    # Create mask (shared seed for consistency)
    mask_rng = np.random.default_rng(0)
    mask = create_sky_mask(f_sky, nside, mask_rng).astype(np.float32)

    # Generate realizations and predict
    rng = np.random.default_rng(seed)
    r_predictions = []
    tau_predictions = []

    print(f"  Generating {n_realizations} realizations at r={r_fid}, τ={tau_fid}")
    print(f"  f_sky={f_sky}, noise={noise_arcmin} μK-arcmin ({noise_uK:.4f} μK/pixel)")

    t0 = time.time()
    for i in range(n_realizations):
        # Generate map at fiducial point
        q, u, _ = generate_r_tau_map(
            r_fid, tau_fid, nside, lmax, noise_uK, f_sky, rng,
            cl_ee=cl_ee, cl_bb=cl_bb
        )

        # Prepare input tensor
        inp = np.stack([q, u, mask])
        inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(inp_tensor)

        r_pred_log = pred[0, 0].item()
        tau_pred = pred[0, 1].item()

        # Convert log-r back to r
        r_pred = np.exp(r_pred_log) - R_LOG_EPSILON

        r_predictions.append(r_pred)
        tau_predictions.append(tau_pred)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_realizations} ({elapsed:.1f}s)")

    r_predictions = np.array(r_predictions)
    tau_predictions = np.array(tau_predictions)

    # Compute statistics
    sigma_r = np.std(r_predictions)
    sigma_tau = np.std(tau_predictions)
    r_mean = np.mean(r_predictions)
    tau_mean = np.mean(tau_predictions)
    r_bias = r_mean - r_fid
    tau_bias = tau_mean - tau_fid

    elapsed = time.time() - t0

    result = {
        "test": "test4_cnn_at_fiducial",
        "r_fiducial": r_fid,
        "tau_fiducial": tau_fid,
        "f_sky": f_sky,
        "noise_arcmin": noise_arcmin,
        "noise_uK_per_pixel": float(noise_uK),
        "n_realizations": n_realizations,
        "sigma_r": float(sigma_r),
        "sigma_tau": float(sigma_tau),
        "r_pct_error": float(sigma_r / r_fid * 100),
        "tau_pct_error": float(sigma_tau / tau_fid * 100),
        "r_mean": float(r_mean),
        "tau_mean": float(tau_mean),
        "r_bias": float(r_bias),
        "tau_bias": float(tau_bias),
        "nside": nside,
        "lmax": lmax,
        "device": str(device),
        "eval_time_s": elapsed,
    }

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate SpectralCNN at Fisher fiducial point"
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory with .pt model files and output JSONs")
    parser.add_argument("--n_realizations", type=int, default=1000,
                        help="Number of noise realizations per config")
    parser.add_argument("--nside", type=int, default=16,
                        help="HEALPix NSIDE (default: 16)")
    parser.add_argument("--lmax", type=int, default=None,
                        help="Maximum multipole (default: 3*NSIDE-1)")
    args = parser.parse_args()

    if args.lmax is None:
        args.lmax = 3 * args.nside - 1

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        {"f_sky": 1.0, "noise_arcmin": 0, "label": "fsky1.0_noise0"},
        {"f_sky": 1.0, "noise_arcmin": 6, "label": "fsky1.0_noise6"},
        {"f_sky": 0.1, "noise_arcmin": 0, "label": "fsky0.1_noise0"},
        {"f_sky": 0.1, "noise_arcmin": 6, "label": "fsky0.1_noise6"},
    ]

    r_fid = 0.003
    tau_fid = 0.054

    all_results = {}

    for cfg in configs:
        label = cfg["label"]
        # Try multiple naming conventions (checkpoint, HDF5, nside-specific, legacy)
        candidates = [
            os.path.join(results_dir, f"test4_nside128_hdf5_{label}_nside{args.nside}_best.pt"),
            os.path.join(results_dir, f"test4_nside128_hdf5_{label}_nside{args.nside}.pt"),
            os.path.join(results_dir, f"test4_{label}_nside{args.nside}.pt"),
            os.path.join(results_dir, f"test4_{label}.pt"),
        ]
        model_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if model_path is None:
            print(f"Skipping {label}: model not found (tried {candidates})")
            continue

        print(f"\n=== Config: {label} ===")
        result = eval_at_fiducial(
            model_path=model_path,
            r_fid=r_fid,
            tau_fid=tau_fid,
            f_sky=cfg["f_sky"],
            noise_arcmin=cfg["noise_arcmin"],
            n_realizations=args.n_realizations,
            nside=args.nside,
            lmax=args.lmax,
        )

        outpath = os.path.join(results_dir, f"test4_cnn_fiducial_{label}.json")
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  σ(r) = {result['sigma_r']:.6f}  →  {result['r_pct_error']:.1f}%")
        print(f"  σ(τ) = {result['sigma_tau']:.6f}  →  {result['tau_pct_error']:.1f}%")
        print(f"  r_bias = {result['r_bias']:.6f},  τ_bias = {result['tau_bias']:.6f}")
        print(f"  Saved to {outpath}")

        all_results[label] = result

    print("\n" + "=" * 60)
    print("Summary: CNN at Fisher fiducial point (r=0.003, τ=0.054)")
    print("=" * 60)
    print(f"{'Config':<20} {'σ(r)':>10} {'r %err':>10} {'σ(τ)':>10} {'τ %err':>10}")
    print("-" * 60)
    for label, r in all_results.items():
        print(f"{label:<20} {r['sigma_r']:>10.6f} {r['r_pct_error']:>10.1f}% "
              f"{r['sigma_tau']:>10.6f} {r['tau_pct_error']:>10.1f}%")


if __name__ == "__main__":
    main()
