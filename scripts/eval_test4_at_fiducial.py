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
                     nside=16, lmax=47, seed=42,
                     num_blocks=3, hidden_channels=32, mask=None):
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
        nside=nside, num_blocks=num_blocks,
        hidden_channels=hidden_channels, inpaint=inpaint
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Compute noise per pixel
    noise_uK = noise_arcmin_to_uK(noise_arcmin, nside) if noise_arcmin > 0 else 0.0

    # Get CAMB spectra for fiducial point
    cl_ee, cl_bb = generate_camb_spectra_r_tau(r_fid, tau_fid, lmax)

    # Use the provided mask, or recreate the training mask (shared seed 0,
    # matching train_test4.py's on-the-fly mode). For HDF5-trained models
    # the mask MUST be loaded from the HDF5 file (generated with a
    # different seed) — pass it via the `mask` argument.
    if mask is None:
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
            cl_ee=cl_ee, cl_bb=cl_bb, mask=mask
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

    rmse_r = np.sqrt(sigma_r**2 + r_bias**2)
    rmse_tau = np.sqrt(sigma_tau**2 + tau_bias**2)

    result = {
        "test": "test4_cnn_at_fiducial",
        "model_path": os.path.basename(model_path),
        "r_fiducial": r_fid,
        "tau_fiducial": tau_fid,
        "f_sky": f_sky,
        "noise_arcmin": noise_arcmin,
        "noise_uK_per_pixel": float(noise_uK),
        "n_realizations": n_realizations,
        "sigma_r": float(sigma_r),
        "sigma_tau": float(sigma_tau),
        "rmse_r": float(rmse_r),
        "rmse_tau": float(rmse_tau),
        "r_pct_error": float(sigma_r / r_fid * 100),
        "tau_pct_error": float(sigma_tau / tau_fid * 100),
        "rmse_r_pct": float(rmse_r / r_fid * 100),
        "rmse_tau_pct": float(rmse_tau / tau_fid * 100),
        "r_mean": float(r_mean),
        "tau_mean": float(tau_mean),
        "r_bias": float(r_bias),
        "tau_bias": float(tau_bias),
        "nside": nside,
        "lmax": lmax,
        "num_blocks": num_blocks,
        "hidden_channels": hidden_channels,
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
    parser.add_argument("--num_blocks", type=int, default=3,
                        help="Number of spectral convolution blocks")
    parser.add_argument("--hidden_channels", type=int, default=32,
                        help="Number of hidden channels in spectral blocks")
    parser.add_argument("--r_fid", type=float, default=0.003,
                        help="Fiducial tensor-to-scalar ratio (default: 0.003)")
    parser.add_argument("--tau_fid", type=float, default=0.054,
                        help="Fiducial optical depth (default: 0.054)")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="Directory with .pt checkpoints (default: results_dir)")
    parser.add_argument("--hdf5_mask_pattern", type=str, default=None,
                        help="HDF5 path pattern with {label} placeholder; if set, "
                             "load the shared mask from the file's /mask dataset "
                             "(required for HDF5-trained models, e.g. NSIDE=128)")
    parser.add_argument("--mask_seed", type=int, default=0,
                        help="Seed for reconstructing the shared mask when no "
                             "HDF5 is available (0 = on-the-fly training mask; "
                             "42 = generate_test4_hdf5.py mask, verified "
                             "identical to the HDF5 /mask dataset)")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="Output JSON prefix (default: test4_cnn_fiducial_"
                             "nside{nside}_hc{hc}[_r{r_fid}_tau{tau_fid}])")
    args = parser.parse_args()

    if args.lmax is None:
        args.lmax = 3 * args.nside - 1

    results_dir = args.results_dir
    models_dir = args.models_dir or results_dir
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        {"f_sky": 1.0, "noise_arcmin": 0, "label": "fsky1.0_noise0"},
        {"f_sky": 1.0, "noise_arcmin": 6, "label": "fsky1.0_noise6"},
        {"f_sky": 0.1, "noise_arcmin": 0, "label": "fsky0.1_noise0"},
        {"f_sky": 0.1, "noise_arcmin": 6, "label": "fsky0.1_noise6"},
    ]

    r_fid = args.r_fid
    tau_fid = args.tau_fid
    ns, hc = args.nside, args.hidden_channels

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = f"test4_cnn_fiducial_nside{ns}_hc{hc}"
        if (r_fid, tau_fid) != (0.003, 0.054):
            out_prefix += f"_r{r_fid}_tau{tau_fid}"

    all_results = {}

    for cfg in configs:
        label = cfg["label"]
        # Try multiple naming conventions (v3 hc-tagged, v3, HDF5, legacy)
        candidates = [
            os.path.join(models_dir, f"test4_nside{ns}_hc{hc}_{label}_nside{ns}_best.pt"),
            os.path.join(models_dir, f"test4_nside{ns}_hc{hc}_{label}_nside{ns}.pt"),
            os.path.join(models_dir, f"test4_nside{ns}_{label}_nside{ns}_best.pt"),
            os.path.join(models_dir, f"test4_nside{ns}_{label}_nside{ns}.pt"),
            os.path.join(models_dir, f"test4_nside128_hdf5_{label}_nside{ns}_best.pt"),
            os.path.join(models_dir, f"test4_nside128_hdf5_{label}_nside{ns}.pt"),
            os.path.join(models_dir, f"test4_{label}_nside{ns}.pt"),
            os.path.join(models_dir, f"test4_{label}.pt"),
        ]
        model_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if model_path is None:
            print(f"Skipping {label}: model not found (tried {candidates})")
            continue

        mask = None
        if args.hdf5_mask_pattern:
            import h5py
            h5path = args.hdf5_mask_pattern.format(label=label)
            with h5py.File(h5path, "r") as h5:
                mask = h5["mask"][:].astype(np.float32)
            print(f"  Loaded shared mask from {h5path} (f_sky={mask.mean():.3f})")
        elif args.mask_seed != 0:
            mask_rng = np.random.default_rng(args.mask_seed)
            mask = create_sky_mask(cfg["f_sky"], args.nside, mask_rng).astype(np.float32)
            print(f"  Reconstructed mask with seed {args.mask_seed} "
                  f"(f_sky={mask.mean():.3f})")

        print(f"\n=== Config: {label} (model: {os.path.basename(model_path)}) ===")
        result = eval_at_fiducial(
            model_path=model_path,
            r_fid=r_fid,
            tau_fid=tau_fid,
            f_sky=cfg["f_sky"],
            noise_arcmin=cfg["noise_arcmin"],
            n_realizations=args.n_realizations,
            nside=args.nside,
            lmax=args.lmax,
            num_blocks=args.num_blocks,
            hidden_channels=args.hidden_channels,
            mask=mask,
        )

        outpath = os.path.join(results_dir, f"{out_prefix}_{label}.json")
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  σ(r) = {result['sigma_r']:.6f}  →  {result['r_pct_error']:.1f}%")
        print(f"  σ(τ) = {result['sigma_tau']:.6f}  →  {result['tau_pct_error']:.1f}%")
        print(f"  r_bias = {result['r_bias']:.6f},  τ_bias = {result['tau_bias']:.6f}")
        print(f"  Saved to {outpath}")

        all_results[label] = result

    print("\n" + "=" * 78)
    print(f"Summary: CNN at fiducial point (r={r_fid}, τ={tau_fid}), "
          f"NSIDE={ns}, hc={hc}")
    print("=" * 78)
    print(f"{'Config':<20} {'σ(r)':>10} {'bias(r)':>10} {'RMSE(r)':>10} "
          f"{'σ(τ)':>10} {'bias(τ)':>10} {'RMSE(τ)':>10}")
    print("-" * 78)
    for label, r in all_results.items():
        print(f"{label:<20} {r['sigma_r']:>10.6f} {r['r_bias']:>10.6f} "
              f"{r['rmse_r']:>10.6f} {r['sigma_tau']:>10.6f} "
              f"{r['tau_bias']:>10.6f} {r['rmse_tau']:>10.6f}")


if __name__ == "__main__":
    main()
