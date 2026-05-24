#!/usr/bin/env python3
"""MCMC baselines for Test 4 (joint r/τ estimation).

Runs chi-squared grid search on 1000 test maps per configuration.
Saves results as JSON files and tracks CPU time + peak memory usage.
"""

import numpy as np
import healpy as hp
import json
import time
import os
import psutil

from torch_harmonics_healpix.data_generation_test4 import (
    precompute_camb_spectra_r_tau, generate_r_tau_map, NSIDE, LMAX, R_MAX
)
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
from torch_harmonics_healpix.mcmc_baselines_test2_3 import mcmc_baseline_r_tau

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def noise_arcmin_to_uK(noise_arcmin, nside):
    """Convert noise from μK-arcmin to μK per HEALPix pixel.

    σ_pix = σ_arcmin / sqrt(Ω_pix [arcmin²]), where Ω_pix = 4π/npix sr.

    Args:
        noise_arcmin: White noise level in μK-arcmin.
        nside: HEALPix Nside parameter.

    Returns:
        Noise standard deviation in μK per pixel.
    """
    npix = hp.nside2npix(nside)
    pixel_area_rad2 = 4 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_rad2 * (180 * 60 / np.pi) ** 2
    return noise_arcmin / np.sqrt(pixel_area_arcmin2)


configs = [
    {"f_sky": 1.0, "noise_arcmin": 0, "label": "fsky1.0_noise0"},
    {"f_sky": 1.0, "noise_arcmin": 6, "label": "fsky1.0_noise6"},
    {"f_sky": 0.1, "noise_arcmin": 0, "label": "fsky0.1_noise0"},
    {"f_sky": 0.1, "noise_arcmin": 6, "label": "fsky0.1_noise6"},
]

n_test = 1000
rng = np.random.default_rng(42)

# Pre-compute CAMB spectra (with FITS cache)
from astropy.io import fits as pf

camb_cache = os.path.join(RESULTS_DIR, "test4_camb_spectra.fits")
if os.path.exists(camb_cache):
    print(f"Loading cached CAMB spectra from {camb_cache}")
    with pf.open(camb_cache) as hdul:
        r_values = np.array(hdul["R_VALUES"].data, dtype=np.float32)
        tau_values = np.array(hdul["TAU_VALUES"].data, dtype=np.float32)
        cl_ee_array = np.array(hdul["CL_EE"].data, dtype=np.float64)
        cl_bb_array = np.array(hdul["CL_BB"].data, dtype=np.float64)
    print(f"Loaded {len(r_values)} spectra from cache")
else:
    print("Pre-computing 5000 CAMB spectra...")
    t0 = time.time()
    r_values, tau_values, cl_ee_array, cl_bb_array = precompute_camb_spectra_r_tau(
        5000, LMAX, seed=12345
    )
    camb_time = time.time() - t0
    print(f"CAMB pre-computation: {camb_time:.1f}s")
    # Save to FITS for future runs
    hdu_r = pf.ImageHDU(r_values)
    hdu_r.name = "R_VALUES"
    hdu_tau = pf.ImageHDU(tau_values)
    hdu_tau.name = "TAU_VALUES"
    hdu_ee = pf.ImageHDU(cl_ee_array)
    hdu_ee.name = "CL_EE"
    hdu_bb = pf.ImageHDU(cl_bb_array)
    hdu_bb.name = "CL_BB"
    hdul = pf.HDUList([pf.PrimaryHDU(), hdu_r, hdu_tau, hdu_ee, hdu_bb])
    hdul.writeto(camb_cache, overwrite=True)
    print(f"Saved CAMB cache to {camb_cache} ({os.path.getsize(camb_cache)/1e6:.1f} MB)")

# Create shared masks
shared_masks = {}
for f_sky in [1.0, 0.1]:
    mask_rng = np.random.default_rng(0)
    shared_masks[f_sky] = create_sky_mask(f_sky, NSIDE, mask_rng).astype(np.float32)

# Grid for MCMC search (50 points each — coarse but sufficient for benchmark comparison)
r_grid = np.linspace(0, R_MAX, 50)
tau_grid = np.linspace(0.03, 0.08, 50)

# Memory tracking
process = psutil.Process(os.getpid())
mem_peak = process.memory_info().rss / 1e9

for cfg in configs:
    f_sky = cfg["f_sky"]
    noise_uK = (
        noise_arcmin_to_uK(cfg["noise_arcmin"], NSIDE)
        if cfg["noise_arcmin"] > 0
        else 0.0
    )
    mask = shared_masks[f_sky]

    print(f"")
    print(f"=== Config: {cfg['label']} ===")
    print(f"  f_sky={f_sky}, noise={cfg['noise_arcmin']} μK-arcmin = {noise_uK:.2f} μK/pixel")

    r_errors = []
    tau_errors = []
    t_start = time.time()

    for i in range(n_test):
        idx = rng.integers(0, len(r_values))
        r_true = r_values[idx]
        tau_true = tau_values[idx]

        q, u, _ = generate_r_tau_map(
            r_true, tau_true, NSIDE, LMAX, noise_uK, f_sky, rng,
            cl_ee=cl_ee_array[idx], cl_bb=cl_bb_array[idx]
        )

        result = mcmc_baseline_r_tau(
            q, u, mask, r_grid, tau_grid,
            cl_ee_array, cl_bb_array,
            noise_std=noise_uK, nside=NSIDE, lmax=LMAX
        )

        r_pct = abs(result["r_best"] - r_true) / max(r_true, 0.001) * 100
        tau_pct = abs(result["tau_best"] - tau_true) / tau_true * 100
        r_errors.append(r_pct)
        tau_errors.append(tau_pct)

        if (i + 1) % 100 == 0:
            mem = process.memory_info().rss / 1e9
            mem_peak = max(mem_peak, mem)
            print(f"  {i+1}/{n_test}: r={np.mean(r_errors):.1f}%, τ={np.mean(tau_errors):.1f}%, mem={mem:.1f}GB")

    elapsed = time.time() - t_start
    mem = process.memory_info().rss / 1e9
    mem_peak = max(mem_peak, mem)

    result_data = {
        "test": "test4_mcmc",
        "config": cfg["label"],
        "f_sky": f_sky,
        "noise_arcmin": cfg["noise_arcmin"],
        "noise_uK_per_pixel": float(noise_uK),
        "n_test": n_test,
        "r_pct_error": float(np.mean(r_errors)),
        "tau_pct_error": float(np.mean(tau_errors)),
        "r_pct_error_std": float(np.std(r_errors)),
        "tau_pct_error_std": float(np.std(tau_errors)),
        "cpu_time_seconds": elapsed,
        "peak_memory_GB": mem_peak,
        "method": "chi2_grid_search",
        "r_grid_size": len(r_grid),
        "tau_grid_size": len(tau_grid),
    }

    outpath = os.path.join(RESULTS_DIR, f"test4_mcmc_{cfg['label']}.json")
    with open(outpath, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved: {outpath}")
    print(f"  r: {np.mean(r_errors):.1f}% ± {np.std(r_errors):.1f}%")
    print(f"  τ: {np.mean(tau_errors):.1f}% ± {np.std(tau_errors):.1f}%")
    print(f"  CPU time: {elapsed:.1f}s, Peak memory: {mem_peak:.1f}GB")

print("")
print("=== All MCMC baselines complete ===")
