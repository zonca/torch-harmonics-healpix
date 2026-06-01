#!/usr/bin/env python3
"""Improved MCMC baseline for Test 4 (joint r/τ estimation).

Improvements over the original:
1. Two-stage grid search: coarse 100×100 → fine 50×50 around minimum
2. Quadratic interpolation for sub-grid resolution
3. Configurable ℓ_max for BB (BB signal only at low-ℓ, high-ℓ adds noise)
4. Fiducial-point evaluation (same as CNN) for apples-to-apples comparison
5. CAMB spectra on a REGULAR (r, τ) grid (not random sampling)
"""

import argparse
import numpy as np
import healpy as hp
import json
import time
import os
import psutil

from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau, R_MAX
)
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
from torch_harmonics_healpix.data_generation_test4 import generate_r_tau_map


def noise_arcmin_to_uK(noise_arcmin, nside):
    """Convert noise from μK-arcmin to μK per HEALPix pixel."""
    npix = hp.nside2npix(nside)
    pixel_area_rad2 = 4 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_rad2 * (180 * 60 / np.pi) ** 2
    return noise_arcmin / np.sqrt(pixel_area_arcmin2)


def chi2_grid_search(
    cl_ee_obs, cl_bb_obs,
    r_grid, tau_grid,
    cl_ee_array, cl_bb_array,
    noise_cl, f_sky, lmax,
    lmax_bb=None,
):
    """Compute chi-squared grid over (r, τ) and return best-fit via quadratic interpolation.

    Args:
        cl_ee_obs: Observed EE power spectrum (length lmax+1).
        cl_bb_obs: Observed BB power spectrum (length lmax+1).
        r_grid: 1D array of r values.
        tau_grid: 1D array of τ values.
        cl_ee_array: Pre-computed EE spectra, shape (n_tau * n_r, lmax+1).
            Ordered as flattened (tau_idx * n_r + r_idx).
        cl_bb_array: Pre-computed BB spectra, same shape convention.
        noise_cl: Noise power (scalar).
        f_sky: Sky fraction.
        lmax: Maximum multipole for EE.
        lmax_bb: Maximum multipole for BB (default: lmax).

    Returns:
        (r_best, tau_best, chi2_min) with quadratic interpolation.
    """
    if lmax_bb is None:
        lmax_bb = lmax

    ell = np.arange(lmax + 1)
    n_r = len(r_grid)
    n_tau = len(tau_grid)

    chi2_grid = np.full((n_tau, n_r), np.inf)

    for i in range(n_tau):
        for j in range(n_r):
            spec_idx = i * n_r + j
            cl_ee_model = cl_ee_array[spec_idx]
            cl_bb_model = cl_bb_array[spec_idx]

            # EE chi-squared: all multipoles
            sigma_ee2 = (2.0 / (2 * ell + 1)) * (cl_ee_model + noise_cl) ** 2 / f_sky
            sigma_ee2 = np.maximum(sigma_ee2, 1e-30)
            chi2_ee = np.sum((cl_ee_obs - cl_ee_model) ** 2 / sigma_ee2)

            # BB chi-squared: only up to lmax_bb
            ell_bb = np.arange(lmax_bb + 1)
            sigma_bb2 = (2.0 / (2 * ell_bb + 1)) * (cl_bb_model[:lmax_bb+1] + noise_cl) ** 2 / f_sky
            sigma_bb2 = np.maximum(sigma_bb2, 1e-30)
            chi2_bb = np.sum((cl_bb_obs[:lmax_bb+1] - cl_bb_model[:lmax_bb+1]) ** 2 / sigma_bb2)

            chi2_grid[i, j] = chi2_ee + chi2_bb

    # Find grid minimum
    best_flat = np.argmin(chi2_grid)
    best_tau_idx, best_r_idx = np.unravel_index(best_flat, chi2_grid.shape)
    chi2_min = chi2_grid[best_tau_idx, best_r_idx]

    r_best = float(r_grid[best_r_idx])
    tau_best = float(tau_grid[best_tau_idx])

    # Quadratic interpolation for sub-grid refinement
    if 1 <= best_r_idx <= n_r - 2:
        dr = r_grid[1] - r_grid[0]
        y_m = chi2_grid[best_tau_idx, best_r_idx - 1]
        y_0 = chi2_grid[best_tau_idx, best_r_idx]
        y_p = chi2_grid[best_tau_idx, best_r_idx + 1]
        denom = y_m - 2 * y_0 + y_p
        if abs(denom) > 1e-30:
            offset = 0.5 * (y_m - y_p) / denom
            offset = np.clip(offset, -0.5, 0.5)
            r_best = float(r_grid[best_r_idx] + offset * dr)

    if 1 <= best_tau_idx <= n_tau - 2:
        dtau = tau_grid[1] - tau_grid[0]
        y_m = chi2_grid[best_tau_idx - 1, best_r_idx]
        y_0 = chi2_grid[best_tau_idx, best_r_idx]
        y_p = chi2_grid[best_tau_idx + 1, best_r_idx]
        denom = y_m - 2 * y_0 + y_p
        if abs(denom) > 1e-30:
            offset = 0.5 * (y_m - y_p) / denom
            offset = np.clip(offset, -0.5, 0.5)
            tau_best = float(tau_grid[best_tau_idx] + offset * dtau)

    return r_best, tau_best, chi2_min


def precompute_spectra_on_grid(r_grid, tau_grid, lmax, cache_path=None):
    """Pre-compute CAMB spectra on a regular (r, τ) grid.

    Returns arrays of shape (n_tau * n_r, lmax+1).
    """
    from astropy.io import fits as pf

    n_r = len(r_grid)
    n_tau = len(tau_grid)
    n_spectra = n_r * n_tau

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached spectra from {cache_path}")
        with pf.open(cache_path) as hdul:
            cl_ee_array = np.array(hdul["CL_EE"].data, dtype=np.float64)
            cl_bb_array = np.array(hdul["CL_BB"].data, dtype=np.float64)
        print(f"Loaded {cl_ee_array.shape[0]} spectra from cache")
        return cl_ee_array, cl_bb_array

    print(f"Pre-computing {n_spectra} CAMB spectra on "
          f"{n_tau}×{n_r} regular grid...")
    cl_ee_array = np.zeros((n_spectra, lmax + 1), dtype=np.float64)
    cl_bb_array = np.zeros((n_spectra, lmax + 1), dtype=np.float64)

    t0 = time.time()
    for i in range(n_tau):
        for j in range(n_r):
            spec_idx = i * n_r + j
            cl_ee, cl_bb = generate_camb_spectra_r_tau(
                r_grid[j], tau_grid[i], lmax
            )
            cl_ee_array[spec_idx] = cl_ee
            cl_bb_array[spec_idx] = cl_bb
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (n_tau - i - 1)
        print(f"  τ={tau_grid[i]:.4f}: {i+1}/{n_tau} rows done "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"CAMB pre-computation: {total_time:.1f}s")

    if cache_path:
        hdu_ee = pf.ImageHDU(cl_ee_array); hdu_ee.name = "CL_EE"
        hdu_bb = pf.ImageHDU(cl_bb_array); hdu_bb.name = "CL_BB"
        hdul = pf.HDUList([pf.PrimaryHDU(), hdu_ee, hdu_bb])
        hdul.writeto(cache_path, overwrite=True)
        print(f"Saved cache to {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")

    return cl_ee_array, cl_bb_array


def main():
    parser = argparse.ArgumentParser(
        description="Improved MCMC baseline for Test 4 (joint r/τ estimation)"
    )
    parser.add_argument("--nside", type=int, default=128,
                        help="HEALPix NSIDE (default: 128)")
    parser.add_argument("--lmax", type=int, default=None,
                        help="Maximum multipole (default: 3*NSIDE-1)")
    parser.add_argument("--lmax_bb", type=int, default=10,
                        help="Maximum ℓ for BB spectrum (default: 10)")
    parser.add_argument("--results_dir", type=str,
                        default=os.environ.get("RESULTS_DIR", "results"),
                        help="Directory for output JSON files")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Number of test maps per config")
    parser.add_argument("--coarse_grid", type=int, default=100,
                        help="Coarse grid size per dimension (default: 100)")
    parser.add_argument("--fine_grid", type=int, default=50,
                        help="Fine grid size for refinement (default: 50)")
    parser.add_argument("--r_fiducial", type=float, default=0.003,
                        help="Fiducial r value (default: 0.003)")
    parser.add_argument("--tau_fiducial", type=float, default=0.054,
                        help="Fiducial τ value (default: 0.054)")
    parser.add_argument("--fiducial_only", action="store_true",
                        help="Only evaluate at fiducial point")
    args = parser.parse_args()

    if args.lmax is None:
        args.lmax = 3 * args.nside - 1

    nside = args.nside
    lmax = args.lmax
    lmax_bb = args.lmax_bb
    n_test = args.n_test
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        {"f_sky": 1.0, "noise_arcmin": 0, "label": "fsky1.0_noise0"},
        {"f_sky": 1.0, "noise_arcmin": 6, "label": "fsky1.0_noise6"},
        {"f_sky": 0.1, "noise_arcmin": 0, "label": "fsky0.1_noise0"},
        {"f_sky": 0.1, "noise_arcmin": 6, "label": "fsky0.1_noise6"},
    ]

    rng = np.random.default_rng(42)

    # Define regular (r, τ) grid
    r_grid = np.linspace(0, R_MAX, args.coarse_grid)
    tau_grid = np.linspace(0.03, 0.08, args.coarse_grid)

    # Pre-compute CAMB spectra on REGULAR grid
    from astropy.io import fits as pf
    camb_cache = os.path.join(results_dir,
        f"test4_camb_spectra_regular_nside{nside}_{args.coarse_grid}grid.fits")
    cl_ee_array, cl_bb_array = precompute_spectra_on_grid(
        r_grid, tau_grid, lmax, cache_path=camb_cache
    )

    # Build spectra interpolator for two-stage refinement
    cl_ee_2d = cl_ee_array.reshape(args.coarse_grid, args.coarse_grid, -1)
    cl_bb_2d = cl_bb_array.reshape(args.coarse_grid, args.coarse_grid, -1)

    from scipy.interpolate import RegularGridInterpolator

    def interp_spectra_to_grid(r_grid_new, tau_grid_new):
        """Interpolate CAMB spectra from coarse grid to a new (r, τ) grid."""
        n_r_new = len(r_grid_new)
        n_tau_new = len(tau_grid_new)
        r_mesh, tau_mesh = np.meshgrid(r_grid_new, tau_grid_new)
        pts = np.column_stack([tau_mesh.ravel(), r_mesh.ravel()])

        cl_ee_new = np.zeros((n_tau_new * n_r_new, lmax + 1))
        cl_bb_new = np.zeros((n_tau_new * n_r_new, lmax + 1))

        for ell_idx in range(lmax + 1):
            interp_ee = RegularGridInterpolator(
                (tau_grid, r_grid), cl_ee_2d[:, :, ell_idx],
                method='linear', bounds_error=False, fill_value=0.0,
            )
            interp_bb = RegularGridInterpolator(
                (tau_grid, r_grid), cl_bb_2d[:, :, ell_idx],
                method='linear', bounds_error=False, fill_value=0.0,
            )
            cl_ee_new[:, ell_idx] = interp_ee(pts)
            cl_bb_new[:, ell_idx] = interp_bb(pts)

        return cl_ee_new, cl_bb_new

    # Create shared masks
    shared_masks = {}
    for f_sky in [1.0, 0.1]:
        mask_rng = np.random.default_rng(0)
        shared_masks[f_sky] = create_sky_mask(f_sky, nside, mask_rng).astype(np.float32)

    process = psutil.Process(os.getpid())
    mem_peak = process.memory_info().rss / 1e9

    print(f"\nNSIDE = {nside}, LMAX = {lmax}, LMAX_BB = {lmax_bb}")
    print(f"Grid: {args.coarse_grid}×{args.coarse_grid} coarse + "
          f"{args.fine_grid}×{args.fine_grid} fine refinement + quadratic interp")
    print(f"Test maps per config: {n_test}")
    print(f"Fiducial point: r={args.r_fiducial}, τ={args.tau_fiducial}")

    for cfg in configs:
        f_sky = cfg["f_sky"]
        noise_uK = (
            noise_arcmin_to_uK(cfg["noise_arcmin"], nside)
            if cfg["noise_arcmin"] > 0 else 0.0
        )
        mask = shared_masks[f_sky]
        npix = hp.nside2npix(nside)
        noise_cl = noise_uK**2 * 4.0 * np.pi / npix if noise_uK > 0 else 0.0

        print(f"\n=== Config: {cfg['label']} ===")
        print(f"  f_sky={f_sky}, noise={cfg['noise_arcmin']} μK-arcmin "
              f"= {noise_uK:.4f} μK/pixel")

        r_preds = []
        tau_preds = []
        t_start = time.time()

        for i in range(n_test):
            # Generate map at fiducial point
            if args.fiducial_only:
                r_true = args.r_fiducial
                tau_true = args.tau_fiducial
            else:
                r_true = rng.uniform(0, R_MAX)
                tau_true = rng.uniform(0.03, 0.08)

            q, u, _ = generate_r_tau_map(
                r_true, tau_true, nside, lmax, noise_uK, f_sky, rng,
            )

            # Compute observed pseudo-C_ℓ from masked Q/U maps
            # De-bias by dividing by f_sky
            maps_in = np.array([np.zeros_like(q), q, u])
            cl_obs = hp.anafast(maps_in, lmax=lmax, pol=True)
            cl_ee_obs = cl_obs[1] / f_sky
            cl_bb_obs = cl_obs[2] / f_sky

            # Stage 1: Coarse grid search
            r_best, tau_best, _ = chi2_grid_search(
                cl_ee_obs, cl_bb_obs,
                r_grid, tau_grid,
                cl_ee_array, cl_bb_array,
                noise_cl, f_sky, lmax, lmax_bb,
            )

            # Stage 2: Fine refinement around coarse best
            dr = r_grid[1] - r_grid[0]
            dtau = tau_grid[1] - tau_grid[0]
            r_lo = max(0, r_best - 1.5 * dr)
            r_hi = min(R_MAX, r_best + 1.5 * dr)
            tau_lo = max(0.03, tau_best - 1.5 * dtau)
            tau_hi = min(0.08, tau_best + 1.5 * dtau)

            r_grid_fine = np.linspace(r_lo, r_hi, args.fine_grid)
            tau_grid_fine = np.linspace(tau_lo, tau_hi, args.fine_grid)

            cl_ee_fine, cl_bb_fine = interp_spectra_to_grid(r_grid_fine, tau_grid_fine)

            r_best, tau_best, _ = chi2_grid_search(
                cl_ee_obs, cl_bb_obs,
                r_grid_fine, tau_grid_fine,
                cl_ee_fine, cl_bb_fine,
                noise_cl, f_sky, lmax, lmax_bb,
            )

            r_preds.append(r_best)
            tau_preds.append(tau_best)

            if (i + 1) % 100 == 0:
                mem = process.memory_info().rss / 1e9
                mem_peak = max(mem_peak, mem)
                elapsed = time.time() - t_start
                print(f"  {i+1}/{n_test}: r_mean={np.mean(r_preds):.5f}, "
                      f"τ_mean={np.mean(tau_preds):.5f}, "
                      f"time={elapsed:.0f}s, mem={mem:.1f}GB")

        elapsed = time.time() - t_start
        mem = process.memory_info().rss / 1e9
        mem_peak = max(mem_peak, mem)

        r_preds = np.array(r_preds)
        tau_preds = np.array(tau_preds)

        # Compute fiducial-point statistics
        if args.fiducial_only:
            r_bias = float(np.mean(r_preds) - args.r_fiducial)
            tau_bias = float(np.mean(tau_preds) - args.tau_fiducial)
            sigma_r = float(np.std(r_preds))
            sigma_tau = float(np.std(tau_preds))
            rmse_r = float(np.sqrt(r_bias**2 + sigma_r**2))
            rmse_tau = float(np.sqrt(tau_bias**2 + sigma_tau**2))
            r_pct_error = float(rmse_r / args.r_fiducial * 100)
            tau_pct_error = float(rmse_tau / args.tau_fiducial * 100)
        else:
            r_bias = float(np.mean(r_preds))
            tau_bias = float(np.mean(tau_preds))
            sigma_r = float(np.std(r_preds))
            sigma_tau = float(np.std(tau_preds))
            rmse_r = sigma_r
            rmse_tau = sigma_tau
            r_pct_error = None
            tau_pct_error = None

        result_data = {
            "test": "test4_mcmc_improved",
            "config": cfg["label"],
            "f_sky": f_sky,
            "noise_arcmin": cfg["noise_arcmin"],
            "noise_uK_per_pixel": float(noise_uK),
            "n_test": n_test,
            "coarse_grid": args.coarse_grid,
            "fine_grid": args.fine_grid,
            "lmax_bb": lmax_bb,
            "r_mean": float(np.mean(r_preds)),
            "tau_mean": float(np.mean(tau_preds)),
            "r_bias": r_bias,
            "tau_bias": tau_bias,
            "sigma_r": sigma_r,
            "sigma_tau": sigma_tau,
            "rmse_r": rmse_r,
            "rmse_tau": rmse_tau,
            "r_pct_error": r_pct_error,
            "tau_pct_error": tau_pct_error,
            "r_fiducial": args.r_fiducial,
            "tau_fiducial": args.tau_fiducial,
            "cpu_time_seconds": elapsed,
            "peak_memory_GB": mem_peak,
            "method": "chi2_grid_search_two_stage",
            "nside": nside,
            "lmax": lmax,
        }

        outpath = os.path.join(results_dir, f"test4_mcmc_improved_{cfg['label']}.json")
        with open(outpath, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n  Results saved: {outpath}")
        if r_pct_error is not None:
            print(f"  r: mean={np.mean(r_preds):.5f}, bias={r_bias:.5f}, "
                  f"σ={sigma_r:.5f}, RMSE={rmse_r:.5f} ({r_pct_error:.1f}%)")
            print(f"  τ: mean={np.mean(tau_preds):.5f}, bias={tau_bias:.5f}, "
                  f"σ={sigma_tau:.5f}, RMSE={rmse_tau:.5f} ({tau_pct_error:.1f}%)")
        else:
            print(f"  r: mean={np.mean(r_preds):.5f}, σ={sigma_r:.5f}")
            print(f"  τ: mean={np.mean(tau_preds):.5f}, σ={sigma_tau:.5f}")
        print(f"  CPU time: {elapsed:.1f}s, Peak memory: {mem_peak:.1f}GB")

    print("\n=== All improved MCMC baselines complete ===")


if __name__ == "__main__":
    main()
