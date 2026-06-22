#!/usr/bin/env python3
"""Fisher matrix and MCMC baseline for Test 4 (joint r, τ estimation).

Computes:
1. Fisher information matrix for (r, τ) at fiducial point
2. Cramér-Rao lower bounds (σ_Fisher) for both NSIDE=16 and 128
3. Proper MCMC with Metropolis-Hastings sampling, including
   lensing BB as a nuisance parameter (A_lens)

All CPU work — runs on Popeye.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import healpy as hp

# Add project src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau,
    PLANCK_PARAMS, R_MAX, TAU_MIN, TAU_MAX,
)


# ── Fisher Matrix ──────────────────────────────────────────────────────

# Fixed high lmax for CAMB calculations. CAMB's set_for_lmax() changes
# internal accuracy parameters (max_eta_k, etc.), producing *different*
# C_ℓ values at the same ℓ depending on the lmax setting. To make Fisher
# bounds comparable across NSIDEs, all spectra must be computed with the
# same CAMB configuration. We use a fixed high lmax and truncate.
LMAX_CALC_FIXED = 500


def _generate_spectra_fixed_lmax(r, tau, lmax_out, lmax_calc=LMAX_CALC_FIXED):
    """Generate CAMB spectra at fixed high lmax, then truncate to lmax_out.

    This ensures consistent C_ℓ values across different output lmax values,
    which is essential for comparing Fisher bounds across NSIDEs.
    """
    cl_ee, cl_bb = generate_camb_spectra_r_tau(r, tau, lmax_calc)
    return cl_ee[: lmax_out + 1], cl_bb[: lmax_out + 1]


def compute_fisher_matrix(r_fid, tau_fid, lmax, f_sky, noise_uK=None, nside=None,
                          lmax_calc=LMAX_CALC_FIXED):
    """Compute Fisher matrix for (r, τ) at fiducial point.

    Uses the Cramér-Rao bound: F_ij = Σ_ℓ (2ℓ+1) f_sky / 2
        × Tr[C_ℓ^{-1} ∂C_ℓ/∂θ_i C_ℓ^{-1} ∂C_ℓ/∂θ_j]

    For diagonal covariance (no TE cross-correlation in the fit):
        F_ij = Σ_ℓ (2ℓ+1) f_sky / 2
            × [ (∂C_ℓ^EE/∂θ_i)(∂C_ℓ^EE/∂θ_j) / (C_ℓ^EE + N_ℓ)²
              + (∂C_ℓ^BB/∂θ_i)(∂C_ℓ^BB/∂θ_j) / (C_ℓ^BB + N_ℓ)² ]

    where N_ℓ is the noise power spectrum.

    Args:
        lmax_calc: Fixed lmax for CAMB calculation (ensures consistent
            C_ℓ across NSIDEs). Default 500.
    """
    # Fiducial spectra — computed at fixed high lmax, truncated to lmax
    cl_ee_fid, cl_bb_fid = _generate_spectra_fixed_lmax(
        r_fid, tau_fid, lmax, lmax_calc
    )

    # Noise power spectrum
    if noise_uK is not None and nside is not None:
        npix = hp.nside2npix(nside)
        N_ell = noise_uK**2 * 4.0 * np.pi / npix
    else:
        N_ell = 0.0

    # Total (signal + noise) spectra
    cl_ee_tot = cl_ee_fid + N_ell
    cl_bb_tot = cl_bb_fid + N_ell

    # Numerical derivatives with step sizes
    dr = max(r_fid * 1e-3, 1e-6)
    dtau = max(tau_fid * 1e-3, 1e-5)

    # ∂C_ℓ/∂r via finite differences
    cl_ee_rp, cl_bb_rp = _generate_spectra_fixed_lmax(
        r_fid + dr, tau_fid, lmax, lmax_calc
    )
    cl_ee_rm, cl_bb_rm = _generate_spectra_fixed_lmax(
        max(r_fid - dr, 0), tau_fid, lmax, lmax_calc
    )
    dcl_ee_dr = (cl_ee_rp - cl_ee_rm) / (2 * dr if r_fid > dr else dr)
    dcl_bb_dr = (cl_bb_rp - cl_bb_rm) / (2 * dr if r_fid > dr else dr)

    # ∂C_ℓ/∂τ via finite differences
    cl_ee_tp, cl_bb_tp = _generate_spectra_fixed_lmax(
        r_fid, tau_fid + dtau, lmax, lmax_calc
    )
    cl_ee_tm, cl_bb_tm = _generate_spectra_fixed_lmax(
        r_fid, max(tau_fid - dtau, 0.01), lmax, lmax_calc
    )
    dcl_ee_dtau = (cl_ee_tp - cl_ee_tm) / (2 * dtau)
    dcl_bb_dtau = (cl_bb_tp - cl_bb_tm) / (2 * dtau)

    # Fisher matrix elements — skip ℓ where total C_ℓ = 0 (e.g. BB at ℓ=0,1)
    ell = np.arange(lmax + 1)
    weight = (2 * ell + 1) * f_sky / 2.0

    # Mask: only include multipoles where both EE and BB have signal
    valid_ee = cl_ee_tot > 0
    valid_bb = cl_bb_tot > 0

    F_rr = np.sum(weight[valid_ee] * (dcl_ee_dr[valid_ee] / cl_ee_tot[valid_ee])**2) + \
           np.sum(weight[valid_bb] * (dcl_bb_dr[valid_bb] / cl_bb_tot[valid_bb])**2)
    F_tt = np.sum(weight[valid_ee] * (dcl_ee_dtau[valid_ee] / cl_ee_tot[valid_ee])**2) + \
           np.sum(weight[valid_bb] * (dcl_bb_dtau[valid_bb] / cl_bb_tot[valid_bb])**2)
    F_rt = np.sum(weight[valid_ee] * dcl_ee_dr[valid_ee] * dcl_ee_dtau[valid_ee] / cl_ee_tot[valid_ee]**2) + \
           np.sum(weight[valid_bb] * dcl_bb_dr[valid_bb] * dcl_bb_dtau[valid_bb] / cl_bb_tot[valid_bb]**2)

    F = np.array([[F_rr, F_rt], [F_rt, F_tt]])
    return F


# ── MCMC with Metropolis-Hastings ─────────────────────────────────────

def log_likelihood(cl_ee_obs, cl_bb_obs, cl_ee_model, cl_bb_model,
                   noise_cl, f_sky, lmax_ee=None, lmax_bb=None):
    """Gaussian log-likelihood for observed vs model C_ℓ.

    L = -1/2 Σ_ℓ (2ℓ+1) f_sky / 2
        × [ (Ĉ_ℓ^EE - C_ℓ^EE)² / Var(C_ℓ^EE)
          + (Ĉ_ℓ^BB - C_ℓ^BB)² / Var(C_ℓ^BB) ]

    where Var(C_ℓ) = 2/(2ℓ+1)/f_sky × (C_ℓ + N_ℓ)²  (cosmic + noise variance).
    """
    if lmax_ee is None:
        lmax_ee = len(cl_ee_obs) - 1
    if lmax_bb is None:
        lmax_bb = len(cl_bb_obs) - 1

    ell = np.arange(max(lmax_ee, lmax_bb) + 1)

    # EE contribution
    ee_slice = slice(0, lmax_ee + 1)
    sig_ee = cl_ee_model[ee_slice] + noise_cl
    valid_ee = sig_ee > 0
    # Variance = 2/(2ℓ+1)/f_sky * sig²  →  weight = (2ℓ+1)*f_sky/2 / sig²
    var_ee = 2.0 / ((2 * ell[ee_slice] + 1) * f_sky) * sig_ee**2
    chi2_ee = np.sum((cl_ee_obs[ee_slice][valid_ee] - cl_ee_model[ee_slice][valid_ee])**2
                     / var_ee[valid_ee])

    # BB contribution
    bb_slice = slice(0, lmax_bb + 1)
    sig_bb = cl_bb_model[bb_slice] + noise_cl
    valid_bb = sig_bb > 0
    var_bb = 2.0 / ((2 * ell[bb_slice] + 1) * f_sky) * sig_bb**2
    chi2_bb = np.sum((cl_bb_obs[bb_slice][valid_bb] - cl_bb_model[bb_slice][valid_bb])**2
                     / var_bb[valid_bb])

    return -0.5 * (chi2_ee + chi2_bb)


def precompute_spectral_grid(lmax, n_r=50, n_tau=50):
    """Pre-compute CAMB C_ℓ on a regular (r, τ) grid for fast interpolation.

    Returns: r_grid, tau_grid, cl_ee_grid[n_tau, n_r, lmax+1], cl_bb_grid[...]
    Also returns lensing BB template (at r=0, tau=0.054).
    """
    r_grid = np.linspace(0, R_MAX, n_r)
    tau_grid = np.linspace(TAU_MIN, TAU_MAX, n_tau)

    cl_ee_grid = np.zeros((n_tau, n_r, lmax + 1))
    cl_bb_grid = np.zeros((n_tau, n_r, lmax + 1))

    print(f"  Pre-computing {n_r}×{n_tau} CAMB grid for MCMC...")
    t0 = time.time()
    for i, tau in enumerate(tau_grid):
        for j, r in enumerate(r_grid):
            try:
                cl_ee, cl_bb = generate_camb_spectra_r_tau(r, tau, lmax)
                cl_ee_grid[i, j] = cl_ee
                cl_bb_grid[i, j] = cl_bb
            except Exception:
                cl_ee_grid[i, j] = 0
                cl_bb_grid[i, j] = 0
        if (i + 1) % 10 == 0:
            print(f"    τ row {i+1}/{n_tau} ({time.time()-t0:.0f}s)")

    # Lensing BB template (r=0, τ=0.054 → all BB is lensing)
    _, cl_lensing_bb = generate_camb_spectra_r_tau(0.0, 0.054, lmax)

    print(f"  Grid computed in {time.time()-t0:.0f}s")
    return r_grid, tau_grid, cl_ee_grid, cl_bb_grid, cl_lensing_bb


def interp_spectra(r, tau, r_grid, tau_grid, cl_ee_grid, cl_bb_grid):
    """Bilinear interpolation of C_ℓ on the (r, τ) grid."""
    n_tau, n_r = cl_ee_grid.shape[:2]

    # Find grid cell
    r_idx = np.searchsorted(r_grid, r) - 1
    tau_idx = np.searchsorted(tau_grid, tau) - 1
    r_idx = np.clip(r_idx, 0, n_r - 2)
    tau_idx = np.clip(tau_idx, 0, n_tau - 2)

    # Fractional positions
    dr = (r - r_grid[r_idx]) / (r_grid[r_idx + 1] - r_grid[r_idx])
    dtau = (tau - tau_grid[tau_idx]) / (tau_grid[tau_idx + 1] - tau_grid[tau_idx])
    dr = np.clip(dr, 0, 1)
    dtau = np.clip(dtau, 0, 1)

    # Bilinear interpolation
    w00 = (1 - dtau) * (1 - dr)
    w01 = (1 - dtau) * dr
    w10 = dtau * (1 - dr)
    w11 = dtau * dr

    cl_ee = (w00 * cl_ee_grid[tau_idx, r_idx] +
             w01 * cl_ee_grid[tau_idx, r_idx + 1] +
             w10 * cl_ee_grid[tau_idx + 1, r_idx] +
             w11 * cl_ee_grid[tau_idx + 1, r_idx + 1])

    cl_bb = (w00 * cl_bb_grid[tau_idx, r_idx] +
             w01 * cl_bb_grid[tau_idx, r_idx + 1] +
             w10 * cl_bb_grid[tau_idx + 1, r_idx] +
             w11 * cl_bb_grid[tau_idx + 1, r_idx + 1])

    return cl_ee, cl_bb


def run_mcmc_one_map(cl_ee_obs, cl_bb_obs, r_true, tau_true,
                     lmax, f_sky, noise_cl, nside,
                     r_grid, tau_grid, cl_ee_grid, cl_bb_grid, cl_lensing_bb,
                     n_steps=5000, burn_in=2000,
                     lmax_ee=None, lmax_bb=None):
    """Run Metropolis-Hastings MCMC for a single map.

    Parameters: (r, τ, A_lens) where A_lens scales the lensing BB.
    Uses pre-computed spectral grid with bilinear interpolation — no CAMB calls.
    Prior: r ∈ [0, 0.01], τ ∈ [0.03, 0.08], A_lens ∈ [0.5, 2.0].
    """
    # Starting point
    r_cur = r_true + np.random.normal(0, 0.001)
    tau_cur = tau_true + np.random.normal(0, 0.005)
    a_lens_cur = 1.0
    r_cur = np.clip(r_cur, 0, R_MAX)
    tau_cur = np.clip(tau_cur, TAU_MIN, TAU_MAX)
    a_lens_cur = np.clip(a_lens_cur, 0.5, 2.0)

    # Get model at current point via interpolation
    cl_ee_cur, cl_bb_cur = interp_spectra(r_cur, tau_cur, r_grid, tau_grid,
                                           cl_ee_grid, cl_bb_grid)
    # Scale lensing BB by A_lens
    cl_bb_cur = cl_bb_cur - cl_lensing_bb + a_lens_cur * cl_lensing_bb

    ll_cur = log_likelihood(cl_ee_obs, cl_bb_obs, cl_ee_cur, cl_bb_cur,
                            noise_cl, f_sky, lmax_ee, lmax_bb)

    # Proposal scales (tuned for ~25% acceptance)
    sigma_r = 0.002
    sigma_tau = 0.005
    sigma_alens = 0.2

    samples_r = np.zeros(n_steps)
    samples_tau = np.zeros(n_steps)
    samples_alens = np.zeros(n_steps)
    n_accept = 0

    for step in range(n_steps):
        # Propose new parameters
        r_prop = r_cur + np.random.normal(0, sigma_r)
        tau_prop = tau_cur + np.random.normal(0, sigma_tau)
        a_lens_prop = a_lens_cur + np.random.normal(0, sigma_alens)

        # Apply priors
        if r_prop < 0 or r_prop > R_MAX or \
           tau_prop < TAU_MIN or tau_prop > TAU_MAX or \
           a_lens_prop < 0.5 or a_lens_prop > 2.0:
            samples_r[step] = r_cur
            samples_tau[step] = tau_cur
            samples_alens[step] = a_lens_cur
            continue

        # Compute model at proposed point via interpolation (fast!)
        cl_ee_prop, cl_bb_prop = interp_spectra(r_prop, tau_prop, r_grid, tau_grid,
                                                  cl_ee_grid, cl_bb_grid)
        cl_bb_prop = cl_bb_prop - cl_lensing_bb + a_lens_prop * cl_lensing_bb

        ll_prop = log_likelihood(cl_ee_obs, cl_bb_obs, cl_ee_prop, cl_bb_prop,
                                 noise_cl, f_sky, lmax_ee, lmax_bb)

        # Metropolis acceptance
        if np.log(np.random.uniform()) < ll_prop - ll_cur:
            r_cur = r_prop
            tau_cur = tau_prop
            a_lens_cur = a_lens_prop
            ll_cur = ll_prop
            n_accept += 1

        samples_r[step] = r_cur
        samples_tau[step] = tau_cur
        samples_alens[step] = a_lens_cur

    # Post burn-in statistics
    post_burn = slice(burn_in, None)
    r_mean = np.mean(samples_r[post_burn])
    tau_mean = np.mean(samples_tau[post_burn])
    r_std = np.std(samples_r[post_burn])
    tau_std = np.std(samples_tau[post_burn])
    acceptance = n_accept / n_steps

    return r_mean, tau_mean, r_std, tau_std, acceptance


def noise_arcmin_to_uK(noise_arcmin, nside):
    """Convert noise in μK-arcmin to μK per pixel."""
    npix = hp.nside2npix(nside)
    pix_area_arcmin2 = 4.0 * np.pi * (180 * 60 / np.pi)**2 / npix
    return noise_arcmin / np.sqrt(pix_area_arcmin2)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fisher + MCMC baseline for Test 4")
    parser.add_argument("--nside", type=int, nargs="+", default=[16, 128],
                        help="NSIDE values to test")
    parser.add_argument("--n_test", type=int, default=100,
                        help="Number of test maps per config")
    parser.add_argument("--mcmc_steps", type=int, default=5000,
                        help="MCMC steps per map")
    parser.add_argument("--mcmc_burn", type=int, default=2000,
                        help="MCMC burn-in steps")
    parser.add_argument("--fiducial_only", action="store_true",
                        help="Only test at fiducial (r=0.003, τ=0.054)")
    parser.add_argument("--r_fiducial", type=float, default=0.003)
    parser.add_argument("--tau_fiducial", type=float, default=0.054)
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results'))
    parser.add_argument("--skip_mcmc", action="store_true",
                        help="Only compute Fisher matrix, skip MCMC")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    configs = [
        {"f_sky": 1.0, "noise_arcmin": 0, "label": "fsky1.0_noise0"},
        {"f_sky": 1.0, "noise_arcmin": 6, "label": "fsky1.0_noise6"},
        {"f_sky": 0.1, "noise_arcmin": 0, "label": "fsky0.1_noise0"},
        {"f_sky": 0.1, "noise_arcmin": 6, "label": "fsky0.1_noise6"},
    ]

    rng = np.random.default_rng(42)

    for nside in args.nside:
        lmax = 3 * nside - 1
        print(f"\n{'='*60}")
        print(f"NSIDE={nside}, lmax={lmax}")
        print(f"{'='*60}")

        # ── Fisher matrix ──────────────────────────────────────────
        print("\n--- Fisher Matrix ---")
        for cfg in configs:
            noise_uK = (noise_arcmin_to_uK(cfg["noise_arcmin"], nside)
                        if cfg["noise_arcmin"] > 0 else 0.0)

            F = compute_fisher_matrix(
                args.r_fiducial, args.tau_fiducial, lmax,
                cfg["f_sky"], noise_uK, nside
            )

            try:
                F_inv = np.linalg.inv(F)
                sigma_r_fisher = np.sqrt(F_inv[0, 0])
                sigma_tau_fisher = np.sqrt(F_inv[1, 1])
                corr = F_inv[0, 1] / (sigma_r_fisher * sigma_tau_fisher)
            except np.linalg.LinAlgError:
                sigma_r_fisher = np.inf
                sigma_tau_fisher = np.inf
                corr = 0.0

            fisher_result = {
                "nside": nside,
                "config": cfg["label"],
                "f_sky": cfg["f_sky"],
                "noise_arcmin": cfg["noise_arcmin"],
                "r_fiducial": args.r_fiducial,
                "tau_fiducial": args.tau_fiducial,
                "sigma_r_fisher": sigma_r_fisher,
                "sigma_tau_fisher": sigma_tau_fisher,
                "r_fisher_pct": sigma_r_fisher / args.r_fiducial * 100,
                "tau_fisher_pct": sigma_tau_fisher / args.tau_fiducial * 100,
                "correlation_r_tau": corr,
                "fisher_matrix": F.tolist(),
                "lmax": lmax,
            }

            outfile = os.path.join(args.results_dir,
                                   f"test4_fisher_nside{nside}_{cfg['label']}.json")
            with open(outfile, "w") as f:
                json.dump(fisher_result, f, indent=2)

            print(f"  {cfg['label']}: σ_r={sigma_r_fisher:.6f} ({sigma_r_fisher/args.r_fiducial*100:.1f}%), "
                  f"σ_τ={sigma_tau_fisher:.6f} ({sigma_tau_fisher/args.tau_fiducial*100:.1f}%), "
                  f"corr={corr:.3f}")

        # ── MCMC ──────────────────────────────────────────────────
        if args.skip_mcmc:
            continue

        print("\n--- MCMC (Metropolis-Hastings) ---")

        # Pre-compute CAMB spectral grid for fast interpolation (once per NSIDE)
        r_grid, tau_grid, cl_ee_grid, cl_bb_grid, cl_lensing_bb = \
            precompute_spectral_grid(lmax, n_r=50, n_tau=50)

        for cfg in configs:
            f_sky = cfg["f_sky"]
            noise_uK = (noise_arcmin_to_uK(cfg["noise_arcmin"], nside)
                        if cfg["noise_arcmin"] > 0 else 0.0)
            noise_cl = noise_uK**2 * 4.0 * np.pi / hp.nside2npix(nside) if noise_uK > 0 else 0.0

            print(f"\n  Config: {cfg['label']}")
            print(f"  f_sky={f_sky}, noise={cfg['noise_arcmin']} μK-arcmin = {noise_uK:.4f} μK/pixel")

            r_preds = []
            tau_preds = []
            r_stds = []
            tau_stds = []
            t_start = time.time()

            for i in range(args.n_test):
                # Generate map at fiducial point using C_ℓ (raw_cl=True)
                r_true = args.r_fiducial
                tau_true = args.tau_fiducial

                cl_ee_true, cl_bb_true = generate_camb_spectra_r_tau(r_true, tau_true, lmax)
                cl_tt = np.full(lmax + 1, 1e-5)
                cl_te = np.zeros(lmax + 1)
                cl_eb = np.zeros(lmax + 1)
                cl_tb = np.zeros(lmax + 1)
                cl_full = np.array([cl_tt, cl_ee_true, cl_bb_true, cl_te, cl_eb, cl_tb])

                seed = int(rng.integers(0, 2**31))
                np.random.seed(seed)
                maps = hp.synfast(cl_full, nside=nside, lmax=lmax)

                q = maps[1].astype(np.float32)
                u = maps[2].astype(np.float32)

                if noise_uK > 0:
                    q += rng.normal(0, noise_uK, size=q.shape).astype(np.float32)
                    u += rng.normal(0, noise_uK, size=u.shape).astype(np.float32)

                # Apply mask
                from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
                mask = create_sky_mask(f_sky, nside, rng)
                q *= mask
                u *= mask

                # Compute observed C_ℓ
                maps_in = np.array([np.zeros_like(q), q, u])
                cl_obs = hp.anafast(maps_in, lmax=lmax, pol=True)
                cl_ee_obs = cl_obs[1] / f_sky
                cl_bb_obs = cl_obs[2] / f_sky

                # Run MCMC
                r_mean, tau_mean, r_std, tau_std, acc = run_mcmc_one_map(
                    cl_ee_obs, cl_bb_obs, r_true, tau_true,
                    lmax, f_sky, noise_cl, nside,
                    r_grid, tau_grid, cl_ee_grid, cl_bb_grid, cl_lensing_bb,
                    n_steps=args.mcmc_steps, burn_in=args.mcmc_burn,
                    lmax_ee=lmax, lmax_bb=lmax,
                )

                r_preds.append(r_mean)
                tau_preds.append(tau_mean)
                r_stds.append(r_std)
                tau_stds.append(tau_std)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"    {i+1}/{args.n_test}: "
                          f"r_mean={np.mean(r_preds):.5f}, "
                          f"τ_mean={np.mean(tau_preds):.5f}, "
                          f"time={elapsed:.0f}s")

            # Aggregate results
            r_preds = np.array(r_preds)
            tau_preds = np.array(tau_preds)
            r_stds = np.array(r_stds)
            tau_stds = np.array(tau_stds)

            result = {
                "test": "test4_mcmc_mh",
                "nside": nside,
                "config": cfg["label"],
                "f_sky": f_sky,
                "noise_arcmin": cfg["noise_arcmin"],
                "n_test": args.n_test,
                "mcmc_steps": args.mcmc_steps,
                "mcmc_burn_in": args.mcmc_burn,
                "r_mean": float(np.mean(r_preds)),
                "tau_mean": float(np.mean(tau_preds)),
                "r_bias": float(np.mean(r_preds) - args.r_fiducial),
                "tau_bias": float(np.mean(tau_preds) - args.tau_fiducial),
                "sigma_r": float(np.mean(r_stds)),
                "sigma_tau": float(np.mean(tau_stds)),
                "rmse_r": float(np.sqrt(np.mean((r_preds - args.r_fiducial)**2))),
                "rmse_tau": float(np.sqrt(np.mean((tau_preds - args.tau_fiducial)**2))),
                "r_pct_error": float(np.sqrt(np.mean((r_preds - args.r_fiducial)**2)) / args.r_fiducial * 100),
                "tau_pct_error": float(np.sqrt(np.mean((tau_preds - args.tau_fiducial)**2)) / args.tau_fiducial * 100),
                "r_fiducial": args.r_fiducial,
                "tau_fiducial": args.tau_fiducial,
                "cpu_time_seconds": time.time() - t_start,
                "method": "metropolis_hastings_A_lens",
            }

            outfile = os.path.join(args.results_dir,
                                   f"test4_mcmc_mh_nside{nside}_{cfg['label']}.json")
            with open(outfile, "w") as f:
                json.dump(result, f, indent=2)

            print(f"\n  Results: {outfile}")
            print(f"  r: mean={result['r_mean']:.5f}, bias={result['r_bias']:.5f}, "
                  f"σ={result['sigma_r']:.5f}, RMSE={result['rmse_r']:.5f} ({result['r_pct_error']:.1f}%)")
            print(f"  τ: mean={result['tau_mean']:.5f}, bias={result['tau_bias']:.5f}, "
                  f"σ={result['sigma_tau']:.5f}, RMSE={result['rmse_tau']:.5f} ({result['tau_pct_error']:.1f}%)")
            print(f"  CPU time: {result['cpu_time_seconds']:.1f}s")

    print("\n=== All Fisher + MCMC baselines complete ===")


if __name__ == "__main__":
    main()
