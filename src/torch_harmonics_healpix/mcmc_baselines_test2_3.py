"""MCMC baselines for Test 2 (polarization), Test 3 (τ estimation), and Test 4 (joint r/τ estimation).

These reproduce the maximum-likelihood estimators from Krachmalnicoff & Tomasi (2019)
Sections 6.2 and 6.3. Run on CPU (Popeye) since they don't need GPU.

Test 2 MCMC: Fit ℓ_Ep and ℓ_Bp from the EE and BB power spectra of Q/U maps.
Test 3 MCMC: Fit τ from the EE power spectrum of Q/U maps using CAMB templates.
Test 4 MCMC (mcmc_baseline_r_tau): Joint r/τ estimation via chi-squared grid search.

v4 (2026-10) improvements:
  - Exact Gaussian (Wishart) likelihood instead of the χ² approximation
    (see mcmc_baseline.py for the full rationale; removes the +2.7% bias).
  - NO pixel window in the model (healpy synfast ≥1.15 defaults to
    pixwin=False; verified against a flat-floor test).
  - Fit range: polarization spectra start at ℓ=2 (anafast returns exact
    zeros at ℓ=0,1 for E/B — log(0) would poison the exact likelihood)
    and stop at ℓ_fit = 2·N_side (the 2·N_side < ℓ ≤ 3·N_side−1 band is
    aliasing-dominated; verified to degrade the fit).
  - Partial-sky maps: Q/U are zeroed outside the mask (KT19 methodology,
    same input the CNN sees) and the f_sky pseudo-C_ℓ correction is applied
    (C_obs/f_sky, mode count (2ℓ+1)·f_sky). NOTE: E/B leakage from the mask
    is NOT corrected — a proper treatment needs a full pseudo-C_ℓ framework
    (e.g. NaMaster/pysm-pol). KT19 report no partial-sky MCMC number; at
    f_sky≤0.5 this estimator degrades strongly (see BENCHMARKS.md), which
    is itself the paper's point: spectral estimation on patches is where
    a CNN operating on pixels can win.
  - Test 3: the true τ of each map is the τ of the grid spectrum that
    generated it (tau_grid[idx]); previously τ_true was drawn independently
    of the spectrum index, so most "errors" were label noise.
"""

import numpy as np
import healpy as hp
from scipy.optimize import minimize_scalar

from .data_generation import NSIDE, LMAX, SIGMA_P
from .data_generation_test2 import (
    LEP_MIN, LEP_MAX, LBP_MIN, LBP_MAX,
    generate_polarization_map,
)


def _exact_nll(cl_obs, cl_model, ell, f_sky=1.0):
    """Exact Gaussian -2lnL for one auto-spectrum, with f_sky mode scaling.

    -2 ln L = Σ_ℓ (2ℓ+1) f_sky [ln(C_model/C_obs) + C_obs/C_model - 1]
    """
    n_modes = (2 * ell + 1) * f_sky
    return np.sum(n_modes * (np.log(cl_model / cl_obs) + cl_obs / cl_model - 1))


def mcmc_estimate_ell_ep_bp(
    q_map: np.ndarray,
    u_map: np.ndarray,
    sigma_p: float = SIGMA_P,
    lmax: int = LMAX,
    nside: int = NSIDE,
    f_sky: float = 1.0,
) -> tuple:
    """Estimate ℓ_Ep and ℓ_Bp from Q/U maps using maximum likelihood.

    Computes EE and BB pseudo-power spectra, applies the f_sky correction,
    then fits the Gaussian-peak model with the exact Gaussian likelihood
    over ℓ ∈ [2, 2·N_side].

    Args:
        q_map: Q polarization HEALPix map.
        u_map: U polarization HEALPix map.
        sigma_p: Width of the Gaussian peak.
        lmax: Maximum multipole for anafast.
        nside: HEALPix Nside.
        f_sky: Sky fraction used for the pseudo-C_ℓ correction.

    Returns:
        (estimated_ℓ_Ep, estimated_ℓ_Bp)
    """
    t_map = np.zeros_like(q_map)
    cl = hp.anafast([t_map, q_map, u_map], lmax=lmax, pol=True)
    # Pseudo-C_ℓ of a masked (mean-inpainted) map is ≈ f_sky · C_ℓ
    cl_ee = cl[1] / f_sky
    cl_bb = cl[2] / f_sky

    lmax_fit = min(2 * nside, lmax)
    ell = np.arange(2, lmax_fit + 1)
    # anafast returns exact zeros at ℓ=0,1 for E/B: exclude from the
    # exact likelihood (log of zero), and floor any other zeros
    cl_ee = np.maximum(cl_ee[2: lmax_fit + 1], 1e-30)
    cl_bb = np.maximum(cl_bb[2: lmax_fit + 1], 1e-30)

    def nll_ee(ell_ep):
        cl_model = np.exp(-((ell - ell_ep) ** 2) / (2 * sigma_p**2)) + 1e-5
        return _exact_nll(cl_ee, cl_model, ell, f_sky)

    def nll_bb(ell_bp):
        cl_model = np.exp(-((ell - ell_bp) ** 2) / (2 * sigma_p**2)) + 1e-5
        return _exact_nll(cl_bb, cl_model, ell, f_sky)

    result_ep = minimize_scalar(nll_ee, bounds=(LEP_MIN, LEP_MAX), method="bounded")
    result_bp = minimize_scalar(nll_bb, bounds=(LBP_MIN, LBP_MAX), method="bounded")

    return result_ep.x, result_bp.x


def evaluate_mcmc_test2(
    n_maps: int = 1000,
    nside: int = NSIDE,
    lmax: int = LMAX,
    sigma_p: float = SIGMA_P,
    f_sky: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run MCMC baseline for Test 2 (polarization) and compute mean % error.

    Args:
        n_maps: Number of test maps.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        sigma_p: Peak width.
        f_sky: Sky fraction.
        seed: Random seed.

    Returns:
        Dict with ep_pct_error, bp_pct_error, median errors, and time per map.
    """
    import time
    from .data_generation_test2 import create_sky_mask

    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)

    # Generate true parameter values
    ell_ep_true = rng.uniform(LEP_MIN, LEP_MAX, size=n_maps).astype(np.float32)
    ell_bp_true = rng.uniform(LBP_MIN, LBP_MAX, size=n_maps).astype(np.float32)

    # Create mask (same for all maps, matching paper methodology)
    mask = create_sky_mask(f_sky, nside, rng).astype(np.float32)

    ep_errors = []
    bp_errors = []
    t0 = time.time()

    for i in range(n_maps):
        q, u = generate_polarization_map(
            ell_ep_true[i], ell_bp_true[i],
            nside, lmax, sigma_p, 0.0, rng
        )

        # Apply mask: zero out unobserved pixels — exactly the input the
        # SpectralCNN sees (KT19 Sect. 6.2: "the Q/U signal outside the
        # mask have been set to zero"). No inpainting: it would give the
        # baseline a different input than the CNN.
        q = q * mask
        u = u * mask

        ell_ep_pred, ell_bp_pred = mcmc_estimate_ell_ep_bp(
            q, u, sigma_p, lmax, nside, f_sky
        )

        ep_pct = abs(ell_ep_pred - ell_ep_true[i]) / ell_ep_true[i] * 100
        bp_pct = abs(ell_bp_pred - ell_bp_true[i]) / ell_bp_true[i] * 100
        ep_errors.append(ep_pct)
        bp_errors.append(bp_pct)

        if (i + 1) % 100 == 0:
            print(f"  MCMC Test 2: {i+1}/{n_maps} maps done")

    elapsed = time.time() - t0

    return {
        "ep_pct_error": float(np.mean(ep_errors)),
        "bp_pct_error": float(np.mean(bp_errors)),
        "ep_median_pct_error": float(np.median(ep_errors)),
        "bp_median_pct_error": float(np.median(bp_errors)),
        "time_per_map": elapsed / n_maps,
        "n_maps": n_maps,
        "f_sky": f_sky,
    }


def evaluate_mcmc_test3(
    n_maps: int = 1000,
    nside: int = NSIDE,
    lmax: int = LMAX,
    seed: int = 42,
) -> dict:
    """Run MCMC baseline for Test 3 (τ estimation) and compute mean % error.

    Uses pre-computed CAMB template EE spectra, then fits the observed EE
    spectrum with the exact Gaussian likelihood (pixel-window convolved).

    Label integrity (v4): each map's true τ is the τ of the grid spectrum
    that generated it — tau_grid[spectrum_index] — NOT an independent draw.

    Args:
        n_maps: Number of test maps.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        seed: Random seed.

    Returns:
        Dict with tau_pct_error, median error, and time per map.
    """
    import time
    from .data_generation_test3 import (
        TAU_MIN, TAU_MAX, N_CAMB_SPECTRA,
        precompute_camb_spectra, generate_tau_map,
    )

    rng = np.random.default_rng(seed)

    # Pre-compute CAMB template spectra for fitting
    print("  Pre-computing CAMB template spectra for MCMC fitting...")
    tau_grid, cl_ee_array, cl_bb_array = precompute_camb_spectra(
        N_CAMB_SPECTRA, lmax, seed=seed + 100
    )

    # Maps are generated from grid spectra; the label IS the grid τ
    spectrum_indices = rng.integers(0, N_CAMB_SPECTRA, size=n_maps)
    tau_true = tau_grid[spectrum_indices]

    # Fit range ℓ ∈ [2, 2·N_side]: no pixel window (maps are pixwin=False,
    # see mcmc_baseline.py), E/B are exactly zero at ℓ=0,1 (log(0) hazard),
    # and the ℓ > 2·N_side band is aliasing-dominated.
    lmax_fit = min(2 * nside, lmax)
    ell = np.arange(2, lmax_fit + 1)
    twoellp1 = (2 * ell + 1)
    cl_ee_model = np.maximum(cl_ee_array[:, 2: lmax_fit + 1], 1e-30)

    tau_errors = []
    t0 = time.time()

    for i in range(n_maps):
        idx = spectrum_indices[i]
        q, u, _ = generate_tau_map(
            tau_true[i], nside, lmax, 0.0, 1.0, rng,
            cl_ee=cl_ee_array[idx],
            cl_bb=cl_bb_array[idx],
        )

        # Compute observed EE power spectrum (full sky, f_sky=1)
        t_map = np.zeros_like(q)
        cl = hp.anafast([t_map, q, u], lmax=lmax, pol=True)
        cl_ee_obs = np.maximum(cl[1][2: lmax_fit + 1], 1e-30)

        # Exact Gaussian likelihood over the template grid
        nll = np.sum(
            twoellp1[None, :]
            * (np.log(cl_ee_model / cl_ee_obs[None, :])
               + cl_ee_obs[None, :] / cl_ee_model - 1),
            axis=1,
        )
        best_tau = tau_grid[np.argmin(nll)]

        tau_pct = abs(best_tau - tau_true[i]) / max(tau_true[i], 0.01) * 100
        tau_errors.append(tau_pct)

        if (i + 1) % 100 == 0:
            print(f"  MCMC Test 3: {i+1}/{n_maps} maps done")

    elapsed = time.time() - t0

    return {
        "tau_pct_error": float(np.mean(tau_errors)),
        "tau_median_pct_error": float(np.median(tau_errors)),
        "time_per_map": elapsed / n_maps,
        "n_maps": n_maps,
    }


def mcmc_baseline_r_tau(
    q_map: np.ndarray,
    u_map: np.ndarray,
    mask: np.ndarray,
    r_grid: np.ndarray,
    tau_grid: np.ndarray,
    cl_ee_array: np.ndarray,
    cl_bb_array: np.ndarray,
    noise_std: float = 0.0,
    nside: int = 16,
    lmax: int = 47,
) -> dict:
    """MCMC baseline for Test 4: joint r/τ estimation via exact-likelihood grid search.

    For each (r, τ) in the grid, compare the (noise-bias included, f_sky-corrected)
    model spectra to the observed pseudo-C_ℓ via the exact Gaussian likelihood
    over ℓ ∈ [2, 2·N_side], and find the best-fit (r, τ).

    Args:
        q_map: Q polarization map (1D HEALPix array).
        u_map: U polarization map (1D HEALPix array).
        mask: Sky mask (1D HEALPix array, 1=observed, 0=masked).
        r_grid: 1D array of r values to search over.
        tau_grid: 1D array of τ values to search over.
        cl_ee_array: Pre-computed EE spectra, shape (n_spectra, lmax+1).
            Must be ordered as a flattened (r, τ) grid with index = tau_idx * n_r + r_idx.
        cl_bb_array: Pre-computed BB spectra, shape (n_spectra, lmax+1).
            Same ordering as cl_ee_array.
        noise_std: White noise in μK (0 for no noise).
        nside: HEALPix Nside.
        lmax: Maximum multipole.

    Returns:
        Dict with keys: r_best, tau_best, r_pct_error, tau_pct_error, chi2_grid.
        Note: r_pct_error and tau_pct_error are always NaN; the caller must compute
        percentage errors using known true values.
    """
    npix = hp.nside2npix(nside)
    f_sky = float(np.mean(mask))

    # Observed pseudo-C_ℓ from masked Q/U maps, corrected to full sky.
    # Fit range ℓ ∈ [2, 2·N_side] (no pixel window: maps are pixwin=False;
    # E/B are zero at ℓ=0,1; ℓ > 2·N_side is aliasing-dominated).
    maps_in = np.array([np.zeros_like(q_map), q_map, u_map])
    cl_obs = hp.anafast(maps_in, lmax=lmax, pol=True)
    lmax_fit = min(2 * nside, lmax)
    cl_ee_obs = np.maximum(cl_obs[1][2: lmax_fit + 1] / f_sky, 1e-30)
    cl_bb_obs = np.maximum(cl_obs[2][2: lmax_fit + 1] / f_sky, 1e-30)

    # Noise power spectrum (white noise)
    noise_cl = noise_std**2 * 4.0 * np.pi / npix if noise_std > 0.0 else 0.0

    ell = np.arange(2, lmax_fit + 1)
    n_modes = (2 * ell + 1) * f_sky

    # Build 2D grid over (r, τ)
    r_mesh, tau_mesh = np.meshgrid(r_grid, tau_grid)
    chi2_grid = np.full(r_mesh.shape, np.inf)

    n_spectra = cl_ee_array.shape[0]
    n_r = len(r_grid)
    n_tau = len(tau_grid)

    for i in range(r_mesh.shape[0]):  # tau index
        for j in range(r_mesh.shape[1]):  # r index
            spec_idx = i * n_r + j
            if spec_idx >= n_spectra:
                continue

            cl_ee_model = np.maximum(cl_ee_array[spec_idx][2: lmax_fit + 1] + noise_cl, 1e-30)
            cl_bb_model = np.maximum(cl_bb_array[spec_idx][2: lmax_fit + 1] + noise_cl, 1e-30)

            nll_ee = np.sum(n_modes * (np.log(cl_ee_model / cl_ee_obs) + cl_ee_obs / cl_ee_model - 1))
            nll_bb = np.sum(n_modes * (np.log(cl_bb_model / cl_bb_obs) + cl_bb_obs / cl_bb_model - 1))
            chi2_grid[i, j] = nll_ee + nll_bb

    # Find best-fit (r, τ)
    best_flat = np.argmin(chi2_grid)
    best_tau_idx, best_r_idx = np.unravel_index(best_flat, chi2_grid.shape)

    r_best = float(r_grid[best_r_idx])
    tau_best = float(tau_grid[best_tau_idx])

    # Convention: return NaN for pct errors; caller computes them from true values.
    r_pct_error = np.nan
    tau_pct_error = np.nan

    return {
        "r_best": r_best,
        "tau_best": tau_best,
        "r_pct_error": r_pct_error,
        "tau_pct_error": tau_pct_error,
        "chi2_grid": chi2_grid,
    }
