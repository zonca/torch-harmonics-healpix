"""MCMC baselines for Test 2 (polarization) and Test 3 (τ estimation).

These reproduce the maximum-likelihood estimators from Krachmalnicoff & Tomasi (2019)
Sections 6.2 and 6.3. Run on CPU (Popeye) since they don't need GPU.

Test 2 MCMC: Fit ℓ_Ep and ℓ_Bp from the EE and BB power spectra of Q/U maps.
Test 3 MCMC: Fit τ from the EE power spectrum of Q/U maps using CAMB templates.
"""

import numpy as np
import healpy as hp
from scipy.optimize import minimize_scalar, minimize

from .data_generation import NSIDE, LMAX, SIGMA_P
from .data_generation_test2 import (
    LEP_MIN, LEP_MAX, LBP_MIN, LBP_MAX,
    generate_polarization_map,
)


def mcmc_estimate_ell_ep_bp(
    q_map: np.ndarray,
    u_map: np.ndarray,
    sigma_p: float = SIGMA_P,
    lmax: int = LMAX,
    nside: int = NSIDE,
) -> tuple:
    """Estimate ℓ_Ep and ℓ_Bp from Q/U maps using maximum-likelihood.

    Computes EE and BB power spectra, then fits Gaussian peaks
    to estimate the peak positions.

    Args:
        q_map: Q polarization HEALPix map.
        u_map: U polarization HEALPix map.
        sigma_p: Width of the Gaussian peak.
        lmax: Maximum multipole.
        nside: HEALPix Nside.

    Returns:
        (estimated_ℓ_Ep, estimated_ℓ_Bp)
    """
    # Compute polarization power spectra
    # healpy.anafast with pol=True needs [T, Q, U] (3 maps)
    # Use zeros for T since we only have polarization
    t_map = np.zeros_like(q_map)
    cl = hp.anafast([t_map, q_map, u_map], lmax=lmax, pol=True)
    cl_ee = cl[1]  # EE spectrum
    cl_bb = cl[2]  # BB spectrum

    ell = np.arange(lmax + 1)

    def neg_log_likelihood_ee(ell_ep):
        cl_model = np.exp(-((ell - ell_ep) ** 2) / (2 * sigma_p**2)) + 1e-5
        sigma_cl = np.abs(cl_model) * np.sqrt(2.0 / (2 * ell + 1))
        sigma_cl = np.maximum(sigma_cl, 1e-10)
        chi2 = np.sum(((cl_ee - cl_model) / sigma_cl) ** 2)
        return chi2

    def neg_log_likelihood_bb(ell_bp):
        cl_model = np.exp(-((ell - ell_bp) ** 2) / (2 * sigma_p**2)) + 1e-5
        sigma_cl = np.abs(cl_model) * np.sqrt(2.0 / (2 * ell + 1))
        sigma_cl = np.maximum(sigma_cl, 1e-10)
        chi2 = np.sum(((cl_bb - cl_model) / sigma_cl) ** 2)
        return chi2

    result_ep = minimize_scalar(neg_log_likelihood_ee, bounds=(LEP_MIN, LEP_MAX), method="bounded")
    result_bp = minimize_scalar(neg_log_likelihood_bb, bounds=(LBP_MIN, LBP_MAX), method="bounded")

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

        # Apply mask
        q = q * mask
        u = u * mask

        if f_sky < 1.0:
            # For partial sky, fill masked pixels with mean of observed pixels
            # (same inpainting as used in SpectralCNN for fair comparison)
            q_obs_mean = np.sum(q * mask) / np.sum(mask)
            u_obs_mean = np.sum(u * mask) / np.sum(mask)
            q = q * mask + q_obs_mean * (1 - mask)
            u = u * mask + u_obs_mean * (1 - mask)

        ell_ep_pred, ell_bp_pred = mcmc_estimate_ell_ep_bp(q, u, sigma_p, lmax, nside)

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

    Uses CAMB to compute template EE spectra for different τ values,
    then fits the observed EE spectrum to estimate τ.

    Args:
        n_maps: Number of test maps.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        seed: Random seed.

    Returns:
        Dict with tau_pct_error, median error, and time per map.
    """
    import time
    import camb
    from .data_generation_test3 import (
        TAU_MIN, TAU_MAX, N_CAMB_SPECTRA,
        precompute_camb_spectra, generate_tau_map,
    )

    rng = np.random.default_rng(seed)

    # Pre-compute CAMB template spectra for fitting
    print("  Pre-computing CAMB template spectra for MCMC fitting...")
    tau_grid, cl_ee_array, _ = precompute_camb_spectra(N_CAMB_SPECTRA, lmax, seed=seed + 100)

    # Generate test maps
    tau_true = rng.uniform(TAU_MIN, TAU_MAX, size=n_maps).astype(np.float32)
    spectrum_indices = rng.integers(0, N_CAMB_SPECTRA, size=n_maps)

    tau_errors = []
    t0 = time.time()

    for i in range(n_maps):
        q, u, _ = generate_tau_map(
            tau_true[i], nside, lmax, 0.0, 1.0, rng,
            cl_ee=cl_ee_array[spectrum_indices[i]],
            cl_bb=None,  # Will be generated internally
        )

        # Compute EE power spectrum
        # healpy.anafast with pol=True needs [T, Q, U]
        t_map = np.zeros_like(q)
        cl = hp.anafast([t_map, q, u], lmax=lmax, pol=True)
        cl_ee_obs = cl[1]

        # Find best-fit τ by minimizing chi-squared over template grid
        best_chi2 = np.inf
        best_tau = tau_grid[0]

        ell = np.arange(lmax + 1)
        for j, tau_j in enumerate(tau_grid):
            cl_model = cl_ee_array[j]
            # Cosmic variance weighting
            sigma_cl = np.abs(cl_model) * np.sqrt(2.0 / (2 * ell + 1))
            sigma_cl = np.maximum(sigma_cl, 1e-10)
            chi2 = np.sum(((cl_ee_obs - cl_model) / sigma_cl) ** 2)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_tau = tau_j

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
