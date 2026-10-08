"""MCMC baseline for ℓ_p estimation (Test 1).

Reproduces the maximum-likelihood estimator from Krachmalnicoff & Tomasi (2019)
Section 6.1.1, Eqs. (5)-(8).

v4 (2026-10) design (validated against KT19 Table 1: 0.65/2.41/4.58/7.67%
at σ_n = 0/5/10/15 vs paper 0.7/2.5/4.8/7.8%):

  - Exact Gaussian (Wishart) likelihood
    -2 ln L = Σ_ℓ (2ℓ+1) [ln(C_model/C_obs) + C_obs/C_model - 1]
    instead of the χ² approximation (first-order expansion of the same);
    the exact form removes the +2.7% bias of the χ² fit at zero noise.
  - NO pixel window in the model: healpy ≥1.15 synfast defaults to
    pixwin=False, and our maps are generated with that default (verified:
    flat-floor C_47 ≈ 1e-5, not the windowed ~2e-6). Convolving the model
    with the window (or deconvolving the data) biases the fit badly.
  - Fit range capped at ℓ_fit = 2·N_side: the band 2·N_side < ℓ ≤ 3·N_side−1
    of a HEALPix pseudo-C_ℓ is dominated by pixelization/aliasing noise and
    degrades the fit (verified: lmax_fit=47 → 1.2%, lmax_fit=32 → 0.65%,
    equal to the Fisher bound 0.70%).
  - Noise enters the model as N_ℓ = 4π σ_n² / N_pix (KT19 Eq. 7).
"""

import numpy as np
import healpy as hp
from scipy.optimize import minimize_scalar

from .data_generation import NSIDE, LMAX, SIGMA_P


def mcmc_estimate_ell_p(
    map_data: np.ndarray,
    sigma_p: float = SIGMA_P,
    lmax: int = LMAX,
    noise_std: float = 0.0,
    nside: int = NSIDE,
    lmax_fit: "int | None" = None,
) -> float:
    """Estimate ℓ_p from a single HEALPix map using maximum likelihood.

    Computes the power spectrum of the map via healpy.anafast, then
    minimizes the exact Gaussian likelihood against the model spectrum
    over ℓ ≤ lmax_fit (default 2·N_side).

    Args:
        map_data: 1D HEALPix map array.
        sigma_p: Width of the Gaussian peak in the power spectrum model.
        lmax: Maximum multipole for anafast.
        noise_std: White noise standard deviation (adds N_ℓ to model).
        nside: HEALPix Nside (used for noise power and fit-range default).
        lmax_fit: Highest ℓ used in the likelihood (default 2·N_side).

    Returns:
        Estimated ℓ_p value.
    """
    if lmax_fit is None:
        lmax_fit = min(2 * nside, lmax)

    cl_obs = hp.anafast(map_data, lmax=lmax)[: lmax_fit + 1]
    ell = np.arange(lmax_fit + 1)
    n_ell = 4 * np.pi * noise_std**2 / hp.nside2npix(nside) if noise_std > 0 else 0.0

    def neg_log_likelihood(ell_p):
        cl_model = np.exp(-((ell - ell_p) ** 2) / (2 * sigma_p**2)) + 1e-5 + n_ell
        # Exact Gaussian (Wishart) likelihood for one observed C_ℓ:
        # -2 ln L = Σ_ℓ (2ℓ+1) [ln(C_model/C_obs) + C_obs/C_model - 1]
        return np.sum((2 * ell + 1) * (np.log(cl_model / cl_obs) + cl_obs / cl_model - 1))

    result = minimize_scalar(neg_log_likelihood, bounds=(5, 20), method="bounded")
    return result.x


def evaluate_mcmc_baseline(
    maps: np.ndarray,
    ell_p_true: np.ndarray,
    sigma_p: float = SIGMA_P,
    lmax: int = LMAX,
    noise_std: float = 0.0,
    nside: int = NSIDE,
) -> float:
    """Run MCMC baseline on a dataset and compute mean percentage error.

    Args:
        maps: [n_maps, npix] array of HEALPix maps.
        ell_p_true: [n_maps] array of true ℓ_p values.
        sigma_p: Width of the Gaussian peak.
        lmax: Maximum multipole.
        noise_std: White noise standard deviation.
        nside: HEALPix Nside.

    Returns:
        Mean percentage error: avg(|ℓ_p_pred - ℓ_p_true| / ℓ_p_true * 100)
    """
    ell_p_pred = np.array([
        mcmc_estimate_ell_p(m, sigma_p, lmax, noise_std, nside)
        for m in maps
    ])
    mean_pct_error = np.mean(np.abs(ell_p_pred - ell_p_true) / ell_p_true * 100)
    return mean_pct_error
