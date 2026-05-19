"""MCMC baseline for ℓ_p estimation.

Reproduces the maximum-likelihood estimator from Krachmalnicoff & Tomasi (2019)
Section 6.1.1. Uses a chi-squared fit of the observed power spectrum against
the model C_ℓ = exp(-(ℓ - ℓ_p)² / (2σ²_p)) + 10⁻⁵.
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
) -> float:
    """Estimate ℓ_p from a single HEALPix map using maximum-likelihood.

    Computes the power spectrum of the map via healpy.anafast, then
    minimizes the chi-squared distance to the model spectrum.

    Args:
        map_data: 1D HEALPix map array.
        sigma_p: Width of the Gaussian peak in the power spectrum model.
        lmax: Maximum multipole.
        noise_std: White noise standard deviation (adds N_ℓ to model).
        nside: HEALPix Nside (used for noise power calculation).

    Returns:
        Estimated ℓ_p value.
    """
    cl_est = hp.anafast(map_data, lmax=lmax)

    def neg_log_likelihood(ell_p):
        ell = np.arange(lmax + 1)
        cl_model = np.exp(-((ell - ell_p) ** 2) / (2 * sigma_p**2)) + 1e-5
        if noise_std > 0:
            npix = hp.nside2npix(nside)
            n_ell = 4 * np.pi * noise_std**2 / npix
            cl_model = cl_model + n_ell
        # Cosmic variance: σ(C_ℓ) = C_ℓ * sqrt(2 / (2ℓ + 1))
        sigma_cl = cl_model * np.sqrt(2.0 / (2 * np.arange(lmax + 1) + 1))
        chi2 = np.sum(((cl_est - cl_model) / sigma_cl) ** 2)
        return chi2

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
