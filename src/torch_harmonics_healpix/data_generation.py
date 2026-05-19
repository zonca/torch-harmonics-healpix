"""Power spectrum models and data generation for Test 1 benchmark.

Generates Gaussian random fields on HEALPix spheres with a peaked power spectrum:
    C_ℓ = exp(-(ℓ - ℓ_p)² / (2 σ_p²)) + 10⁻⁵

where ℓ_p ∈ [5, 20] is the parameter to estimate.
This reproduces the benchmark from Krachmalnicoff & Tomasi (2019, arXiv:1902.04083) §6.1.1.
"""

import numpy as np
import healpy as hp


NSIDE = 16
LMAX = 3 * NSIDE - 1  # 47
SIGMA_P = 5
LP_MIN = 5.0
LP_MAX = 20.0


def generate_power_spectrum(
    ell_p: float, sigma_p: float = SIGMA_P, lmax: int = LMAX
) -> np.ndarray:
    """Generate a Gaussian-peaked angular power spectrum.

    Args:
        ell_p: Peak multipole (the parameter to estimate).
        sigma_p: Width of the Gaussian peak.
        lmax: Maximum multipole.

    Returns:
        1D array of C_ℓ values, length lmax+1.
    """
    ell = np.arange(lmax + 1)
    cl = np.exp(-((ell - ell_p) ** 2) / (2 * sigma_p**2)) + 1e-5
    return cl


def generate_map(
    ell_p: float,
    nside: int = NSIDE,
    lmax: int = LMAX,
    sigma_p: float = SIGMA_P,
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a single HEALPix map with given ℓ_p and optional noise.

    Args:
        ell_p: Peak multipole.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        sigma_p: Width of the Gaussian peak.
        noise_std: Standard deviation of additive white noise (0 = no noise).
        rng: NumPy random Generator for reproducibility.

    Returns:
        1D HEALPix map array of length nside2npix(nside).
    """
    cl = generate_power_spectrum(ell_p, sigma_p, lmax)

    if rng is not None:
        # Use the rng to create a seed for healpy's synfast
        # healpy synfast doesn't accept rng objects, so we derive an integer seed
        seed = int(rng.integers(0, 2**31))
        m = hp.synfast(cl, nside=nside, lmax=lmax, verbose=False, seed=seed)
    else:
        m = hp.synfast(cl, nside=nside, lmax=lmax, verbose=False)

    if noise_std > 0:
        if rng is not None:
            noise = rng.normal(0, noise_std, size=m.shape)
        else:
            noise = np.random.normal(0, noise_std, size=m.shape)
        m = m + noise

    return m.astype(np.float32)


def generate_dataset(
    n_maps: int,
    nside: int = NSIDE,
    lmax: int = LMAX,
    sigma_p: float = SIGMA_P,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of HEALPix maps with random ℓ_p values.

    Args:
        n_maps: Number of maps to generate.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        sigma_p: Width of the Gaussian peak.
        noise_std: Standard deviation of additive white noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (maps, ell_p_values) where:
            maps: [n_maps, npix] float32 array
            ell_p_values: [n_maps] float32 array
    """
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)

    maps = np.zeros((n_maps, npix), dtype=np.float32)
    ell_p_values = rng.uniform(LP_MIN, LP_MAX, size=n_maps).astype(np.float32)

    for i in range(n_maps):
        maps[i] = generate_map(
            ell_p_values[i],
            nside=nside,
            lmax=lmax,
            sigma_p=sigma_p,
            noise_std=noise_std,
            rng=rng,
        )

    return maps, ell_p_values
