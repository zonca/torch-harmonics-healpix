"""Data generation for Test 2: ℓ_Ep/ℓ_Bp estimation from Q/U polarization maps.

Reproduces the benchmark from Krachmalnicoff & Tomasi (2019) Section 6.1.2.
Generates spin-2 (Q/U) HEALPix maps with Gaussian-peaked E and B power spectra,
with optional partial sky masking.
"""

import numpy as np
import healpy as hp


NSIDE = 16
LMAX = 3 * NSIDE - 1  # 47
SIGMA_P = 5
LEP_MIN = 5.0
LEP_MAX = 20.0
LBP_MIN = 5.0
LBP_MAX = 20.0


def generate_polarization_power_spectra(
    ell_ep: float,
    ell_bp: float,
    sigma_p: float = SIGMA_P,
    lmax: int = LMAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate E-mode and B-mode power spectra with Gaussian peaks.

    Args:
        ell_ep: Peak multipole for E-mode.
        ell_bp: Peak multipole for B-mode.
        sigma_p: Width of the Gaussian peaks.
        lmax: Maximum multipole.

    Returns:
        Tuple of (cl_ee, cl_bb, cl_eb) arrays, each length lmax+1.
        cl_eb is always zero (no E-B correlation).
    """
    ell = np.arange(lmax + 1)
    cl_ee = np.exp(-((ell - ell_ep) ** 2) / (2 * sigma_p**2)) + 1e-5
    cl_bb = np.exp(-((ell - ell_bp) ** 2) / (2 * sigma_p**2)) + 1e-5
    cl_eb = np.zeros(lmax + 1)
    return cl_ee, cl_bb, cl_eb


def generate_polarization_map(
    ell_ep: float,
    ell_bp: float,
    nside: int = NSIDE,
    lmax: int = LMAX,
    sigma_p: float = SIGMA_P,
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Q/U polarization maps from E/B power spectra.

    Args:
        ell_ep: Peak multipole for E-mode.
        ell_bp: Peak multipole for B-mode.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        sigma_p: Width of the Gaussian peaks.
        noise_std: Standard deviation of additive white noise.
        rng: NumPy random Generator for reproducibility.

    Returns:
        Tuple of (Q_map, U_map), each 1D HEALPix array.
    """
    cl_ee, cl_bb, cl_eb = generate_polarization_power_spectra(
        ell_ep, ell_bp, sigma_p, lmax
    )

    # Build full TEB power spectrum matrix for healpy.synalm
    # cl_tt is just noise floor (no temperature signal in Test 2)
    cl_tt = np.full(lmax + 1, 1e-5)
    cl_te = np.zeros(lmax + 1)
    cl_tb = np.zeros(lmax + 1)

    # synfast expects array of shape (6, lmax+1): TT, EE, BB, TE, EB, TB
    cl_full = np.array([cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb])

    # Reproducibility via numpy random state
    if rng is not None:
        rand_state = np.random.get_state()
        seed = int(rng.integers(0, 2**31))
        np.random.seed(seed)

    maps = hp.synfast(cl_full, nside=nside, lmax=lmax)

    if rng is not None:
        np.random.set_state(rand_state)

    q_map = maps[1].astype(np.float32)
    u_map = maps[2].astype(np.float32)

    if noise_std > 0:
        if rng is not None:
            q_map = q_map + rng.normal(0, noise_std, size=q_map.shape).astype(np.float32)
            u_map = u_map + rng.normal(0, noise_std, size=u_map.shape).astype(np.float32)
        else:
            q_map = q_map + np.random.normal(0, noise_std, size=q_map.shape).astype(np.float32)
            u_map = u_map + np.random.normal(0, noise_std, size=u_map.shape).astype(np.float32)

    return q_map, u_map


def create_sky_mask(
    f_sky: float,
    nside: int = NSIDE,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Create a random sky mask with given sky fraction.

    For f_sky < 1, masks out a contiguous patch of the sky (roughly).
    Uses a simple approach: mask based on colatitude from the north pole.

    Args:
        f_sky: Sky fraction to keep (0 to 1).
        nside: HEALPix Nside.
        rng: Random generator (for random rotation of mask).

    Returns:
        Boolean mask array of length npix. True = observed, False = masked.
    """
    npix = hp.nside2npix(nside)

    if f_sky >= 1.0:
        return np.ones(npix, dtype=bool)

    # Compute colatitude for each pixel
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Rotate the mask center randomly if rng provided
    if rng is not None:
        # Random rotation: pick a random center
        center_theta = rng.uniform(0, np.pi)
        center_phi = rng.uniform(0, 2 * np.pi)
    else:
        center_theta = 0.0
        center_phi = 0.0

    # Angular distance from center
    cos_dist = (np.sin(theta) * np.sin(center_theta) *
                np.cos(phi - center_phi) +
                np.cos(theta) * np.cos(center_theta))
    cos_dist = np.clip(cos_dist, -1, 1)
    ang_dist = np.arccos(cos_dist)

    # Find the angular radius that gives the desired f_sky
    # f_sky = (1 - cos(radius)) / 2 for a cap
    cos_radius = 1 - 2 * f_sky
    if cos_radius < -1:
        cos_radius = -1
    radius = np.arccos(cos_radius)

    mask = ang_dist <= radius
    return mask


def generate_test2_dataset(
    n_maps: int,
    nside: int = NSIDE,
    lmax: int = LMAX,
    sigma_p: float = SIGMA_P,
    noise_std: float = 0.0,
    f_sky: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate a dataset of Q/U maps for Test 2.

    Args:
        n_maps: Number of map pairs to generate.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        sigma_p: Width of the Gaussian peaks.
        noise_std: White noise standard deviation.
        f_sky: Sky fraction (1.0 = full sky).
        seed: Random seed.

    Returns:
        Dict with keys: q_maps, u_maps, ell_ep, ell_bp, masks
    """
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)

    q_maps = np.zeros((n_maps, npix), dtype=np.float32)
    u_maps = np.zeros((n_maps, npix), dtype=np.float32)
    ell_ep = rng.uniform(LEP_MIN, LEP_MAX, size=n_maps).astype(np.float32)
    ell_bp = rng.uniform(LBP_MIN, LBP_MAX, size=n_maps).astype(np.float32)
    masks = np.zeros((n_maps, npix), dtype=bool)

    for i in range(n_maps):
        q, u = generate_polarization_map(
            ell_ep[i], ell_bp[i], nside, lmax, sigma_p, noise_std, rng
        )
        mask = create_sky_mask(f_sky, nside, rng)

        # Apply mask: zero out unobserved pixels
        q_maps[i] = q * mask
        u_maps[i] = u * mask
        masks[i] = mask

    return {
        "q_maps": q_maps,
        "u_maps": u_maps,
        "ell_ep": ell_ep,
        "ell_bp": ell_bp,
        "masks": masks,
    }
