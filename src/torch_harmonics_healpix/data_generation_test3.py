"""Data generation for Test 3: τ estimation from Q/U polarization maps.

Reproduces the benchmark from Krachmalnicoff & Tomasi (2019) Section 6.1.3.
Uses realistic CAMB power spectra with varying optical depth τ ∈ [0.03, 0.08].

Note: This module requires the `camb` package to generate theoretical spectra.
If camb is not available, you can pre-generate spectra and load from files.
"""

import numpy as np
import healpy as hp

from .data_generation_test2 import create_sky_mask

NSIDE = 16
LMAX = 3 * NSIDE - 1  # 47
TAU_MIN = 0.03
TAU_MAX = 0.08

# Planck 2018 best-fit (other parameters fixed, only τ varies)
# These are approximate values — exact values depend on CAMB version
PLANCK_PARAMS = {
    "H0": 67.4,
    "ombh2": 0.0224,
    "omch2": 0.120,
    "As": 2.1e-9,
    "ns": 0.965,
}


def generate_camb_spectra(tau: float, lmax: int = LMAX) -> tuple[np.ndarray, np.ndarray]:
    """Generate CAMB E-mode and B-mode power spectra for given τ.

    Args:
        tau: Optical depth to reionization.
        lmax: Maximum multipole.

    Returns:
        Tuple of (cl_ee, cl_bb) arrays.

    Raises:
        ImportError: If camb is not installed.
    """
    try:
        import camb
    except ImportError:
        raise ImportError(
            "CAMB is required for Test 3 data generation. "
            "Install with: pip install camb"
        )

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=PLANCK_PARAMS["H0"],
        ombh2=PLANCK_PARAMS["ombh2"],
        omch2=PLANCK_PARAMS["omch2"],
        tau=tau,
    )
    pars.InitPower.set_params(As=PLANCK_PARAMS["As"], ns=PLANCK_PARAMS["ns"])
    pars.WantTensors = True
    pars.set_for_lmax(lmax)

    results = camb.get_transfer_functions(pars)
    cls = results.get_total_cls(lmax, CMB_unit="muK")

    cl_ee = cls[:, 1]  # EE
    cl_bb = cls[:, 2]  # BB (from tensors, very small for standard ΛCDM)

    # Pad or truncate to lmax+1
    if len(cl_ee) < lmax + 1:
        cl_ee = np.pad(cl_ee, (0, lmax + 1 - len(cl_ee)))
        cl_bb = np.pad(cl_bb, (0, lmax + 1 - len(cl_bb)))
    else:
        cl_ee = cl_ee[: lmax + 1]
        cl_bb = cl_bb[: lmax + 1]

    return cl_ee, cl_bb


def generate_tau_map(
    tau: float,
    nside: int = NSIDE,
    lmax: int = LMAX,
    noise_std: float = 0.0,
    f_sky: float = 1.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Q/U maps from CAMB spectra with given τ.

    Args:
        tau: Optical depth to reionization.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        noise_std: White noise standard deviation (in μK).
        f_sky: Sky fraction.
        rng: NumPy random Generator.

    Returns:
        Tuple of (q_map, u_map, mask), each 1D HEALPix array.
    """
    cl_ee, cl_bb = generate_camb_spectra(tau, lmax)

    # Build full TEB power spectrum
    cl_tt = np.full(lmax + 1, 1e-5)  # No T signal for this test
    cl_te = np.zeros(lmax + 1)
    cl_eb = np.zeros(lmax + 1)
    cl_tb = np.zeros(lmax + 1)

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
            q_map += rng.normal(0, noise_std, size=q_map.shape).astype(np.float32)
            u_map += rng.normal(0, noise_std, size=u_map.shape).astype(np.float32)
        else:
            q_map += np.random.normal(0, noise_std, size=q_map.shape).astype(np.float32)
            u_map += np.random.normal(0, noise_std, size=u_map.shape).astype(np.float32)

    mask = create_sky_mask(f_sky, nside, rng)
    q_map *= mask
    u_map *= mask

    return q_map, u_map, mask


def generate_test3_dataset(
    n_maps: int,
    nside: int = NSIDE,
    lmax: int = LMAX,
    noise_std: float = 0.0,
    f_sky: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate a dataset of Q/U maps for Test 3 (τ estimation).

    Args:
        n_maps: Number of map pairs.
        nside: HEALPix Nside.
        lmax: Maximum multipole.
        noise_std: White noise standard deviation (μK).
        f_sky: Sky fraction.
        seed: Random seed.

    Returns:
        Dict with keys: q_maps, u_maps, tau_values, masks
    """
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)

    q_maps = np.zeros((n_maps, npix), dtype=np.float32)
    u_maps = np.zeros((n_maps, npix), dtype=np.float32)
    tau_values = rng.uniform(TAU_MIN, TAU_MAX, size=n_maps).astype(np.float32)
    masks = np.zeros((n_maps, npix), dtype=bool)

    for i in range(n_maps):
        q, u, mask = generate_tau_map(
            tau_values[i], nside, lmax, noise_std, f_sky, rng
        )
        q_maps[i] = q
        u_maps[i] = u
        masks[i] = mask

    return {
        "q_maps": q_maps,
        "u_maps": u_maps,
        "tau_values": tau_values,
        "masks": masks,
    }
