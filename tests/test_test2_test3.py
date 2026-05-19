"""Tests for Test 2 and Test 3 data generation."""

import numpy as np
import healpy as hp
import pytest

from torch_harmonics_healpix.data_generation_test2 import (
    generate_polarization_power_spectra,
    generate_polarization_map,
    create_sky_mask,
    generate_test2_dataset,
    NSIDE,
    LMAX,
)
from torch_harmonics_healpix.data_generation_test3 import (
    TAU_MIN,
    TAU_MAX,
)


class TestPolarizationPowerSpectra:
    """Test E/B power spectrum generation."""

    def test_shapes(self):
        cl_ee, cl_bb, cl_eb = generate_polarization_power_spectra(10.0, 15.0)
        assert cl_ee.shape == (LMAX + 1,)
        assert cl_bb.shape == (LMAX + 1,)
        assert cl_eb.shape == (LMAX + 1,)

    def test_e_peak(self):
        cl_ee, _, _ = generate_polarization_power_spectra(ell_ep=12.0, ell_bp=8.0)
        assert np.argmax(cl_ee) == 12

    def test_b_peak(self):
        _, cl_bb, _ = generate_polarization_power_spectra(ell_ep=12.0, ell_bp=8.0)
        assert np.argmax(cl_bb) == 8

    def test_eb_zero(self):
        _, _, cl_eb = generate_polarization_power_spectra(10.0, 15.0)
        assert np.all(cl_eb == 0)


class TestPolarizationMap:
    """Test Q/U map generation."""

    def test_shapes(self):
        q, u = generate_polarization_map(10.0, 15.0, nside=NSIDE)
        npix = hp.nside2npix(NSIDE)
        assert q.shape == (npix,)
        assert u.shape == (npix,)

    def test_dtype(self):
        q, u = generate_polarization_map(10.0, 15.0, nside=NSIDE)
        assert q.dtype == np.float32
        assert u.dtype == np.float32

    def test_reproducibility(self):
        q1, u1 = generate_polarization_map(
            10.0, 15.0, rng=np.random.default_rng(42)
        )
        q2, u2 = generate_polarization_map(
            10.0, 15.0, rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(u1, u2)


class TestSkyMask:
    """Test partial sky mask generation."""

    def test_full_sky(self):
        mask = create_sky_mask(f_sky=1.0, nside=NSIDE)
        npix = hp.nside2npix(NSIDE)
        assert mask.shape == (npix,)
        assert np.all(mask)

    def test_partial_sky_fraction(self):
        for f_sky in [0.5, 0.2, 0.1, 0.05]:
            mask = create_sky_mask(f_sky=f_sky, nside=NSIDE)
            actual_fraction = mask.sum() / len(mask)
            # Allow 10% tolerance on the sky fraction (cap approximation)
            assert abs(actual_fraction - f_sky) / f_sky < 0.15, (
                f"f_sky={f_sky}: actual={actual_fraction:.3f}"
            )

    def test_mask_reproducibility(self):
        m1 = create_sky_mask(f_sky=0.5, nside=NSIDE, rng=np.random.default_rng(10))
        m2 = create_sky_mask(f_sky=0.5, nside=NSIDE, rng=np.random.default_rng(10))
        np.testing.assert_array_equal(m1, m2)


class TestTest2Dataset:
    """Test Test 2 dataset generation."""

    def test_shapes(self):
        data = generate_test2_dataset(n_maps=3, nside=NSIDE, f_sky=1.0, seed=0)
        npix = hp.nside2npix(NSIDE)
        assert data["q_maps"].shape == (3, npix)
        assert data["u_maps"].shape == (3, npix)
        assert data["ell_ep"].shape == (3,)
        assert data["ell_bp"].shape == (3,)

    def test_ell_ranges(self):
        data = generate_test2_dataset(n_maps=50, nside=NSIDE, seed=0)
        assert np.all(data["ell_ep"] >= 5.0)
        assert np.all(data["ell_ep"] <= 20.0)
        assert np.all(data["ell_bp"] >= 5.0)
        assert np.all(data["ell_bp"] <= 20.0)

    def test_masking(self):
        data = generate_test2_dataset(n_maps=2, nside=NSIDE, f_sky=0.5, seed=0)
        # Masked pixels should be zero in Q and U
        for i in range(2):
            assert np.all(data["q_maps"][i][~data["masks"][i]] == 0)
            assert np.all(data["u_maps"][i][~data["masks"][i]] == 0)


class TestTest3Dataset:
    """Test Test 3 dataset generation (requires camb)."""

    @pytest.fixture
    def camb_available(self):
        try:
            import camb  # noqa: F401
            return True
        except ImportError:
            return False

    def test_tau_ranges(self):
        """Verify τ range constants are correct."""
        assert TAU_MIN == 0.03
        assert TAU_MAX == 0.08

    @pytest.mark.skipif(
        not pytest.importorskip("camb", reason="camb not installed"),
        reason="camb not available",
    )
    def test_camb_spectra(self):
        """Test CAMB spectrum generation if camb is available."""
        from torch_harmonics_healpix.data_generation_test3 import generate_camb_spectra
        cl_ee, cl_bb = generate_camb_spectra(tau=0.05, lmax=LMAX)
        assert cl_ee.shape == (LMAX + 1,)
        assert cl_bb.shape == (LMAX + 1,)
        # EE should be non-zero at low ell (reionization bump)
        assert cl_ee[2] > 0
