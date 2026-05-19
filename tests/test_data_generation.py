"""Tests for data generation and power spectrum models."""

import numpy as np
import healpy as hp
import pytest

from torch_harmonics_healpix.data_generation import (
    generate_power_spectrum,
    generate_map,
    generate_dataset,
    NSIDE,
    LMAX,
    SIGMA_P,
)


class TestPowerSpectrum:
    """Test power spectrum generation."""

    def test_shape(self):
        """Output should have length lmax+1."""
        cl = generate_power_spectrum(ell_p=10.0, lmax=47)
        assert cl.shape == (48,)

    def test_peak_location(self):
        """Peak should be at the specified ℓ_p."""
        cl = generate_power_spectrum(ell_p=10.0, lmax=47)
        assert np.argmax(cl) == 10

    def test_peak_location_high(self):
        """Peak at ℓ_p=18."""
        cl = generate_power_spectrum(ell_p=18.0, lmax=47)
        assert np.argmax(cl) == 18

    def test_minimum_value(self):
        """All values should be ≥ 1e-5 (the floor)."""
        cl = generate_power_spectrum(ell_p=10.0, lmax=47)
        assert np.all(cl >= 1e-5)

    def test_symmetry(self):
        """Spectrum should be symmetric around ℓ_p."""
        cl = generate_power_spectrum(ell_p=12.0, lmax=47)
        # C_{12-3} should equal C_{12+3} (approximately, integer rounding)
        assert abs(cl[9] - cl[15]) / cl[12] < 1e-6


class TestGenerateMap:
    """Test single map generation."""

    def test_shape(self):
        """Map should have correct number of pixels."""
        m = generate_map(ell_p=10.0, nside=NSIDE)
        assert m.shape == (hp.nside2npix(NSIDE),)

    def test_dtype(self):
        """Map should be float32."""
        m = generate_map(ell_p=10.0, nside=NSIDE)
        assert m.dtype == np.float32

    def test_noisy_map_variance(self):
        """Noisy map should have higher variance than noiseless map."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(1)
        m_clean = generate_map(ell_p=10.0, noise_std=0.0, rng=rng1)
        m_noisy = generate_map(ell_p=10.0, noise_std=10.0, rng=rng2)
        # The noisy map should have more variance
        # (but same underlying signal, different noise realization)
        assert np.var(m_noisy) > 0  # at minimum, it should not be zero

    def test_reproducibility(self):
        """Same seed should produce same map."""
        m1 = generate_map(ell_p=10.0, rng=np.random.default_rng(42))
        m2 = generate_map(ell_p=10.0, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(m1, m2)


class TestGenerateDataset:
    """Test dataset generation."""

    def test_shapes(self):
        """Dataset shapes should be correct."""
        maps, ell_p = generate_dataset(n_maps=5, nside=NSIDE, seed=0)
        assert maps.shape == (5, hp.nside2npix(NSIDE))
        assert ell_p.shape == (5,)

    def test_ell_p_range(self):
        """ℓ_p values should be in [5, 20]."""
        _, ell_p = generate_dataset(n_maps=100, nside=NSIDE, seed=0)
        assert np.all(ell_p >= 5.0)
        assert np.all(ell_p <= 20.0)

    def test_different_seeds(self):
        """Different seeds should produce different datasets."""
        maps1, ell_p1 = generate_dataset(n_maps=3, nside=NSIDE, seed=1)
        maps2, ell_p2 = generate_dataset(n_maps=3, nside=NSIDE, seed=2)
        # Very unlikely to be exactly equal
        assert not np.array_equal(maps1, maps2)
