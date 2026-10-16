"""Tests for the MCMC baseline."""

import numpy as np
import pytest

from torch_harmonics_healpix.data_generation import generate_map, NSIDE, LMAX, SIGMA_P
from torch_harmonics_healpix.mcmc_baseline import (
    mcmc_estimate_ell_p,
    evaluate_mcmc_baseline,
)


class TestMCMCBaseline:
    """Test the maximum-likelihood ℓ_p estimator."""

    def test_perfect_spectrum(self):
        """On a noiseless map with ℓ_p=12, estimate should be close."""
        m = generate_map(ell_p=12.0, nside=NSIDE, noise_std=0.0, rng=np.random.default_rng(42))
        ell_p_est = mcmc_estimate_ell_p(m, sigma_p=SIGMA_P, lmax=LMAX, noise_std=0.0)
        # Should be within 2 of the true value (maps are random realizations)
        assert abs(ell_p_est - 12.0) < 3.0, f"Estimate {ell_p_est:.1f} too far from 12.0"

    def test_noisy_spectrum(self):
        """On a noisy map, estimate should still be in [5, 20] range."""
        m = generate_map(ell_p=10.0, nside=NSIDE, noise_std=5.0, rng=np.random.default_rng(42))
        ell_p_est = mcmc_estimate_ell_p(
            m, sigma_p=SIGMA_P, lmax=LMAX, noise_std=5.0
        )
        assert 5.0 <= ell_p_est <= 20.0, f"Estimate {ell_p_est:.1f} out of range"

    def test_bounds_enforced(self):
        """Estimate should always be in [5, 20]."""
        # Try a few different true values
        for ell_p_true in [5.5, 10.0, 19.5]:
            m = generate_map(ell_p=ell_p_true, nside=NSIDE, noise_std=0.0, rng=np.random.default_rng(99))
            ell_p_est = mcmc_estimate_ell_p(m)
            assert 5.0 <= ell_p_est <= 20.0

    def test_evaluate_mcmc_baseline_batch(self):
        """Batch helper: mean |Δℓ_p|/ℓ_p percentage error over maps."""
        true_values = [10.0, 12.0]
        maps = [generate_map(ell_p=lp, nside=NSIDE, noise_std=0.0,
                             rng=np.random.default_rng(42 + i))
                for i, lp in enumerate(true_values)]
        err = evaluate_mcmc_baseline(maps, ell_p_true=true_values,
                                     sigma_p=SIGMA_P, lmax=LMAX,
                                     noise_std=0.0, nside=NSIDE)
        assert np.isfinite(err)
        assert err >= 0
        # Same map twice → identical to a single-map evaluation
        single = evaluate_mcmc_baseline(maps[:1], ell_p_true=true_values[:1],
                                        sigma_p=SIGMA_P, lmax=LMAX,
                                        noise_std=0.0, nside=NSIDE)
        twice = evaluate_mcmc_baseline([maps[0], maps[0]],
                                       ell_p_true=true_values[:1] * 2,
                                       sigma_p=SIGMA_P, lmax=LMAX,
                                       noise_std=0.0, nside=NSIDE)
        assert single == pytest.approx(twice)
