"""Tests for the Test 2 / Test 4 MCMC baselines (CPU, no CAMB required).

  - mcmc_estimate_ell_ep_bp: exact-likelihood ℓ_Ep/ℓ_Bp recovery on full sky
  - evaluate_mcmc_test2: harness smoke test
  - mcmc_baseline_r_tau: joint (r, τ) grid search on a SYNTHETIC spectrum
    grid (physically-shaped EE ∝ τ² bump, BB ∝ r bump), so the recovery
    logic is tested without any CAMB dependency.
"""

import numpy as np
import healpy as hp
import pytest

from torch_harmonics_healpix.data_generation_test2 import (
    generate_polarization_map,
    NSIDE,
    LMAX,
    SIGMA_P,
)
from torch_harmonics_healpix.mcmc_baselines_test2_3 import (
    mcmc_estimate_ell_ep_bp,
    evaluate_mcmc_test2,
    mcmc_baseline_r_tau,
)

R_GRID = np.array([0.0, 0.01, 0.02])
TAU_GRID = np.array([0.03, 0.055, 0.08])


class TestMcmcTest2Estimator:
    """Exact-likelihood ℓ_Ep/ℓ_Bp estimation from Q/U maps."""

    def test_recovery_full_sky(self):
        """On a noiseless full-sky map the fit should land near the truth."""
        ell_ep_true, ell_bp_true = 12.0, 8.0
        q, u = generate_polarization_map(
            ell_ep_true, ell_bp_true, NSIDE, LMAX, SIGMA_P, 0.0,
            np.random.default_rng(42),
        )
        ell_ep, ell_bp = mcmc_estimate_ell_ep_bp(q, u, SIGMA_P, LMAX, NSIDE, 1.0)
        assert abs(ell_ep - ell_ep_true) < 3.0, \
            f"ell_ep estimate {ell_ep:.1f} too far from {ell_ep_true}"
        assert abs(ell_bp - ell_bp_true) < 3.0, \
            f"ell_bp estimate {ell_bp:.1f} too far from {ell_bp_true}"

    def test_bounds_enforced(self):
        """Estimates must stay inside the [5, 20] fit bounds."""
        for ep, bp in [(5.5, 5.5), (19.5, 19.5)]:
            q, u = generate_polarization_map(
                ep, bp, NSIDE, LMAX, SIGMA_P, 0.0, np.random.default_rng(1)
            )
            est_ep, est_bp = mcmc_estimate_ell_ep_bp(q, u, SIGMA_P, LMAX, NSIDE, 1.0)
            assert 5.0 <= est_ep <= 20.0
            assert 5.0 <= est_bp <= 20.0

    def test_partial_sky_runs(self):
        """With the f_sky correction the estimator must still run and stay bounded
        (accuracy degrades at low f_sky by design — E/B leakage is uncorrected)."""
        from torch_harmonics_healpix.data_generation_test2 import create_sky_mask

        q, u = generate_polarization_map(
            12.0, 8.0, NSIDE, LMAX, SIGMA_P, 0.0, np.random.default_rng(3)
        )
        mask = create_sky_mask(0.5, NSIDE, np.random.default_rng(3)).astype(np.float32)
        est_ep, est_bp = mcmc_estimate_ell_ep_bp(
            q * mask, u * mask, SIGMA_P, LMAX, NSIDE, 0.5
        )
        assert 5.0 <= est_ep <= 20.0
        assert 5.0 <= est_bp <= 20.0


class TestEvaluateMcmcTest2:
    """Smoke test of the evaluation harness."""

    def test_smoke_full_sky(self):
        result = evaluate_mcmc_test2(n_maps=2, f_sky=1.0, seed=42)
        assert result["n_maps"] == 2
        assert result["f_sky"] == 1.0
        assert np.isfinite(result["ep_pct_error"])
        assert np.isfinite(result["bp_pct_error"])
        assert result["ep_pct_error"] >= 0
        assert result["bp_pct_error"] >= 0
        assert result["time_per_map"] > 0


def _synthetic_grid():
    """Build a synthetic (r, τ) spectrum grid with the library's flattening
    convention (index = tau_idx * n_r + r_idx).

    EE carries a low-ℓ bump with amplitude ∝ τ² (reionization-like);
    BB carries a low-ℓ bump with amplitude ∝ r (tensor-like).
    """
    ell = np.arange(LMAX + 1)
    ee_bump = np.exp(-((ell - 5.0) ** 2) / (2 * 4.0**2))
    bb_bump = np.exp(-((ell - 5.0) ** 2) / (2 * 8.0**2))

    n_r, n_tau = len(R_GRID), len(TAU_GRID)
    cl_ee = np.zeros((n_r * n_tau, LMAX + 1))
    cl_bb = np.zeros((n_r * n_tau, LMAX + 1))
    for i, tau in enumerate(TAU_GRID):
        for j, r in enumerate(R_GRID):
            idx = i * n_r + j
            cl_ee[idx] = (tau / 0.08) ** 2 * ee_bump + 1e-7
            cl_bb[idx] = (r / 0.01) * bb_bump + 1e-8
    return cl_ee, cl_bb


def _synthetic_map(cl_ee_row, cl_bb_row, seed):
    """Synthesize a Q/U map from one grid spectrum (library conventions)."""
    cl_tt = np.full(LMAX + 1, 1e-5)
    zeros = np.zeros(LMAX + 1)
    cl_full = np.array([cl_tt, cl_ee_row, cl_bb_row, zeros, zeros, zeros])
    np.random.seed(seed)
    maps = hp.synfast(cl_full, nside=NSIDE, lmax=LMAX, new=True)
    return maps[1].astype(np.float32), maps[2].astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_grid():
    return _synthetic_grid()


class TestMcmcRTau:
    """Joint (r, τ) grid search on the synthetic grid."""

    def test_recovers_true_cell_full_sky(self, synthetic_grid):
        """A clean map generated at (r=0.01, τ=0.055) must be fit back to
        the same grid cell."""
        cl_ee, cl_bb = synthetic_grid
        true_idx = 1 * len(R_GRID) + 1  # tau_idx=1, r_idx=1
        q, u = _synthetic_map(cl_ee[true_idx], cl_bb[true_idx], seed=11)

        mask = np.ones(hp.nside2npix(NSIDE), dtype=np.float32)
        result = mcmc_baseline_r_tau(
            q, u, mask, R_GRID, TAU_GRID, cl_ee, cl_bb,
            noise_std=0.0, nside=NSIDE, lmax=LMAX,
        )
        assert result["r_best"] == pytest.approx(0.01), \
            f"r_best {result['r_best']} != 0.01"
        assert result["tau_best"] == pytest.approx(0.055), \
            f"tau_best {result['tau_best']} != 0.055"

    def test_grid_shapes_and_nan_convention(self, synthetic_grid):
        """chi2_grid has shape (n_tau, n_r); pct errors are NaN by contract."""
        cl_ee, cl_bb = synthetic_grid
        true_idx = 2 * len(R_GRID) + 0  # tau_idx=2, r_idx=0
        q, u = _synthetic_map(cl_ee[true_idx], cl_bb[true_idx], seed=5)

        mask = np.ones(hp.nside2npix(NSIDE), dtype=np.float32)
        result = mcmc_baseline_r_tau(
            q, u, mask, R_GRID, TAU_GRID, cl_ee, cl_bb,
            noise_std=0.0, nside=NSIDE, lmax=LMAX,
        )
        assert result["chi2_grid"].shape == (len(TAU_GRID), len(R_GRID))
        assert np.isfinite(result["chi2_grid"]).all()
        assert np.isnan(result["r_pct_error"]), \
            "pct errors must be NaN; the caller computes them from truth"
        assert np.isnan(result["tau_pct_error"])
        # The minimum of chi2_grid must correspond to the returned best fit
        i, j = np.unravel_index(np.argmin(result["chi2_grid"]),
                                result["chi2_grid"].shape)
        assert result["r_best"] == pytest.approx(R_GRID[j])
        assert result["tau_best"] == pytest.approx(TAU_GRID[i])

    def test_partial_sky_plumbing(self, synthetic_grid):
        """On a masked sky the estimator must still return valid grid values
        (accuracy degrades; plumbing must not break)."""
        cl_ee, cl_bb = synthetic_grid
        true_idx = 1 * len(R_GRID) + 1
        q, u = _synthetic_map(cl_ee[true_idx], cl_bb[true_idx], seed=13)

        mask = np.zeros(hp.nside2npix(NSIDE), dtype=np.float32)
        mask[: mask.size // 2] = 1.0
        result = mcmc_baseline_r_tau(
            q * mask, u * mask, mask, R_GRID, TAU_GRID, cl_ee, cl_bb,
            noise_std=0.0, nside=NSIDE, lmax=LMAX,
        )
        assert result["r_best"] in R_GRID
        assert result["tau_best"] in TAU_GRID
