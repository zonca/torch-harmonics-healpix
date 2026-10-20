"""Tests for CAMB-based data generation (Tests 3 and 4).

The whole module requires the optional `camb` dependency — it is skipped
if camb is not installed. CAMB calls at lmax=47 take ~2 s each; this
module makes ~18 of them (~35 s), shared via module-scoped fixtures.

Coverage:
  - generate_camb_spectra (Test 3): shape, τ → reionization-bump monotonicity
  - generate_camb_spectra_r_tau (Test 4): shape, r → BB scaling, raw C_ℓ convention
  - generate_tau_map / generate_r_tau_map: shapes, masking, noise, reproducibility
  - precompute_camb_spectra: disk cache (write once, read back identical)
  - generate_test3_dataset: end-to-end with a monkeypatched tiny spectrum grid
  - evaluate_mcmc_test3: smoke test with a tiny grid
"""

import sys
import builtins

import numpy as np
import healpy as hp
import pytest

camb = pytest.importorskip("camb", reason="camb not installed")

import torch_harmonics_healpix.data_generation_test3 as dg3
import torch_harmonics_healpix.data_generation_test4 as dg4
from torch_harmonics_healpix.data_generation_test3 import (
    generate_camb_spectra,
    generate_tau_map,
    precompute_camb_spectra,
    TAU_MIN,
    TAU_MAX,
    LMAX,
    NSIDE,
)
from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau,
    generate_r_tau_map,
    precompute_camb_spectra_r_tau,
    R_MAX,
    TAU_MIN as TAU4_MIN,
    TAU_MAX as TAU4_MAX,
)


# --- Module-scoped CAMB fixtures (each = one ~2 s CAMB call) ---------------

@pytest.fixture(scope="module")
def ee_bb_tau_mid():
    """(EE, BB) at τ=0.05, r=0 — shared by shape / lensing-BB / map tests."""
    return generate_camb_spectra(tau=0.05, lmax=LMAX)


@pytest.fixture(scope="module")
def ee_tau_low_high():
    """EE at τ=TAU_MIN and τ=TAU_MAX for the reionization-bump monotonicity."""
    ee_low, _ = generate_camb_spectra(tau=TAU_MIN, lmax=LMAX)
    ee_high, _ = generate_camb_spectra(tau=TAU_MAX, lmax=LMAX)
    return ee_low, ee_high


@pytest.fixture(scope="module")
def ee_bb_r_zero():
    """(EE, BB) at r=0, τ=0.05 — lensing-only BB and the raw-C_ℓ check."""
    return generate_camb_spectra_r_tau(r=0.0, tau=0.05, lmax=LMAX)


@pytest.fixture(scope="module")
def bb_r_max():
    """BB at r=R_MAX, τ=0.05 — tensor B modes."""
    _, bb = generate_camb_spectra_r_tau(r=R_MAX, tau=0.05, lmax=LMAX)
    return bb


# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def camb_cache_dir(tmp_path_factory):
    """One cache dir shared by the whole module (pre-warmed once)."""
    return tmp_path_factory.mktemp("camb_cache")


@pytest.fixture
def tiny_camb_cache(camb_cache_dir, monkeypatch):
    """Redirect the CAMB spectrum cache to a temp dir and shrink the grid.

    N_CAMB_SPECTRA (5000 by default) is looked up at call time from the
    module namespace, so monkeypatching the module attribute shrinks every
    downstream computation (dataset gen, MCMC evaluation) to CI size.

    The seed-142 grid used by generate_test3_dataset(seed=42) and
    evaluate_mcmc_test3(seed=42) is pre-warmed with n_workers=1 (serial,
    in-process — the fork-based pool deadlocks under pytest, which has
    torch's threads loaded at collection time). Subsequent calls hit the
    cache and never reach the pool.
    """
    monkeypatch.setenv("THH_CAMB_CACHE_DIR", str(camb_cache_dir))
    monkeypatch.setattr(dg3, "N_CAMB_SPECTRA", 4)
    precompute_camb_spectra(n_spectra=4, lmax=LMAX, seed=142, n_workers=1,
                            cache_dir=str(camb_cache_dir))
    return camb_cache_dir


class TestCamBTest3Spectra:
    """CAMB E/B spectra as a function of τ (Test 3)."""

    def test_shape_and_positivity(self, ee_bb_tau_mid):
        cl_ee, cl_bb = ee_bb_tau_mid
        assert cl_ee.shape == (LMAX + 1,)
        assert cl_bb.shape == (LMAX + 1,)
        # E modes vanish identically at ℓ=0,1; positive from ℓ=2 up
        assert np.all(cl_ee[2:] > 0), "EE spectrum must be positive for ℓ≥2"
        assert np.all(cl_bb >= 0), "BB spectrum must be non-negative"

    def test_reionization_bump_grows_with_tau(self, ee_tau_low_high):
        """Larger τ → stronger large-scale EE (reionization bump)."""
        ee_low, ee_high = ee_tau_low_high
        # The bump lives at ℓ ≲ 10; compare the band-averaged amplitude.
        low_band = ee_low[2:11].mean()
        high_band = ee_high[2:11].mean()
        assert high_band > low_band, (
            f"EE bump should grow with tau: {high_band:.3e} !> {low_band:.3e}"
        )

    def test_bb_is_lensing_small(self, ee_bb_tau_mid):
        """With WantTensors but r=0, BB is lensing-only: tiny vs EE."""
        cl_ee, cl_bb = ee_bb_tau_mid
        assert cl_bb[2:20].max() < cl_ee[2:20].max(), \
            "r=0 BB (lensing only) should be far below EE"


class TestCamBTest4Spectra:
    """CAMB E/B spectra as a function of (r, τ) (Test 4)."""

    def test_shape(self, ee_bb_r_zero):
        cl_ee, cl_bb = ee_bb_r_zero
        assert cl_ee.shape == (LMAX + 1,)
        assert cl_bb.shape == (LMAX + 1,)
        assert np.all(cl_bb >= 0)

    def test_bb_scales_with_r(self, ee_bb_r_zero, bb_r_max):
        """Tensor BB amplitude grows with r (r=0.01 vs r=0)."""
        _, bb_zero = ee_bb_r_zero
        # Large-scale tensor BB at r=0.01 dominates the lensing floor.
        assert bb_r_max[2:20].mean() > 10 * bb_zero[2:20].mean(), (
            f"BB(r=0.01) should be >> BB(r=0): "
            f"{bb_r_max[2:20].mean():.3e} vs {bb_zero[2:20].mean():.3e}"
        )

    def test_raw_cl_convention(self, ee_bb_r_zero):
        """The D_ℓ→C_ℓ fix: C_ℓ must decay at high ℓ, D_ℓ must stay flat.

        Regression guard for the amplitude bug documented in AGENTS.md:
        CAMB's default returns D_ℓ, but hp.synfast expects C_ℓ.
        """
        cl_ee, _ = ee_bb_r_zero
        ell = np.arange(LMAX + 1)
        d_ell = ell * (ell + 1) * cl_ee / (2 * np.pi)
        # C_ℓ falls by an order of magnitude from the bump to ℓ=47 for EE
        assert cl_ee[47] < 0.1 * cl_ee[5], "C_ell should decay at high ell"
        # D_ℓ stays within an order of magnitude over the same range
        assert 0.1 < d_ell[47] / d_ell[5] < 10, (
            "D_ell should be roughly flat — if this fails, raw_cl was dropped"
        )


class TestTauMapGeneration:
    """generate_tau_map with pre-computed spectra (no CAMB call)."""

    def test_shapes_and_dtype(self, ee_bb_tau_mid):
        cl_ee, cl_bb = ee_bb_tau_mid
        q, u, mask = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                      cl_ee=cl_ee, cl_bb=cl_bb)
        npix = hp.nside2npix(NSIDE)
        assert q.shape == (npix,) and u.shape == (npix,)
        assert q.dtype == np.float32 and u.dtype == np.float32
        assert mask.shape == (npix,)
        assert np.all(mask), "f_sky=1 mask must be all-observed"

    def test_masking(self, ee_bb_tau_mid):
        cl_ee, cl_bb = ee_bb_tau_mid
        q, u, mask = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 0.5, None,
                                      cl_ee=cl_ee, cl_bb=cl_bb)
        assert np.all(q[~mask] == 0), "Q must be zero outside the mask"
        assert np.all(u[~mask] == 0), "U must be zero outside the mask"
        assert 0.3 < mask.mean() < 0.7, "f_sky=0.5 mask fraction sanity"

    def test_reproducibility(self, ee_bb_tau_mid):
        cl_ee, cl_bb = ee_bb_tau_mid
        q1, u1, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0,
                                     np.random.default_rng(7),
                                     cl_ee=cl_ee, cl_bb=cl_bb)
        q2, u2, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0,
                                     np.random.default_rng(7),
                                     cl_ee=cl_ee, cl_bb=cl_bb)
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(u1, u2)

    def test_noise_increases_variance(self, ee_bb_tau_mid):
        cl_ee, cl_bb = ee_bb_tau_mid
        q0, u0, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0,
                                     np.random.default_rng(1),
                                     cl_ee=cl_ee, cl_bb=cl_bb)
        q1, u1, _ = generate_tau_map(0.05, NSIDE, LMAX, 2.0, 1.0,
                                     np.random.default_rng(1),
                                     cl_ee=cl_ee, cl_bb=cl_bb)
        assert np.var(q1) > np.var(q0), "Noise must raise the map variance"

    def test_camb_spectra_generated_when_not_provided(self):
        """cl_ee=None must trigger the internal CAMB call."""
        q, u, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0, None)
        npix = hp.nside2npix(NSIDE)
        assert q.shape == (npix,) and u.shape == (npix,)
        assert np.isfinite(q).all() and np.isfinite(u).all()

    def test_noise_without_rng_uses_legacy_global(self, ee_bb_tau_mid):
        """rng=None + noise falls back to np.random; deterministic only
        under an explicit global seed."""
        cl_ee, cl_bb = ee_bb_tau_mid
        np.random.seed(5)
        q1, u1, _ = generate_tau_map(0.05, NSIDE, LMAX, 2.0, 1.0, None,
                                     cl_ee=cl_ee, cl_bb=cl_bb)
        np.random.seed(5)
        q2, u2, _ = generate_tau_map(0.05, NSIDE, LMAX, 2.0, 1.0, None,
                                     cl_ee=cl_ee, cl_bb=cl_bb)
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(u1, u2)

    def test_missing_bb_regenerates_only_bb(self, ee_bb_tau_mid, monkeypatch):
        """cl_ee given, cl_bb=None: ONLY the missing BB may be regenerated —
        the caller's EE must reach hp.synfast unmodified (regression guard:
        a previous bug regenerated both and discarded the provided EE)."""
        cl_ee, _ = ee_bb_tau_mid
        captured = {}
        real_synfast = hp.synfast

        def recording_synfast(cl_full, **kwargs):
            captured["ee_row"] = np.array(cl_full[1])
            return real_synfast(cl_full, **kwargs)

        monkeypatch.setattr(hp, "synfast", recording_synfast)
        q, u, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0,
                                   np.random.default_rng(1),
                                   cl_ee=cl_ee, cl_bb=None)
        np.testing.assert_array_equal(captured["ee_row"], cl_ee), \
            "Provided cl_ee was overwritten by the BB regeneration"
        assert q.shape == (hp.nside2npix(NSIDE),)
        assert np.isfinite(q).all()


class TestRTauMapGeneration:
    """generate_r_tau_map with pre-computed spectra (no CAMB call)."""

    def test_shapes(self, ee_bb_r_zero):
        cl_ee, cl_bb = ee_bb_r_zero
        q, u, mask = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0,
                                        None, cl_ee=cl_ee, cl_bb=cl_bb)
        npix = hp.nside2npix(NSIDE)
        assert q.shape == (npix,) and u.shape == (npix,) and mask.shape == (npix,)

    def test_shared_mask(self, ee_bb_r_zero):
        """A caller-supplied mask must be used verbatim (mask-channel contract)."""
        cl_ee, cl_bb = ee_bb_r_zero
        mask_in = np.zeros(hp.nside2npix(NSIDE), dtype=np.float32)
        mask_in[: mask_in.size // 2] = 1.0
        q, u, mask_out = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0,
                                            None, cl_ee=cl_ee, cl_bb=cl_bb,
                                            mask=mask_in)
        np.testing.assert_array_equal(mask_in, mask_out)
        assert np.all(q[mask_out == 0] == 0)

    def test_camb_spectra_generated_when_not_provided(self):
        """cl_ee=None must trigger the internal CAMB call (r=0, τ=0.05)."""
        q, u, mask = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0, None)
        npix = hp.nside2npix(NSIDE)
        assert q.shape == (npix,) and u.shape == (npix,)
        assert np.isfinite(q).all() and np.isfinite(u).all()

    def test_rng_reproducible_and_global_state_preserved(self, ee_bb_r_zero):
        """Same-seed Generators give identical maps, and hp.synfast's legacy
        global RNG must be fully restored afterwards (save/restore contract)."""
        cl_ee, cl_bb = ee_bb_r_zero
        state_before = np.random.get_state()

        q1, u1, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0,
                                       np.random.default_rng(7),
                                       cl_ee=cl_ee, cl_bb=cl_bb)
        q2, u2, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0,
                                       np.random.default_rng(7),
                                       cl_ee=cl_ee, cl_bb=cl_bb)
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(u1, u2)

        state_after = np.random.get_state()
        # MT19937 state tuple: (name, 624 keys, pos, has_gauss, cached)
        name_a, keys_a, pos_a, has_gauss_a, _ = np.random.get_state()
        name_b, keys_b, pos_b, has_gauss_b, _ = state_before
        assert name_a == name_b
        np.testing.assert_array_equal(keys_a, keys_b)
        assert pos_a == pos_b and has_gauss_a == has_gauss_b

    def test_noise_with_rng_increases_variance(self, ee_bb_r_zero):
        cl_ee, cl_bb = ee_bb_r_zero
        q0, u0, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0,
                                       np.random.default_rng(1),
                                       cl_ee=cl_ee, cl_bb=cl_bb)
        q1, u1, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 2.0, 1.0,
                                       np.random.default_rng(1),
                                       cl_ee=cl_ee, cl_bb=cl_bb)
        assert np.var(q1) > np.var(q0), "Noise must raise the map variance"
        assert np.var(u1) > np.var(u0)

    def test_noise_without_rng_uses_legacy_global(self, ee_bb_r_zero):
        """rng=None + noise falls back to np.random; deterministic only
        under an explicit global seed."""
        cl_ee, cl_bb = ee_bb_r_zero
        np.random.seed(3)
        q1, u1, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 2.0, 1.0,
                                       None, cl_ee=cl_ee, cl_bb=cl_bb)
        np.random.seed(3)
        q2, u2, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 2.0, 1.0,
                                       None, cl_ee=cl_ee, cl_bb=cl_bb)
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(u1, u2)


class TestPartialSpectrumProvision:
    """Partial spectrum provision: only the missing spectrum is regenerated.

    Regression tests: generate_r_tau_map used to regenerate BOTH spectra
    whenever either was missing (silently discarding the one provided), and
    generate_tau_map discarded a provided cl_bb when cl_ee was missing.
    CAMB is monkeypatched with an amplitude sentinel: a regenerated
    spectrum is tiny (1e-6), so any visible map power proves the provided
    (unit-amplitude) spectrum was the one actually used.
    """

    def _spectra(self, amp):
        ell = np.arange(LMAX + 1)
        cl = amp * np.exp(-((ell - 8.0) ** 2) / (2 * 4.0**2)) + 1e-8
        return cl

    def test_tau_map_provided_ee(self, monkeypatch):
        calls = []

        def fake_gen(tau, lmax):
            calls.append(tau)
            return self._spectra(1e-6), self._spectra(1e-6)

        monkeypatch.setattr(dg3, "generate_camb_spectra", fake_gen)
        q, u, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                   cl_ee=self._spectra(1.0), cl_bb=None)
        assert len(calls) == 1, "must regenerate only the missing BB"
        assert np.std(q) > 0.05 and np.std(u) > 0.05, \
            "provided EE must drive the map power"

    def test_tau_map_provided_bb(self, monkeypatch):
        calls = []

        def fake_gen(tau, lmax):
            calls.append(tau)
            return self._spectra(1e-6), self._spectra(1e-6)

        monkeypatch.setattr(dg3, "generate_camb_spectra", fake_gen)
        q, u, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                   cl_ee=None, cl_bb=self._spectra(1.0))
        assert len(calls) == 1, "must regenerate only the missing EE"
        assert np.std(q) > 0.05 and np.std(u) > 0.05, \
            "provided BB must drive the map power"

    def test_tau_map_both_provided_no_camb(self, monkeypatch):
        def fail_gen(tau, lmax):
            raise AssertionError("CAMB must not be called when both "
                                 "spectra are provided")

        monkeypatch.setattr(dg3, "generate_camb_spectra", fail_gen)
        q, u, _ = generate_tau_map(0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                   cl_ee=self._spectra(1.0),
                                   cl_bb=self._spectra(1.0))
        assert np.std(q) > 0.05

    def test_r_tau_map_provided_ee(self, monkeypatch):
        calls = []

        def fake_gen(r, tau, lmax):
            calls.append((r, tau))
            return self._spectra(1e-6), self._spectra(1e-6)

        monkeypatch.setattr(dg4, "generate_camb_spectra_r_tau", fake_gen)
        q, u, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                     cl_ee=self._spectra(1.0), cl_bb=None)
        assert len(calls) == 1, "must regenerate only the missing BB"
        assert np.std(q) > 0.05, "provided EE must drive the map power"

    def test_r_tau_map_provided_bb(self, monkeypatch):
        calls = []

        def fake_gen(r, tau, lmax):
            calls.append((r, tau))
            return self._spectra(1e-6), self._spectra(1e-6)

        monkeypatch.setattr(dg4, "generate_camb_spectra_r_tau", fake_gen)
        q, u, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                     cl_ee=None, cl_bb=self._spectra(1.0))
        assert len(calls) == 1, "must regenerate only the missing EE"
        assert np.std(q) > 0.05, "provided BB must drive the map power"

    def test_r_tau_map_both_provided_no_camb(self, monkeypatch):
        def fail_gen(r, tau, lmax):
            raise AssertionError("CAMB must not be called when both "
                                 "spectra are provided")

        monkeypatch.setattr(dg4, "generate_camb_spectra_r_tau", fail_gen)
        q, u, _ = generate_r_tau_map(0.0, 0.05, NSIDE, LMAX, 0.0, 1.0, None,
                                     cl_ee=self._spectra(1.0),
                                     cl_bb=self._spectra(1.0))
        assert np.std(q) > 0.05


class TestPrecomputeCache:
    """precompute_camb_spectra disk cache (tiny grid, serial in-process)."""

    def test_shapes_and_cache_roundtrip(self, tiny_camb_cache):
        tau, ee, bb = precompute_camb_spectra(
            n_spectra=4, lmax=LMAX, seed=99, n_workers=1,
            cache_dir=str(tiny_camb_cache),
        )
        assert tau.shape == (4,)
        assert ee.shape == (4, LMAX + 1)
        assert bb.shape == (4, LMAX + 1)
        assert np.all((tau >= TAU_MIN) & (tau <= TAU_MAX))

        cache_file = tiny_camb_cache / "camb_ee_bb_n4_lmax47_seed99.npz"
        assert cache_file.exists(), "Cache file was not written"

        # Second call must hit the cache and return identical data
        tau2, ee2, bb2 = precompute_camb_spectra(
            n_spectra=4, lmax=LMAX, seed=99, n_workers=1,
            cache_dir=str(tiny_camb_cache),
        )
        np.testing.assert_array_equal(tau, tau2)
        np.testing.assert_array_equal(ee, ee2)
        np.testing.assert_array_equal(bb, bb2)

    def test_env_var_cache_dir(self, tiny_camb_cache):
        """With cache_dir=None the THH_CAMB_CACHE_DIR env var must be used."""
        precompute_camb_spectra(n_spectra=2, lmax=LMAX, seed=7, n_workers=1)
        cache_file = tiny_camb_cache / "camb_ee_bb_n2_lmax47_seed7.npz"
        assert cache_file.exists(), "Env-var cache dir was not used"

    def test_pool_default_workers_subprocess(self, tiny_camb_cache):
        """The multiprocessing pool path (default n_workers, cpu-based) must
        work in a clean process. Run in a subprocess because forking from
        pytest — which has torch's thread pools loaded at collection time —
        can deadlock."""
        import subprocess

        script = (
            "from torch_harmonics_healpix.data_generation_test3 import "
            "precompute_camb_spectra, LMAX;"
            f"precompute_camb_spectra(n_spectra=2, lmax=LMAX, seed=21, "
            f"cache_dir={str(tiny_camb_cache)!r})"
        )
        subprocess.run([sys.executable, "-c", script], check=True, timeout=300)
        cache_file = tiny_camb_cache / "camb_ee_bb_n2_lmax47_seed21.npz"
        assert cache_file.exists(), "Pool path did not write the cache"

    def test_default_workers_serial_on_single_core(self, tiny_camb_cache,
                                                   monkeypatch):
        """n_workers=None resolves through os.cpu_count(); on a single-core
        machine it must resolve to 1 → serial in-process path (no fork)."""
        monkeypatch.setattr("os.cpu_count", lambda: 1)
        precompute_camb_spectra(n_spectra=2, lmax=LMAX, seed=31)
        cache_file = tiny_camb_cache / "camb_ee_bb_n2_lmax47_seed31.npz"
        assert cache_file.exists(), "Single-core default did not run serially"


class TestTest3Dataset:
    """End-to-end Test 3 dataset generation with a tiny spectrum grid."""

    def test_dataset_shapes_and_ranges(self, tiny_camb_cache):
        data = dg3.generate_test3_dataset(n_maps=2, seed=42)
        npix = hp.nside2npix(NSIDE)
        assert data["q_maps"].shape == (2, npix)
        assert data["u_maps"].shape == (2, npix)
        assert data["tau_values"].shape == (2,)
        assert data["masks"].shape == (2, npix)
        assert np.all((data["tau_values"] >= TAU_MIN)
                      & (data["tau_values"] <= TAU_MAX))
        for i in range(2):
            assert np.all(data["q_maps"][i][~data["masks"][i]] == 0)


class TestEvaluateMcmcTest3:
    """Smoke test of the Test 3 exact-likelihood MCMC harness (tiny grid)."""

    def test_smoke(self, tiny_camb_cache):
        from torch_harmonics_healpix.mcmc_baselines_test2_3 import evaluate_mcmc_test3

        result = evaluate_mcmc_test3(n_maps=2, seed=42)
        assert result["n_maps"] == 2
        assert np.isfinite(result["tau_pct_error"])
        assert result["tau_pct_error"] >= 0
        assert result["time_per_map"] > 0


# ---------------------------------------------------------------------------
# Error paths, defensive padding, and the r/τ precompute grid
# ---------------------------------------------------------------------------

class TestCambImportError:
    """Without camb importable the generators must raise a clear ImportError
    (not an opaque NameError from the failed import)."""

    @pytest.fixture(autouse=True)
    def _hide_camb(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "camb":
                raise ImportError("simulated: camb not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "camb", raising=False)

    def test_test3_raises(self):
        with pytest.raises(ImportError, match="CAMB is required for Test 3"):
            generate_camb_spectra(tau=0.05, lmax=LMAX)

    def test_test4_raises(self):
        with pytest.raises(ImportError, match="CAMB is required for Test 4"):
            generate_camb_spectra_r_tau(r=0.01, tau=0.05, lmax=LMAX)


class TestSpectrumPadding:
    """If CAMB ever returns fewer than lmax+1 rows, the generators must
    zero-pad to the documented output length (defensive branch)."""

    @pytest.fixture(autouse=True)
    def _short_camb(self, monkeypatch):
        class _ShortResults:
            def get_total_cls(self, lmax, CMB_unit=None, raw_cl=False):
                n = lmax - 5  # five rows short
                rng = np.random.default_rng(0)
                return rng.uniform(1e-8, 1e-6, size=(n, 4))

        monkeypatch.setattr(camb, "get_results", lambda pars: _ShortResults())

    def test_test3_pads_to_lmax_plus_one(self):
        cl_ee, cl_bb = generate_camb_spectra(tau=0.05, lmax=LMAX)
        assert cl_ee.shape == (LMAX + 1,)
        assert cl_bb.shape == (LMAX + 1,)
        np.testing.assert_array_equal(cl_ee[-5:], 0.0)
        np.testing.assert_array_equal(cl_bb[-5:], 0.0)

    def test_test4_pads_to_lmax_plus_one(self):
        cl_ee, cl_bb = generate_camb_spectra_r_tau(r=0.01, tau=0.05, lmax=LMAX)
        assert cl_ee.shape == (LMAX + 1,)
        assert cl_bb.shape == (LMAX + 1,)
        np.testing.assert_array_equal(cl_ee[-5:], 0.0)
        np.testing.assert_array_equal(cl_bb[-5:], 0.0)


class TestPrecomputeRTauGrid:
    """precompute_camb_spectra_r_tau: (r, τ) sampling + per-pair CAMB loop."""

    def test_shapes_ranges_reproducibility(self):
        r, tau, ee, bb = precompute_camb_spectra_r_tau(
            n_spectra=2, lmax=LMAX, seed=42)
        assert r.shape == (2,) and tau.shape == (2,)
        assert ee.shape == (2, LMAX + 1)
        assert bb.shape == (2, LMAX + 1)
        # r is sampled in log space over (0, R_MAX]; τ uniform in range
        assert np.all((r >= 0) & (r <= R_MAX))
        assert np.all((tau >= TAU4_MIN) & (tau <= TAU4_MAX))
        assert np.all(np.isfinite(ee)) and np.all(bb >= 0)

        # Same seed → bit-identical grid
        r2, tau2, ee2, bb2 = precompute_camb_spectra_r_tau(
            n_spectra=2, lmax=LMAX, seed=42)
        np.testing.assert_array_equal(r, r2)
        np.testing.assert_array_equal(tau, tau2)
        np.testing.assert_array_equal(ee, ee2)
        np.testing.assert_array_equal(bb, bb2)

        # Different seed → different parameter draw
        r3, _, _, _ = precompute_camb_spectra_r_tau(
            n_spectra=2, lmax=LMAX, seed=7)
        assert not np.allclose(r, r3), "different seeds must differ"
