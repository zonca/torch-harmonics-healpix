"""CPU tests for SpectralCNN and MultiResSpectralCNN.

Mirrors tests/test_model_gpu.py (which skips without CUDA) so the full
model surface — forward, backward, save/load, inpainting end-to-end —
is exercised on every CPU-only CI runner.

Parameter-count regression anchors are pinned to the shipped v4b
checkpoints:
  - SpectralCNN(nside=16, hc=32, nb=4, in=3, out=2)   6,902,402  (Test 2/3)
  - MultiResSpectralCNN(nside=16, hc=32, nb=4, 1→1)  1,525,665  (Test 1)
"""

import numpy as np
import healpy as hp
import torch
import pytest

from torch_harmonics_healpix.models import SpectralCNN, MultiResSpectralCNN
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask, NSIDE


def _masked_qu_input(batch, f_sky, seed):
    """Build a [batch, 3, npix] (Q, U, mask) tensor with a partial-sky mask."""
    npix = hp.nside2npix(16)
    mask = create_sky_mask(f_sky, NSIDE, np.random.default_rng(seed)).astype(np.float32)
    mask_t = torch.from_numpy(mask)
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, npix, generator=g) * mask_t
    u = torch.randn(batch, npix, generator=g) * mask_t
    return torch.stack([q, u, mask_t.unsqueeze(0).expand(batch, -1)], dim=1)


class TestSpectralCNNcpu:
    """SpectralCNN forward/backward/serialization on CPU."""

    def test_forward_shape_single_channel(self):
        nside = 16
        model = SpectralCNN(nside=nside).eval()
        x = torch.randn(4, hp.nside2npix(nside))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,)
        assert out.dtype == torch.float32

    def test_forward_single_sample(self):
        nside = 16
        model = SpectralCNN(nside=nside).eval()
        x = torch.randn(1, hp.nside2npix(nside))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_forward_multichannel_two_outputs(self):
        nside = 16
        model = SpectralCNN(nside=nside, in_channels=3, out_channels=2).eval()
        x = torch.randn(2, 3, hp.nside2npix(nside))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()

    def test_backward(self):
        nside = 16
        model = SpectralCNN(nside=nside)
        x = torch.randn(2, hp.nside2npix(nside))
        target = torch.tensor([10.0, 15.0])
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients"

    def test_eval_mode_deterministic(self):
        """Two eval-mode forwards must agree (dropout disabled, BN running stats)."""
        nside = 16
        model = SpectralCNN(nside=nside, in_channels=3, out_channels=2).eval()
        x = torch.randn(2, 3, hp.nside2npix(nside))
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_state_dict_roundtrip(self):
        """Reload into a fresh instance must reproduce outputs exactly."""
        nside = 16
        model = SpectralCNN(nside=nside, in_channels=3, out_channels=2).eval()
        x = torch.randn(2, 3, hp.nside2npix(nside))
        with torch.no_grad():
            out_orig = model(x)

        clone = SpectralCNN(nside=nside, in_channels=3, out_channels=2).eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            out_clone = clone(x)
        assert torch.allclose(out_orig, out_clone, atol=1e-6)

    def test_default_grid_sizes(self):
        model = SpectralCNN(nside=16)
        assert model.nlat == 32          # 2 * nside
        assert model.nlon == 64          # 2 * nlat
        assert model.lmax == 3 * 16 - 1  # 47
        # torch-harmonics may clamp the stored transform dims (e.g. 0.9.x
        # caps lmax at mmax); require the transform pair to stay consistent
        # and the learned weights to match the transform dimensions.
        assert model.sht.mmax == 33      # min(lmax, nlon//2 + 1)
        assert model.isht.lmax == model.sht.lmax
        assert model.isht.mmax == model.sht.mmax
        w = model.blocks[0].weight_real
        assert w.shape[2:] == (model.sht.lmax, model.sht.mmax)

    def test_param_count_regression(self):
        """Anchor to the shipped v4b Test 2/3 checkpoint size (6.9M)."""
        model = SpectralCNN(nside=16, hidden_channels=32, num_blocks=4,
                            in_channels=3, out_channels=2)
        n = sum(p.numel() for p in model.parameters())
        assert n == 6_902_402, f"Parameter count drifted: {n} != 6,902,402"

    def test_small_nside(self):
        """Architecture must also build and run at a smaller resolution."""
        nside = 8
        model = SpectralCNN(nside=nside).eval()
        x = torch.randn(2, hp.nside2npix(nside))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()


class TestMultiResSpectralCNNcpu:
    """MultiResSpectralCNN forward/backward/serialization on CPU."""

    def test_forward_shape(self):
        nside = 16
        model = MultiResSpectralCNN(nside=nside).eval()
        x = torch.randn(4, hp.nside2npix(nside))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,)
        assert torch.isfinite(out).all()

    def test_forward_multichannel(self):
        nside = 16
        model = MultiResSpectralCNN(nside=nside, in_channels=3, out_channels=2).eval()
        x = torch.randn(2, 3, hp.nside2npix(nside))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()

    def test_backward(self):
        nside = 16
        model = MultiResSpectralCNN(nside=nside)
        x = torch.randn(2, hp.nside2npix(nside))
        target = torch.tensor([10.0, 15.0])
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        assert all(torch.isfinite(g).all() for g in grads)

    def test_multires_block_structure(self):
        """Default 4 blocks: ℓ_max halves each time (47 → 23 → 11 → 5)."""
        model = MultiResSpectralCNN(nside=16, num_blocks=4)
        expected_lmax = [47, 23, 11, 5]
        for i, block in enumerate(model.blocks):
            assert block.forward_transform.lmax == expected_lmax[i], (
                f"block {i}: lmax {block.forward_transform.lmax} != {expected_lmax[i]}"
            )

    def test_lmax_ratios_validation(self):
        """Too few lmax_ratios for num_blocks must fail loudly (assert)."""
        with pytest.raises(AssertionError):
            MultiResSpectralCNN(nside=16, num_blocks=4, lmax_ratios=[1.0, 0.5])

    def test_state_dict_roundtrip(self):
        nside = 16
        model = MultiResSpectralCNN(nside=nside, hidden_channels=32,
                                    num_blocks=4).eval()
        x = torch.randn(2, hp.nside2npix(nside))
        with torch.no_grad():
            out_orig = model(x)

        clone = MultiResSpectralCNN(nside=nside, hidden_channels=32,
                                    num_blocks=4).eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            out_clone = clone(x)
        assert torch.allclose(out_orig, out_clone, atol=1e-6)

    def test_param_count_regression(self):
        """Anchor to the shipped v4b Test 1 checkpoint size (1.53M)."""
        model = MultiResSpectralCNN(nside=16, hidden_channels=32, num_blocks=4,
                                    in_channels=1, out_channels=1)
        n = sum(p.numel() for p in model.parameters())
        assert n == 1_525_665, f"Parameter count drifted: {n} != 1,525,665"


class TestInpaintingEndToEndCPU:
    """CPU versions of the CUDA-gated end-to-end inpainting tests."""

    def test_forward_with_inpainting(self):
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2,
                            inpaint=True).eval()
        x = _masked_qu_input(4, f_sky=0.2, seed=42)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_backward_with_inpainting(self):
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True)
        x = _masked_qu_input(2, f_sky=0.1, seed=0)
        target = torch.tensor([[10.0, 12.0], [15.0, 8.0]])
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_forward_invokes_inpainting(self, monkeypatch):
        """With inpaint=True the forward pass must route through
        _inpaint_masked_pixels; with inpaint=False it must not.

        (An output-delta assertion is not robust here: with untrained
        weights the ReLU stack collapses the inpainting delta to ~1e-8,
        so 'outputs differ' only holds for trained models.)
        """
        x = _masked_qu_input(2, f_sky=0.2, seed=42)

        calls = {"n": 0}
        for enabled in (False, True):
            model = SpectralCNN(nside=16, in_channels=3, out_channels=2,
                                inpaint=enabled).eval()
            original = model._inpaint_masked_pixels

            def spy(t, _original=original, _calls=calls):
                _calls["n"] += 1
                return _original(t)

            monkeypatch.setattr(model, "_inpaint_masked_pixels", spy)
            before = calls["n"]
            with torch.no_grad():
                model(x)
            used = calls["n"] - before
            assert used == (1 if enabled else 0), (
                f"inpaint={enabled}: expected {'1' if enabled else '0'} "
                f"_inpaint_masked_pixels call(s), got {used}"
            )

    def test_full_sky_inpainting_noop(self):
        """At f_sky=1 every pixel is observed: inpainting must be a no-op."""
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2,
                            inpaint=True).eval()
        mask = torch.ones(hp.nside2npix(16))
        g = torch.Generator().manual_seed(3)
        q = torch.randn(2, hp.nside2npix(16), generator=g)
        u = torch.randn(2, hp.nside2npix(16), generator=g)
        x = torch.stack([q, u, mask.unsqueeze(0).expand(2, -1)], dim=1)

        x_eq = torch.stack([model.to_equi(x[:, c, :]) for c in range(3)], dim=1)
        x_inp = model._inpaint_masked_pixels(x_eq)
        assert torch.allclose(x_eq, x_inp, atol=1e-6), \
            "Inpainting modified an all-observed map"
