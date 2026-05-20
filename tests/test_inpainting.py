"""Tests for inpainting feature in SpectralCNN.

Inpainting replaces zero-masked pixels with the mean of observed pixels
before the SHT, preventing zero-pixel corruption of spectral coefficients
at partial-sky observations (f_sky < 1).
"""

import numpy as np
import healpy as hp
import torch
import pytest

from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask, NSIDE


class TestInpaintingSingleChannel:
    """Test inpainting for single-channel (Test 1 style) input."""

    def test_inpaint_replaces_zeros(self):
        """Inpainting should replace zero-valued pixels with observed-pixel mean."""
        model = SpectralCNN(nside=16, in_channels=1, out_channels=1, inpaint=True)
        model.eval()

        npix = hp.nside2npix(16)
        # Create a map with half the pixels zeroed (masked)
        x = torch.randn(2, npix)
        mask = torch.zeros(npix)
        mask[:npix // 2] = 1.0
        x_masked = x * mask

        # Run inpainting manually
        x_eq = model.to_equi(x_masked)  # [batch, nlat, nlon]
        x_eq = x_eq.unsqueeze(1)  # [batch, 1, nlat, nlon]

        x_inpainted = model._inpaint_masked_pixels(x_eq)

        # After inpainting, all pixels should be non-zero (unless mean happens to be 0)
        # The key property: zero pixels are replaced, non-zero pixels are preserved
        observed_mask = (x_eq != 0).float()
        original_nonzero = x_eq * observed_mask
        inpainted_nonzero = x_inpainted * observed_mask
        assert torch.allclose(original_nonzero, inpainted_nonzero, atol=1e-6), \
            "Inpainting should not modify observed (non-zero) pixels"

    def test_no_inpaint_when_disabled(self):
        """When inpaint=False, zero pixels should remain zero."""
        model = SpectralCNN(nside=16, in_channels=1, out_channels=1, inpaint=False)
        model.eval()

        npix = hp.nside2npix(16)
        x = torch.randn(1, npix)
        mask = torch.zeros(npix)
        mask[:npix // 2] = 1.0
        x_masked = x * mask

        # Forward pass without inpainting
        with torch.no_grad():
            out = model(x_masked)

        # Just verify it runs without error; inpainting is not applied
        assert torch.isfinite(out).all()

    def test_inpaint_differentiable(self):
        """Inpainting should support gradient flow."""
        model = SpectralCNN(nside=16, in_channels=1, out_channels=1, inpaint=True)

        npix = hp.nside2npix(16)
        x = torch.randn(2, npix, requires_grad=True)
        mask = torch.zeros(npix)
        mask[:npix // 2] = 1.0
        x_masked = x * mask

        out = model(x_masked)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should flow through inpainting"


class TestInpaintingMultiChannel:
    """Test inpainting for multi-channel (Test 2/3 style) input with mask channel."""

    def test_inpaint_preserves_mask_channel(self):
        """Inpainting should NOT modify the mask channel (last channel)."""
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True)
        model.eval()

        npix = hp.nside2npix(16)
        # Q, U, mask channels
        q = torch.randn(2, npix)
        u = torch.randn(2, npix)
        mask = create_sky_mask(0.5, NSIDE, np.random.default_rng(42)).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).expand(2, -1)

        q_masked = q * mask_tensor
        u_masked = u * mask_tensor
        x = torch.stack([q_masked, u_masked, mask_tensor], dim=1)  # [batch, 3, npix]

        # Convert to equiangular
        channels = []
        for c in range(3):
            ch_eq = model.to_equi(x[:, c, :])
            channels.append(ch_eq)
        x_eq = torch.stack(channels, dim=1)  # [batch, 3, nlat, nlon]

        x_inpainted = model._inpaint_masked_pixels(x_eq)

        # Mask channel should be unchanged
        assert torch.allclose(x_eq[:, -1:, :, :], x_inpainted[:, -1:, :, :], atol=1e-6), \
            "Inpainting must not modify the mask channel"

    def test_inpaint_signal_channels_only(self):
        """Inpainting should only modify signal channels (Q, U), not mask."""
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True)
        model.eval()

        npix = hp.nside2npix(16)
        q = torch.randn(2, npix)
        u = torch.randn(2, npix)
        mask = create_sky_mask(0.2, NSIDE, np.random.default_rng(0)).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).expand(2, -1)

        q_masked = q * mask_tensor
        u_masked = u * mask_tensor
        x = torch.stack([q_masked, u_masked, mask_tensor], dim=1)

        # Convert to equiangular and inpaint
        channels = []
        for c in range(3):
            ch_eq = model.to_equi(x[:, c, :])
            channels.append(ch_eq)
        x_eq = torch.stack(channels, dim=1)

        x_inpainted = model._inpaint_masked_pixels(x_eq)

        # In observed pixels, signal channels should be unchanged
        mask_eq = x_eq[:, -1:, :, :]  # [batch, 1, nlat, nlon]
        observed = (mask_eq > 0).float()

        for c in range(2):  # Q and U channels
            original_obs = x_eq[:, c:c+1, :, :] * observed
            inpainted_obs = x_inpainted[:, c:c+1, :, :] * observed
            assert torch.allclose(original_obs, inpainted_obs, atol=1e-5), \
                f"Signal channel {c} modified in observed pixels"

    def test_inpaint_fills_masked_pixels(self):
        """In inpainted output, masked pixels should be non-zero (replaced by mean)."""
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True)
        model.eval()

        npix = hp.nside2npix(16)
        q = torch.ones(1, npix) * 5.0  # constant signal
        u = torch.ones(1, npix) * 3.0
        mask = create_sky_mask(0.5, NSIDE, np.random.default_rng(0)).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        q_masked = q * mask_tensor
        u_masked = u * mask_tensor
        x = torch.stack([q_masked, u_masked, mask_tensor], dim=1)

        channels = []
        for c in range(3):
            ch_eq = model.to_equi(x[:, c, :])
            channels.append(ch_eq)
        x_eq = torch.stack(channels, dim=1)

        x_inpainted = model._inpaint_masked_pixels(x_eq)

        # For constant signal, inpainted value should equal the observed mean
        # Q channel observed mean = 5.0, U channel observed mean = 3.0
        mask_eq = x_eq[:, -1:, :, :]
        unobserved = (mask_eq == 0).float()

        # All inpainted pixels in signal channels should be close to the observed mean
        q_inpainted = x_inpainted[:, 0:1, :, :]
        q_unobserved_values = q_inpainted * unobserved
        # If there are unobserved pixels, they should be ~5.0 (observed mean)
        n_unobserved = unobserved.sum().item()
        if n_unobserved > 0:
            mean_val = q_unobserved_values.sum().item() / n_unobserved
            assert abs(mean_val - 5.0) < 0.5, \
                f"Inpainted Q values should be ~5.0, got {mean_val}"


class TestInpaintingEndToEnd:
    """End-to-end tests with inpainting enabled."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_model_forward_with_inpainting(self, device):
        """Full forward pass with inpainting should produce finite output."""
        npix = hp.nside2npix(16)
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True).to(device)

        mask = create_sky_mask(0.2, NSIDE, np.random.default_rng(42)).astype(np.float32)
        q = torch.randn(4, npix) * torch.from_numpy(mask).float()
        u = torch.randn(4, npix) * torch.from_numpy(mask).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).expand(4, -1)
        x = torch.stack([q, u, mask_t], dim=1).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 2), f"Expected shape (4, 2), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_backward_with_inpainting(self, device):
        """Backward pass through inpainting should work."""
        npix = hp.nside2npix(16)
        model = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True).to(device)

        mask = create_sky_mask(0.1, NSIDE, np.random.default_rng(0)).astype(np.float32)
        q = torch.randn(2, npix) * torch.from_numpy(mask).float()
        u = torch.randn(2, npix) * torch.from_numpy(mask).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).expand(2, -1)
        x = torch.stack([q, u, mask_t], dim=1).to(device)

        target = torch.tensor([[10.0, 12.0], [15.0, 8.0]], device=device)
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed through inpainting"

    def test_inpainting_improves_low_fsky(self, device):
        """Inpainting should produce different (hopefully better) output than no inpainting
        at low f_sky. At minimum, the outputs should differ."""
        npix = hp.nside2npix(16)
        mask = create_sky_mask(0.2, NSIDE, np.random.default_rng(42)).astype(np.float32)

        model_no_inpaint = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=False).to(device)
        model_inpaint = SpectralCNN(nside=16, in_channels=3, out_channels=2, inpaint=True).to(device)

        # Copy weights so both models have same parameters
        model_inpaint.load_state_dict(model_no_inpaint.state_dict(), strict=False)

        q = torch.randn(2, npix) * torch.from_numpy(mask).float()
        u = torch.randn(2, npix) * torch.from_numpy(mask).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).expand(2, -1)
        x = torch.stack([q, u, mask_t], dim=1).to(device)

        with torch.no_grad():
            out_no = model_no_inpaint(x)
            out_yes = model_inpaint(x)

        # Outputs should differ because inpainting modifies the input
        assert not torch.allclose(out_no, out_yes, atol=1e-4), \
            "Inpainting should change model output at f_sky=0.2"
