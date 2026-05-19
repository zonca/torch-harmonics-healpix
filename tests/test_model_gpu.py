"""Tests for SpectralCNN model on GPU (requires torch-harmonics + CUDA)."""

import numpy as np
import healpy as hp
import torch
import pytest

from torch_harmonics_healpix.healpix_resample import (
    HealpixToEquiangular,
    EquiangularToHealpix,
)
from torch_harmonics_healpix.data_generation import generate_map, NSIDE, LMAX, SIGMA_P
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN
from torch_harmonics_healpix.mcmc_baseline import mcmc_estimate_ell_p


class TestSpectralCNNGPU:
    """Test SpectralCNN model on GPU (requires CUDA)."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_model_forward_shape(self, device):
        """Model output should be [batch] for [batch, npix] input."""
        nside = 16
        npix = hp.nside2npix(nside)
        model = SpectralCNN(nside=nside).to(device)

        x = torch.randn(4, npix, device=device)
        out = model(x)
        assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"

    def test_model_forward_single(self, device):
        """Single sample forward pass."""
        nside = 16
        npix = hp.nside2npix(nside)
        model = SpectralCNN(nside=nside).to(device)

        x = torch.randn(1, npix, device=device)
        out = model(x)
        assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"
        assert out.dtype == torch.float32

    def test_model_backward(self, device):
        """Backward pass should work (differentiability test)."""
        nside = 16
        npix = hp.nside2npix(nside)
        model = SpectralCNN(nside=nside).to(device)

        x = torch.randn(2, npix, device=device)
        target = torch.tensor([10.0, 15.0], device=device)
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        # Check that gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"

    def test_resampling_on_gpu(self, device):
        """HEALPix ↔ equiangular resampling should work on GPU."""
        nside = 16
        npix = hp.nside2npix(nside)
        nlat, nlon = 32, 64

        to_equi = HealpixToEquiangular(nside, nlat, nlon).to(device)
        to_hp = EquiangularToHealpix(nside, nlat, nlon).to(device)

        x = torch.randn(2, npix, device=device)
        equi = to_equi(x)
        assert equi.shape == (2, nlat, nlon)
        assert equi.device.type == "cuda"

        back = to_hp(equi)
        assert back.shape == (2, npix)
        assert back.device.type == "cuda"

    def test_mcmc_baseline_on_generated_map(self):
        """MCMC baseline should give reasonable estimate on CPU."""
        ell_p_true = 12.0
        m = generate_map(ell_p_true, nside=NSIDE, noise_std=0.0,
                         rng=np.random.default_rng(42))
        ell_p_est = mcmc_estimate_ell_p(m)
        # Allow 20% relative error on a single realization
        rel_error = abs(ell_p_est - ell_p_true) / ell_p_true * 100
        assert rel_error < 20.0, f"MCMC error {rel_error:.1f}% too high"

    def test_end_to_end_pipeline(self, device):
        """Full pipeline: generate map → resample → SHT → verify shape."""
        nside = 16
        npix = hp.nside2npix(nside)

        # Generate a map
        m = generate_map(ell_p=10.0, nside=nside, noise_std=0.0,
                         rng=np.random.default_rng(0))
        x = torch.from_numpy(m).float().unsqueeze(0).to(device)

        # Resample to equiangular
        to_equi = HealpixToEquiangular(nside).to(device)
        equi = to_equi(x)
        assert equi.shape == (1, 32, 64)

        # Run through model
        model = SpectralCNN(nside=nside).to(device)
        out = model(x)
        assert out.shape == (1,)
        # Untrained model output is arbitrary but should be finite
        assert torch.isfinite(out).all()
