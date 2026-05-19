"""Spectral CNN model for ℓ_p estimation from HEALPix temperature maps.

Uses spectral convolutions via SHT (RealSHT + InverseRealSHT) from
torch-harmonics with learned spectral weights, combined with HEALPix↔equiangular
resampling.

Architecture: HEALPix → HealpixToEquiangular → SpectralConv blocks → FC head → ℓ_p

Note: torch-harmonics 0.8.0 does not have SpectralConvS2 in the main package.
We implement spectral convolution manually using RealSHT + learned weights + InverseRealSHT.
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT, InverseRealSHT

from ..healpix_resample import HealpixToEquiangular


class SpectralConvBlock(nn.Module):
    """Spectral convolution block: SHT → learned weights → ISHT.

    Implements spectral convolution as:
    1. Forward SHT: spatial → harmonic coefficients (a_lm)
    2. Multiply by learned weights per (l, m) channel
    3. Inverse SHT: harmonic → spatial

    This is rotation-equivariant by construction.

    Args:
        forward_transform: RealSHT instance.
        inverse_transform: InverseRealSHT instance.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        forward_transform: RealSHT,
        inverse_transform: InverseRealSHT,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        # Determine actual SHT output shape by running a dummy forward pass
        # RealSHT clips mmax to nlon//2 internally, so we must match
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels,
                                forward_transform.nlat,
                                forward_transform.nlon)
            coeff_shape = forward_transform(dummy).shape  # [1, C, lmax+1, mmax_eff+1]
            self._lmax_out = coeff_shape[-2]
            self._mmax_out = coeff_shape[-1]

        # Spectral weights: [out_channels, in_channels, lmax_eff+1, mmax_eff+1]
        # These are the learned convolution kernels in harmonic space
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self._lmax_out, self._mmax_out)
            * (1.0 / (in_channels * self._lmax_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: [batch, in_channels, nlat, nlon]

        Returns:
            [batch, out_channels, nlat, nlon]
        """
        # SHT forward: [batch, in_channels, nlat, nlon] -> [batch, in_channels, lmax+1, mmax_eff+1]
        coeff = self.forward_transform(x)

        # Spectral convolution: multiply by learned weights
        # coeff: [B, C_in, L, M]
        # weight: [C_out, C_in, L, M]
        # output: [B, C_out, L, M]
        coeff_out = torch.einsum("bilm,oilm->bolm", coeff, self.weight)

        # ISHT inverse: [batch, out_channels, L, M] -> [batch, out_channels, nlat, nlon]
        out = self.inverse_transform(coeff_out)
        return out


class SpectralCNN(nn.Module):
    """Spectral convolution CNN for scalar HEALPix map parameter estimation.

    Architecture:
        1. HEALPix → equiangular resampling
        2. Spectral convolution blocks with ReLU and batch norm
        3. Global average pooling + FC head for regression

    Args:
        nside: HEALPix Nside (default 16 for Test 1).
        nlat: Equiangular grid latitude bands. Default: 2*nside.
        nlon: Equiangular grid longitude points. Default: 2*nlat.
        lmax: Maximum multipole for SHT. Default: 3*nside-1.
        hidden_channels: Number of channels in hidden layers.
        num_blocks: Number of spectral convolution blocks.
    """

    def __init__(
        self,
        nside: int = 16,
        nlat: int = None,
        nlon: int = None,
        lmax: int = None,
        hidden_channels: int = 32,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.nside = nside

        if nlat is None:
            nlat = 2 * nside
        if nlon is None:
            nlon = 2 * nlat
        if lmax is None:
            lmax = 3 * nside - 1

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax

        # RealSHT internally clips mmax = min(mmax, nlon//2)
        # We pass mmax=lmax and let it clip — SpectralConvBlock detects
        # the actual output shape via a dummy forward pass.
        mmax = lmax

        # HEALPix → equiangular conversion
        self.to_equi = HealpixToEquiangular(nside, nlat, nlon)

        # SHT transforms
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")

        # Spectral convolution blocks
        self.blocks = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        in_ch = 1
        for i in range(num_blocks):
            out_ch = hidden_channels
            self.blocks.append(
                SpectralConvBlock(self.sht, self.isht, in_ch, out_ch)
            )
            self.batchnorms.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        self.num_blocks = num_blocks
        self.relu = nn.ReLU()

        # FC regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, healpix_map: torch.Tensor) -> torch.Tensor:
        """Estimate ℓ_p from HEALPix temperature map(s).

        Args:
            healpix_map: [batch, npix] HEALPix map(s) in ring ordering.

        Returns:
            ell_p_pred: [batch] predicted ℓ_p values.
        """
        # HEALPix → equiangular [batch, nlat, nlon]
        x = self.to_equi(healpix_map)

        # Add channel dimension: [batch, 1, nlat, nlon]
        x = x.unsqueeze(1)

        # Spectral convolution blocks with ReLU + BatchNorm
        for i, (block, bn) in enumerate(zip(self.blocks, self.batchnorms)):
            x = block(x)
            x = bn(x)
            if i < self.num_blocks - 1:
                x = self.relu(x)

        # Global average pooling over spatial dims: [batch, channels, nlat, nlon]
        # → [batch, channels]
        x = x.mean(dim=(-2, -1))

        # FC head → [batch, 1] → [batch]
        x = self.fc(x).squeeze(-1)
        return x
