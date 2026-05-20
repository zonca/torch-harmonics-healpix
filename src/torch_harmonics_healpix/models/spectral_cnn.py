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

        # Spectral weights: [out_channels, in_channels, lmax, mmax]
        # torch-harmonics RealSHT output shape is (lmax, mmax), NOT (lmax+1, mmax+1)
        # Coefficients are complex, so weights must be complex too
        lmax = forward_transform.lmax
        mmax = forward_transform.mmax
        self.weight_real = nn.Parameter(
            torch.randn(out_channels, in_channels, lmax, mmax)
            * (1.0 / (in_channels * lmax))
        )
        self.weight_imag = nn.Parameter(
            torch.randn(out_channels, in_channels, lmax, mmax)
            * (1.0 / (in_channels * lmax))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: [batch, in_channels, nlat, nlon]

        Returns:
            [batch, out_channels, nlat, nlon]
        """
        # SHT forward: [batch, in_channels, nlat, nlon] -> complex [batch, in_channels, lmax, mmax]
        coeff = self.forward_transform(x)

        # Build complex weight from real and imaginary parts
        weight = torch.complex(self.weight_real, self.weight_imag)

        # Spectral convolution: multiply by learned complex weights
        # coeff: [B, C_in, L, M] (complex)
        # weight: [C_out, C_in, L, M] (complex)
        # output: [B, C_out, L, M] (complex)
        coeff_out = torch.einsum("bilm,oilm->bolm", coeff, weight)

        # ISHT inverse: complex [batch, out_channels, L, M] -> [batch, out_channels, nlat, nlon]
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
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.nside = nside
        self.in_channels = in_channels
        self.out_channels = out_channels

        if nlat is None:
            nlat = 2 * nside
        if nlon is None:
            nlon = 2 * nlat
        if lmax is None:
            lmax = 3 * nside - 1

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax

        # RealSHT uses mmax as the *size* of the m dimension, default = nlon//2+1
        # and lmax as the *size* of the l dimension, default = nlat
        mmax = min(lmax, nlon // 2 + 1)

        # HEALPix → equiangular conversion
        self.to_equi = HealpixToEquiangular(nside, nlat, nlon)

        # SHT transforms
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")

        # Spectral convolution blocks
        self.blocks = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        in_ch = in_channels
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
            nn.Linear(64, out_channels),
        )

    def forward(self, healpix_map: torch.Tensor) -> torch.Tensor:
        """Estimate parameter(s) from HEALPix map(s).

        Args:
            healpix_map: [batch, npix] or [batch, in_channels, npix]
                         HEALPix map(s) in ring ordering.

        Returns:
            For out_channels=1: [batch] predicted values.
            For out_channels>1: [batch, out_channels] predicted values.
        """
        # Handle input shape: ensure [batch, channels, npix]
        if healpix_map.dim() == 2:
            # Single-channel input: [batch, npix] → [batch, 1, npix]
            healpix_map = healpix_map.unsqueeze(1)

        # HEALPix → equiangular for each channel
        # to_equi expects [batch, npix] → [batch, nlat, nlon]
        # Apply per-channel then stack
        if self.in_channels == 1:
            x = self.to_equi(healpix_map.squeeze(1))  # [batch, nlat, nlon]
            x = x.unsqueeze(1)  # [batch, 1, nlat, nlon]
        else:
            # Multi-channel: resample each channel separately
            channels = []
            for c in range(self.in_channels):
                ch_eq = self.to_equi(healpix_map[:, c, :])  # [batch, nlat, nlon]
                channels.append(ch_eq)
            x = torch.stack(channels, dim=1)  # [batch, in_channels, nlat, nlon]

        # Spectral convolution blocks with ReLU + BatchNorm
        for i, (block, bn) in enumerate(zip(self.blocks, self.batchnorms)):
            x = block(x)
            x = bn(x)
            if i < self.num_blocks - 1:
                x = self.relu(x)

        # Global average pooling over spatial dims: [batch, channels, nlat, nlon]
        # → [batch, channels]
        x = x.mean(dim=(-2, -1))

        # FC head → [batch, out_channels]
        x = self.fc(x)

        # For single-output, squeeze last dim
        if self.out_channels == 1:
            x = x.squeeze(-1)

        return x
