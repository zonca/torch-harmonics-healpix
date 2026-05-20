"""Multi-resolution Spectral CNN for HEALPix map parameter estimation.

Extends SpectralCNN with progressively decreasing spectral resolution,
mimicking NNhealpix's multi-resolution pooling (Nside 16→8→4→2→1).

Architecture:
  Block 1: Full ℓ_max (e.g., 47) — captures small-scale features
  Block 2: ℓ_max/2 (e.g., 23)  — captures medium-scale features
  Block 3: ℓ_max/4 (e.g., 11)  — captures large-scale features
  Block 4: ℓ_max/8 (e.g., 5)   — captures very large-scale features

Each block uses its own SHT/ISHT pair at the appropriate resolution.
Spatial grids are downsampled accordingly (fewer lat/lon points).
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT, InverseRealSHT

from ..healpix_resample import HealpixToEquiangular


class SpectralConvBlock(nn.Module):
    """Spectral convolution block: SHT → learned weights → ISHT."""

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
        coeff = self.forward_transform(x)
        weight = torch.complex(self.weight_real, self.weight_imag)
        coeff_out = torch.einsum("bilm,oilm->bolm", coeff, weight)
        out = self.inverse_transform(coeff_out)
        return out


class MultiResSpectralCNN(nn.Module):
    """Multi-resolution Spectral CNN with progressively decreasing ℓ_max.

    Unlike SpectralCNN (fixed ℓ_max for all blocks), this model uses
    separate SHT/ISHT pairs at decreasing resolutions, mimicking
    NNhealpix's multi-resolution pooling approach.

    Args:
        nside: HEALPix Nside (default 16).
        hidden_channels: Number of channels in hidden layers.
        num_blocks: Number of spectral convolution blocks (default 4).
        in_channels: Number of input channels.
        out_channels: Number of output channels (regression targets).
        lmax_base: Base ℓ_max for first block (default: 3*nside-1).
        lmax_ratios: Multiplicative factors for ℓ_max per block.
                     Default: [1.0, 0.5, 0.25, 0.125] (4 blocks).
    """

    def __init__(
        self,
        nside: int = 16,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        lmax_base: int = None,
        lmax_ratios: list = None,
    ):
        super().__init__()
        self.nside = nside
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        if lmax_base is None:
            lmax_base = 3 * nside - 1

        if lmax_ratios is None:
            # Default: progressively halve ℓ_max
            lmax_ratios = [1.0 / (2 ** i) for i in range(num_blocks)]

        assert len(lmax_ratios) >= num_blocks, \
            f"Need at least {num_blocks} lmax_ratios, got {len(lmax_ratios)}"

        # HEALPix → equiangular conversion (for initial input)
        nlat_base = 2 * nside
        nlon_base = 2 * nlat_base
        self.to_equi = HealpixToEquiangular(nside, nlat_base, nlon_base)

        # Build blocks at decreasing resolutions
        self.blocks = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.downsamplers = nn.ModuleList()  # Spatial interpolation between blocks

        in_ch = in_channels
        prev_nlat = nlat_base
        prev_nlon = nlon_base

        for i in range(num_blocks):
            # Compute resolution for this block
            block_lmax = max(int(lmax_base * lmax_ratios[i]), 4)
            block_nlat = max(block_lmax + 1, 4)
            block_nlon = 2 * block_nlat
            block_mmax = min(block_lmax, block_nlon // 2 + 1)

            # Create SHT/ISHT for this resolution
            sht = RealSHT(block_nlat, block_nlon, lmax=block_lmax, mmax=block_mmax, grid="equiangular")
            isht = InverseRealSHT(block_nlat, block_nlon, lmax=block_lmax, mmax=block_mmax, grid="equiangular")

            out_ch = hidden_channels
            self.blocks.append(SpectralConvBlock(sht, isht, in_ch, out_ch))
            self.batchnorms.append(nn.BatchNorm2d(out_ch))

            # Downsampler: interpolate from previous spatial resolution to this block's
            if i > 0:
                self.downsamplers.append(
                    nn.Upsample(size=(block_nlat, block_nlon), mode='bilinear', align_corners=False)
                )
            else:
                # First block: need to downsample from base to block_0 resolution
                if block_nlat != nlat_base or block_nlon != nlon_base:
                    self.downsamplers.append(
                        nn.Upsample(size=(block_nlat, block_nlon), mode='bilinear', align_corners=False)
                    )
                else:
                    self.downsamplers.append(None)

            in_ch = out_ch
            prev_nlat = block_nlat
            prev_nlon = block_nlon

        self.relu = nn.ReLU()

        # FC regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, out_channels),
        )

    def forward(self, healpix_map: torch.Tensor) -> torch.Tensor:
        """Estimate parameter(s) from HEALPix map(s).

        Args:
            healpix_map: [batch, npix] or [batch, in_channels, npix]

        Returns:
            For out_channels=1: [batch] predicted values.
            For out_channels>1: [batch, out_channels] predicted values.
        """
        # Handle input shape
        if healpix_map.dim() == 2:
            healpix_map = healpix_map.unsqueeze(1)

        # HEALPix → equiangular for each channel
        if self.in_channels == 1:
            x = self.to_equi(healpix_map.squeeze(1))
            x = x.unsqueeze(1)
        else:
            channels = []
            for c in range(self.in_channels):
                ch_eq = self.to_equi(healpix_map[:, c, :])
                channels.append(ch_eq)
            x = torch.stack(channels, dim=1)

        # Spectral convolution blocks with decreasing resolution
        for i, (block, bn) in enumerate(zip(self.blocks, self.batchnorms)):
            # Downsample to this block's spatial resolution
            downsampler = self.downsamplers[i]
            if downsampler is not None:
                x = downsampler(x)

            x = block(x)
            x = bn(x)
            if i < self.num_blocks - 1:
                x = self.relu(x)

        # Global average pooling
        x = x.mean(dim=(-2, -1))

        # FC head
        x = self.fc(x)

        if self.out_channels == 1:
            x = x.squeeze(-1)

        return x
