"""Spectral CNN model for ℓ_p estimation from HEALPix temperature maps.

Uses SpectralConvS2 from torch-harmonics for rotation-equivariant
spectral convolutions on equiangular grids, with HEALPix↔equiangular
resampling at input/output.

Architecture: HEALPix → HealpixToEquiangular → SpectralConvS2 blocks → FC head → ℓ_p
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT, InverseRealSHT, SpectralConvS2

from .healpix_resample import HealpixToEquiangular


class SpectralCNN(nn.Module):
    """Spectral convolution CNN for scalar HEALPix map parameter estimation.

    Architecture:
        1. HEALPix → equiangular resampling
        2. 3 spectral convolution blocks with ReLU and batch norm
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

        # HEALPix → equiangular conversion
        self.to_equi = HealpixToEquiangular(nside, nlat, nlon)

        # Spectral convolution blocks
        # Block 1: 1 input channel → hidden_channels
        self.blocks = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        in_ch = 1
        for i in range(num_blocks):
            out_ch = hidden_channels if i < num_blocks - 1 else hidden_channels
            self.blocks.append(
                SpectralConvS2(
                    in_shape=(nlat, nlon),
                    out_shape=(nlat, nlon),
                    in_channels=in_ch,
                    out_channels=out_ch,
                    lmax=lmax,
                    mmax=lmax,  # full mmax for now
                )
            )
            self.batchnorms.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        self.num_blocks = num_blocks
        self.relu = nn.ReLU()

        # FC regression head
        # After spectral convs, global average pool over spatial dims
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
