"""HEALPix ↔ Equiangular grid resampling for torch-harmonics compatibility.

torch-harmonics operates on equiangular [nlat, nlon] grids, but CMB data
lives on HEALPix. This module provides differentiable PyTorch layers that
convert between the two representations using nearest-neighbor interpolation.

Phase 1 approach — simple but sufficient for Nside=16 benchmarks.
"""

import numpy as np
import healpy as hp
import torch
import torch.nn as nn


class HealpixToEquiangular(nn.Module):
    """Convert HEALPix map to equiangular [nlat, nlon] grid for torch-harmonics.

    Uses nearest-neighbor interpolation: for each (lat, lon) point on the
    equiangular grid, find the closest HEALPix pixel and copy its value.

    Args:
        nside: HEALPix Nside parameter.
        nlat: Number of latitude bands in the equiangular grid.
              Default: 2 * nside (sufficient for lmax = 3*nside - 1).
        nlon: Number of longitude points in the equiangular grid.
              Default: 2 * nlat.
    """

    def __init__(self, nside: int, nlat: int = None, nlon: int = None):
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        if nlat is None:
            nlat = 2 * nside
        if nlon is None:
            nlon = 2 * nlat

        self.nlat = nlat
        self.nlon = nlon

        # Precompute interpolation indices: for each equiangular grid point,
        # which HEALPix pixel is closest?
        lats = np.linspace(np.pi / 2, -np.pi / 2, nlat, endpoint=False)
        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # hp.ang2pix expects (theta, phi) = (colatitude, longitude)
        theta_grid = np.pi / 2 - lat_grid
        pixel_indices = hp.ang2pix(nside, theta_grid.ravel(), lon_grid.ravel())

        self.register_buffer(
            "pixel_indices", torch.from_numpy(pixel_indices.astype(np.int64))
        )

    def forward(self, healpix_map: torch.Tensor) -> torch.Tensor:
        """Convert HEALPix map(s) to equiangular grid.

        Args:
            healpix_map: [batch, npix] or [batch, channels, npix]

        Returns:
            equiangular: [batch, nlat, nlon] or [batch, channels, nlat, nlon]
        """
        if healpix_map.dim() == 2:
            # [batch, npix] -> [batch, nlat, nlon]
            equi = healpix_map[:, self.pixel_indices]
            return equi.view(-1, self.nlat, self.nlon)
        elif healpix_map.dim() == 3:
            # [batch, channels, npix] -> [batch, channels, nlat, nlon]
            equi = healpix_map[:, :, self.pixel_indices]
            return equi.view(-1, healpix_map.shape[1], self.nlat, self.nlon)
        else:
            raise ValueError(
                f"Expected 2D or 3D tensor, got {healpix_map.dim()}D"
            )


class EquiangularToHealpix(nn.Module):
    """Convert equiangular [nlat, nlon] grid back to HEALPix map.

    Uses nearest-neighbor interpolation: for each HEALPix pixel, find the
    closest point on the equiangular grid and copy its value.

    Args:
        nside: HEALPix Nside parameter.
        nlat: Number of latitude bands (must match the equiangular grid).
              Default: 2 * nside.
        nlon: Number of longitude points (must match the equiangular grid).
              Default: 2 * nlat.
    """

    def __init__(self, nside: int, nlat: int = None, nlon: int = None):
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        if nlat is None:
            nlat = 2 * nside
        if nlon is None:
            nlon = 2 * nlat

        self.nlat = nlat
        self.nlon = nlon

        # For each HEALPix pixel, find its position on the equiangular grid
        theta, phi = hp.pix2ang(nside, np.arange(self.npix))
        # theta = colatitude [0, pi], phi = longitude [0, 2pi]

        # Convert to equiangular grid indices
        # Equiangular grid: lat from +pi/2 to -pi/2 (top to bottom)
        # lat_idx = 0 at north pole, lat_idx = nlat-1 near south pole
        lat_idx = (np.pi / 2 - theta) / (np.pi / nlat)
        lon_idx = phi / (2 * np.pi / nlon)

        # Nearest-neighbor rounding
        self.register_buffer(
            "lat_indices",
            torch.from_numpy(
                np.clip(np.round(lat_idx).astype(np.int64), 0, nlat - 1)
            ),
        )
        self.register_buffer(
            "lon_indices",
            torch.from_numpy(
                np.clip(np.round(lon_idx).astype(np.int64) % nlon, 0, nlon - 1)
            ),
        )

    def forward(self, equi_map: torch.Tensor) -> torch.Tensor:
        """Convert equiangular grid map(s) to HEALPix.

        Args:
            equi_map: [batch, nlat, nlon] or [batch, channels, nlat, nlon]

        Returns:
            healpix: [batch, npix] or [batch, channels, npix]
        """
        if equi_map.dim() == 3:
            return equi_map[:, self.lat_indices, self.lon_indices]
        elif equi_map.dim() == 4:
            return equi_map[:, :, self.lat_indices, self.lon_indices]
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {equi_map.dim()}D"
            )
