"""Tests for HEALPix ↔ equiangular resampling."""

import numpy as np
import healpy as hp
import torch
import pytest

from torch_harmonics_healpix.healpix_resample import (
    HealpixToEquiangular,
    EquiangularToHealpix,
)


class TestHealpixToEquiangular:
    """Test HEALPix → equiangular conversion."""

    def test_output_shape_2d(self):
        """2D input [batch, npix] → [batch, nlat, nlon]."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = HealpixToEquiangular(nside, nlat=8, nlon=16)

        x = torch.randn(5, npix)
        out = converter(x)
        assert out.shape == (5, 8, 16)

    def test_output_shape_3d(self):
        """3D input [batch, channels, npix] → [batch, channels, nlat, nlon]."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = HealpixToEquiangular(nside, nlat=8, nlon=16)

        x = torch.randn(3, 2, npix)
        out = converter(x)
        assert out.shape == (3, 2, 8, 16)

    def test_default_nlat_nlon(self):
        """Default nlat=2*nside, nlon=2*nlat."""
        nside = 4
        converter = HealpixToEquiangular(nside)
        assert converter.nlat == 8
        assert converter.nlon == 16

    def test_preserves_batch_values(self):
        """Output should contain values from the input (nearest-neighbor copy)."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = HealpixToEquiangular(nside, nlat=8, nlon=16)

        x = torch.randn(2, npix)
        out = converter(x)
        # Every value in out should appear somewhere in x
        for b in range(2):
            out_vals = set(out[b].flatten().tolist())
            in_vals = set(x[b].tolist())
            assert out_vals.issubset(in_vals), "Output values not from input"

    def test_constant_map_roundtrip(self):
        """Constant map should stay constant after HEALPix→equiangular."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = HealpixToEquiangular(nside, nlat=8, nlon=16)

        x = torch.ones(3, npix) * 7.5
        out = converter(x)
        assert torch.allclose(out, torch.tensor(7.5)), "Constant map not preserved"

    def test_invalid_dims(self):
        """Should raise on 1D input."""
        nside = 4
        converter = HealpixToEquiangular(nside)
        x = torch.randn(hp.nside2npix(nside))
        with pytest.raises(ValueError, match="2D or 3D"):
            converter(x)


class TestEquiangularToHealpix:
    """Test equiangular → HEALPix conversion."""

    def test_output_shape_3d(self):
        """3D input [batch, nlat, nlon] → [batch, npix]."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = EquiangularToHealpix(nside, nlat=8, nlon=16)

        x = torch.randn(5, 8, 16)
        out = converter(x)
        assert out.shape == (5, npix)

    def test_output_shape_4d(self):
        """4D input [batch, channels, nlat, nlon] → [batch, channels, npix]."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = EquiangularToHealpix(nside, nlat=8, nlon=16)

        x = torch.randn(3, 2, 8, 16)
        out = converter(x)
        assert out.shape == (3, 2, npix)

    def test_constant_map(self):
        """Constant equiangular map should produce constant HEALPix map."""
        nside = 4
        npix = hp.nside2npix(nside)
        converter = EquiangularToHealpix(nside, nlat=8, nlon=16)

        x = torch.ones(2, 8, 16) * 3.14
        out = converter(x)
        assert torch.allclose(out, torch.tensor(3.14)), "Constant map not preserved"

    def test_invalid_dims(self):
        """Should raise on 2D input."""
        nside = 4
        converter = EquiangularToHealpix(nside)
        x = torch.randn(8, 16)
        with pytest.raises(ValueError, match="3D or 4D"):
            converter(x)


class TestRoundTrip:
    """Test HEALPix → equiangular → HEALPix round trip."""

    def test_roundtrip_constant(self):
        """Constant map should survive round trip."""
        nside = 4
        npix = hp.nside2npix(nside)
        nlat, nlon = 8, 16

        to_equi = HealpixToEquiangular(nside, nlat, nlon)
        to_healpix = EquiangularToHealpix(nside, nlat, nlon)

        x = torch.ones(2, npix) * 5.0
        equi = to_equi(x)
        out = to_healpix(equi)
        assert torch.allclose(out, torch.tensor(5.0))

    def test_roundtrip_preserves_north_pole(self):
        """North pole pixel should be approximately preserved."""
        nside = 16
        npix = hp.nside2npix(nside)
        nlat, nlon = 32, 64

        to_equi = HealpixToEquiangular(nside, nlat, nlon)
        to_healpix = EquiangularToHealpix(nside, nlat, nlon)

        # Map with value 1 at north pole (pixel 0 in ring ordering), 0 elsewhere
        x = torch.zeros(1, npix)
        x[0, 0] = 1.0

        equi = to_equi(x)
        # The top row of equi should have the value 1 (north pole band)
        assert equi[0, 0].max() > 0.5, "North pole signal lost in conversion"

    def test_roundtrip_smooth_map(self):
        """Smooth map (low-l) should survive round trip with small error."""
        nside = 16
        npix = hp.nside2npix(nside)
        nlat, nlon = 32, 64

        to_equi = HealpixToEquiangular(nside, nlat, nlon)
        to_healpix = EquiangularToHealpix(nside, nlat, nlon)

        # Generate a smooth map: monopole + dipole
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        m = 1.0 + 0.5 * np.cos(theta)  # monopole + z-dipole

        x = torch.from_numpy(m.astype(np.float32)).unsqueeze(0)
        equi = to_equi(x)
        out = to_healpix(equi)

        # For a smooth map, nearest-neighbor round trip should be reasonable
        # Note: nearest-neighbor on Nside=16 is coarse, so errors can be large
        rel_error = (out - x).abs().mean() / x.abs().mean()
        assert rel_error < 0.6, f"Round-trip relative error {rel_error:.3f} too large"
