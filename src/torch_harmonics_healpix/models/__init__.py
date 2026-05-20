"""Model architectures for torch-harmonics-healpix."""

from .spectral_cnn import SpectralCNN
from .multires_spectral_cnn import MultiResSpectralCNN

__all__ = ["SpectralCNN", "MultiResSpectralCNN"]
