"""Package-level tests: version, exports, lazy model imports."""

import pytest

import torch_harmonics_healpix


def test_version():
    assert torch_harmonics_healpix.__version__ == "0.1.0"


def test_all_exports_resolve():
    """Every name in __all__ must be importable (models load lazily)."""
    for name in torch_harmonics_healpix.__all__:
        assert hasattr(torch_harmonics_healpix, name), f"missing export: {name}"


def test_lazy_model_imports():
    """SpectralCNN / MultiResSpectralCNN must be resolvable via the lazy
    __getattr__ (no torch-harmonics import needed at package import time)."""
    from torch_harmonics_healpix import SpectralCNN, MultiResSpectralCNN

    assert callable(SpectralCNN)
    assert callable(MultiResSpectralCNN)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute"):
        torch_harmonics_healpix.does_not_exist


def test_models_subpackage():
    from torch_harmonics_healpix.models import SpectralCNN, MultiResSpectralCNN
    from torch_harmonics_healpix import models

    assert set(models.__all__) == {"SpectralCNN", "MultiResSpectralCNN"}
