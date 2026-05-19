"""torch-harmonics-healpix: Bridge HEALPix to torch-harmonics for spherical CNNs."""

__version__ = "0.1.0"

from .healpix_resample import HealpixToEquiangular, EquiangularToHealpix
from .data_generation import generate_power_spectrum, generate_map, generate_dataset
from .data_generation_test2 import (
    generate_polarization_power_spectra,
    generate_polarization_map,
    create_sky_mask,
    generate_test2_dataset,
)
from .mcmc_baseline import mcmc_estimate_ell_p, evaluate_mcmc_baseline

# Model imports are lazy to avoid requiring torch-harmonics for basic usage
def __getattr__(name):
    if name == "SpectralCNN":
        from .models import SpectralCNN
        return SpectralCNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HealpixToEquiangular",
    "EquiangularToHealpix",
    "generate_power_spectrum",
    "generate_map",
    "generate_dataset",
    "generate_polarization_power_spectra",
    "generate_polarization_map",
    "create_sky_mask",
    "generate_test2_dataset",
    "mcmc_estimate_ell_p",
    "evaluate_mcmc_baseline",
    "SpectralCNN",
]
