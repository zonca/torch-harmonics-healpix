# torch-harmonics-healpix

Bridge HEALPix to torch-harmonics for spherical CNNs on CMB data.

Reproduces and benchmarks all 3 tests from [Krachmalnicoff & Tomasi (2019)](https://arxiv.org/abs/1902.04083) using spectral convolution networks instead of pixel-based NNhealpix.

## Quick Results (Test 1: ℓ_p estimation from T maps)

| σ_n | SpectralCNN v2 | MultiResSpectralCNN v3 | NNhealpix | MCMC |
|-----|---------------|----------------------|-----------|------|
| 0   | **1.3%**      | 1.5%                  | 1.3%      | 0.7% |
| 5   | 3.5%          | 3.5%                  | **2.9%**  | 2.5% |
| 10  | 6.8%          | _running_             | **5.2%**  | 4.8% |
| 15  | 11.8%         | _pending_             | **8.4%**  | 7.8% |

**Test 2 (Polarization) preliminary:** SpectralCNN at ℓ_Ep≈2.0%, ℓ_Bp≈1.8% — **beats NNhealpix's 2.7%/2.7%!**

See [BENCHMARKS.md](BENCHMARKS.md) for full results and [ARCHITECTURE.md](ARCHITECTURE.md) for detailed comparison.

## Setup

```bash
# Create venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-harmonics==0.8.0 --no-deps
uv pip install healpy h5py scipy

# Install in dev mode
uv pip install -e .
```

## Project Structure

```
torch-harmonics-healpix/
├── src/torch_harmonics_healpix/
│   ├── __init__.py
│   ├── healpix_resample.py          # HEALPix ↔ equiangular resampling
│   ├── data_generation.py           # Test 1: scalar power spectrum maps
│   ├── data_generation_test2.py     # Test 2: polarization Q/U maps
│   ├── data_generation_test3.py     # Test 3: CAMB spectra + τ maps
│   ├── mcmc_baseline.py             # MCMC ℓ_p estimation baseline
│   └── models/
│       ├── spectral_cnn.py          # SpectralCNN (fixed ℓ_max)
│       └── multires_spectral_cnn.py # MultiResSpectralCNN (decreasing ℓ_max)
├── scripts/
│   ├── train_test1.py               # v1 training (σ_p=3 bug, 50k maps)
│   ├── train_test1_v2.py            # v2 paper-matching (σ_p=5, 100k maps)
│   ├── train_test1_v3.py            # v3 multi-resolution spectral CNN
│   ├── train_test2_v2.py            # Test 2: polarization ℓ_Ep/ℓ_Bp
│   └── train_test3_v2.py            # Test 3: τ estimation
├── slurm/
│   ├── run_train_test1_v2.slurm     # Expanse GPU, Test 1 v2
│   ├── run_train_test1_v3.slurm     # Expanse GPU, Test 1 v3
│   ├── run_train_test2_3_v2.slurm   # Expanse GPU, Tests 2+3
│   └── run_mcmc_benchmark_popeye.slurm  # Popeye CPU, MCMC
├── tests/
│   ├── test_healpix_resample.py
│   ├── test_spectral_cnn.py
│   ├── test_data_generation.py
│   └── test_test2_test3.py
├── results/
│   ├── test1_spectralcnn_v1.json    # v1 results (σ_p=3 bug)
│   ├── test1_spectralcnn_v2.json    # v2 results (paper-matching)
│   ├── test1_v3_noise{0,5,10,15}.json  # v3 results (multi-res)
│   └── mcmc_1000maps_popeye.json    # MCMC baseline
├── ARCHITECTURE.md                  # Detailed architecture comparison
├── BENCHMARKS.md                    # Full benchmark results + hardware
├── AGENTS.md                        # Compute policies + Slurm tips
└── README.md                        # This file
```

## The Three Tests

### Test 1: ℓ_p from scalar (T) maps
Estimate the peak multipole ℓ_p of a Gaussian-peaked power spectrum
from a noisy temperature map. ℓ_p ∈ [5, 20], σ_p=5, HEALPix Nside=16.

### Test 2: ℓ_Ep/ℓ_Bp from polarization (Q/U) maps
Estimate E-mode and B-mode peak multipoles from Q/U polarization maps
with varying sky fraction f_sky. 3-channel input: Q, U, mask.

### Test 3: τ from polarization maps
Estimate the optical depth to reionization τ from Q/U maps using
realistic CAMB power spectra. τ ∈ [0.03, 0.08].

## Key Innovation (Theoretical)

**torch-harmonics provides VectorSHT for spin-2 fields** (Q/U → E/B),
which NNhealpix lacks. For Test 2 (polarization), a spectral CNN with
VectorSHT could directly operate on E/B modes rather than learning the
decomposition from pixel patterns.

**Current limitation:** torch-harmonics 0.8.0's VectorSHT is too slow for
practical use (optimization issue in spin-weighted Legendre precomputation).
Our Test 2 implementation uses scalar SHT with Q/U as independent channels,
same as NNhealpix. This is a key area for future improvement.

## Known Issues

- v1 training had σ_p=3.0 instead of paper's 5.0 (fixed in v2)
- Multi-resolution spectral blocks (v3) don't close the gap with NNhealpix at high noise — gap is due to global vs local convolution, not multi-scale features
- MCMC baseline at σ_n=0 gives 2.3% vs paper's 0.7% (likely minimize_scalar local minima)
- HEALPix→equiangular resampling uses nearest-neighbor (introduces approximation error)
- VectorSHT (spin-2) in torch-harmonics 0.8.0 too slow for training — using scalar SHT for Test 2
