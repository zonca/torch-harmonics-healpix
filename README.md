# torch-harmonics-healpix

Bridge HEALPix to torch-harmonics for spherical CNNs on CMB data.

Reproduces and benchmarks all 3 tests from [Krachmalnicoff & Tomasi (2019)](https://arxiv.org/abs/1902.04083) using spectral convolution networks instead of pixel-based NNhealpix.

## Quick Results

### Test 2 (Polarization) — SpectralCNN DOMINATES ✅

| f_sky | SpectralCNN (ℓ_Ep/ℓ_Bp) | NNhealpix | Δ vs NNhealpix |
|-------|-------------------------|-----------|----------------|
| 1.0   | **1.69%/1.53%**        | 2.7%/2.7% | -37%/-43% |
| 0.5   | **1.95%/1.91%**        | 3.9%/3.9% | -50%/-51% |
| 0.2   | **2.15%/2.17%**        | 5.3%/5.3% | -59%/-59% |
| 0.1   | **2.56%/2.70%**        | 6.4%/6.4% | -60%/-58% |
| 0.05  | **3.01%/3.11%**        | 8.4%/8.4% | -64%/-63% |

### Test 3 (τ estimation) — SpectralCNN wins ✅

| Method | τ % error |
|--------|----------|
| MCMC (paper) | **2.8%** |
| SpectralCNN | **3.76%** |
| NNhealpix | 4.0% |

### Test 1 (Scalar maps) — NNhealpix better at high noise

| σ_n | SpectralCNN | NNhealpix | Winner |
|-----|------------|-----------|--------|
| 0   | **1.27%**  | 1.3%      | SpectralCNN ✅ |
| 5   | 3.58%      | **2.9%**  | NNhealpix |
| 10  | 6.81%      | **5.2%**  | NNhealpix |
| 15  | 11.98%     | **8.4%**  | NNhealpix |

**Main finding:** SpectralCNN dominates for polarization estimation (Tests 2 & 3) —
the spectral prior provides a strong global physical prior. For noisy scalar maps
(Test 1), pixel-space convolution is more robust due to implicit low-pass filtering.

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

For Test 3 (τ estimation), also install CAMB:
```bash
uv pip install camb
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
│   ├── mcmc_baselines_test2_3.py    # MCMC baselines for Tests 2 & 3
│   └── models/
│       ├── spectral_cnn.py          # SpectralCNN (fixed ℓ_max)
│       └── multires_spectral_cnn.py # MultiResSpectralCNN (decreasing ℓ_max)
├── scripts/
│   ├── train_test1_v2.py            # Test 1 training (4 noise levels)
│   ├── train_test2_v2.py            # Test 2 training (5 f_sky values)
│   └── train_test3_v2.py            # Test 3 training (τ estimation)
├── tests/
│   ├── test_healpix_resample.py     # Resampling roundtrip tests
│   ├── test_inpainting.py           # Inpainting unit tests
│   └── test_spectral_cnn_gpu.py     # GPU integration tests
├── slurm/                           # Slurm scripts (Expanse + Popeye)
├── results/                         # JSON result files
├── BENCHMARKS.md                    # Full benchmark comparison
├── ARCHITECTURE.md                  # Architecture comparison
└── GPU_USAGE.md                     # GPU hours tally
```

## Key Design Decisions

### 1. Inpainting for Partial-Sky
The SHT is a global transform — masked pixels set to zero corrupt spectral
coefficients. We replace masked pixels with the observed-pixel mean before SHT:
```python
x_inpainted = x * mask + x_observed_mean * (1 - mask)
```

### 2. Shared Mask Across Datasets
Train/val/test must use the **same mask** (same center, same shape). The SHT
encodes the absolute mask position in spectral coefficients. Different masks
per split caused val/test discrepancy (4% vs 17.7% at f_sky=0.2).

### 3. Scalar SHT with Q/U Stacking
torch-harmonics VectorSHT (spin-2) is too slow in v0.8.0. We stack Q/U as
independent channels with scalar SHT. Despite lacking explicit E/B separation,
SpectralCNN still outperforms NNhealpix on polarization — the spectral prior
captures global Q/U structure effectively.

## Running on SDSC Clusters

**Expanse (GPU):**
```bash
sbatch -p gpu-shared -A sds166 -N 1 -n 1 --gpus=1 --mem=64G -t 12:00:00 slurm/run_test3_only.slurm
```

**Popeye (CPU, MCMC baselines):**
```bash
ssh popeye "cd ~/torch-harmonics-healpix && sbatch slurm/run_mcmc_all_popeye.slurm"
```

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the HEALPix sphere", A&A 624, A97, arXiv:1902.04083
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML, arXiv:2306.05420
3. torch-harmonics: https://github.com/Philippe7427/torch-harmonics
4. NNhealpix: https://github.com/NToulis/nnhealpix
