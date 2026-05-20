# Benchmark Comparison: torch-harmonics-healpix vs Krachmalnicoff & Tomasi (2019)

This document tracks our reproduction of the three benchmarks from
[Krachmalnicoff & Tomasi 2019, arXiv:1902.04083](https://arxiv.org/abs/1902.04083)
(Sections 6.1.1–6.2) using spectral CNNs from torch-harmonics instead of NNhealpix.

---

## Hardware & Runtime

### Expanse (GPU — SDSC)

- **Node:** GPU-shared partition
- **GPU:** NVIDIA Tesla V100-SXM2-32GB, Driver 525.85.12, CUDA 12.0
- **CPU:** Intel Xeon Platinum 8268
- **Software:** Python 3.11.5, PyTorch 2.6.0+cu124, torch-harmonics 0.8.0, healpy 1.19.0, numpy 2.4.4, scipy 1.17.1
- **Account:** sds166

| Component | Time |
|-----------|------|
| Unit tests (34 tests) | ~5 min |
| SpectralCNN v1 training (50k maps, 50 epochs, batch 64) | ~25 min per noise level |
| SpectralCNN v2 training (100k maps, ~70 epochs, batch 32) | ~70 min per noise level |
| SpectralCNN inference (1 map) | ~0.05 ms |
| MCMC inference (1 map) | ~1.3 ms |

### Popeye (CPU — SDSC)

- **Node:** gen partition, Intel Xeon Platinum 8168 @ 2.70GHz
- **Software:** Python 3.11.11, healpy 1.19.0, numpy 2.4.6, scipy 1.17.1
- **Venv:** `~/torch-hh-venv`

| Component | Time |
|-----------|------|
| MCMC baseline (1000 maps × 4 noise levels) | ~2s per noise level |
| MCMC inference (1 map) | ~1.7 ms |

---

## Test 1: ℓ_p estimation from scalar (T) maps

**Problem:** Estimate the peak multipole ℓ_p of a Gaussian-peaked power spectrum
C_ℓ = exp(-(ℓ - ℓ_p)² / (2σ²_p)) + 10⁻⁵, with σ_p=5, ℓ_p ∈ [5, 20].

### v1 Results (σ_p=3.0 — BUG)

⚠️ **Bug:** Training script used σ_p=3.0 instead of paper's σ_p=5.0. This made peaks
narrower and ℓ_p harder to estimate. MCMC evaluation used correct σ_p=5.0, creating
a data mismatch. Results are not directly comparable to paper.

Setup: 50k train / 1k val / 500 test, batch 64, cosine LR, 50 epochs.

| Noise σ_n | NNhealpix | MCMC (paper) | MCMC (ours, 1k maps) | SpectralCNN v1 |
|-----------|-----------|-------------|----------------------|----------------|
| 0         | 1.3%      | 0.7%        | 2.3%                 | **1.2%**       |
| 5         | 2.9%      | 2.5%        | 2.5%                 | 3.0%           |
| 10        | 5.2%      | 4.8%        | 5.0%                 | 6.3%           |
| 15        | 8.4%      | 7.8%        | 7.7%                 | 11.8%          |

### v2 Results (σ_p=5.0 — paper-matching)

Setup: 100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau (patience=5, factor=0.1),
early stopping (patience=20).

| Noise σ_n | NNhealpix | MCMC (paper) | MCMC (ours, 1k maps) | SpectralCNN v2 | Epochs |
|-----------|-----------|-------------|----------------------|----------------|--------|
| 0         | 1.3%      | 0.7%        | 2.3%                 | **1.3%**       | 63     |
| 5         | **2.9%**  | 2.5%        | 2.5%                 | 3.5%           | 72     |
| 10        | **5.2%**  | 4.8%        | 5.0%                 | 6.8%           | 41     |
| 15        | **8.4%**  | 7.8%        | 7.7%                 | _running_      | —      |

**Analysis:** SpectralCNN matches NNhealpix at σ_n=0 but underperforms at higher noise.
The spectral convolution architecture captures global frequency content well in the
noiseless case, but pixel-based NNhealpix with multi-resolution pooling (Nside 16→8→4→2→1)
appears more robust to noise. Possible improvements:
- Add multi-resolution spectral blocks (decreasing ℓ_max per block)
- Increase model capacity (more blocks, wider channels)
- Use spin-weighted SHT for E/B separation (Test 2 advantage)

**Method:** Mean % error = avg(|ℓ_p_pred - ℓ_p_true| / ℓ_p_true × 100) over test set.

**MCMC discrepancy at σ_n=0:** Our MCMC gives 2.3% vs paper's 0.7%. The algorithm
is identical (χ² likelihood with cosmic variance). At σ_n=5,10,15 results match closely.
The σ_n=0 gap likely stems from `minimize_scalar` local minima.

---

## Test 2: ℓ_Ep and ℓ_Bp estimation from tensor (Q/U) maps

**Problem:** Estimate the peak multipoles of E-mode and B-mode power spectra
from polarization Q/U maps, with varying sky fraction f_sky.

**Setup:** HEALPix N_side=16, spin-2 fields, Q/U+mask stacked as 3 input channels.
v2: 100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau, early stopping.

| f_sky | NNhealpix (ℓ_Ep/ℓ_Bp) | MCMC (paper) | SpectralCNN v2 |
|-------|----------------------|-------------|----------------|
| 1.0   | 2.7% / 2.7%          | 0.7% / 0.7% | _pending_      |
| 0.5   | 3.9% / 3.9%          | —           | _pending_      |
| 0.2   | 5.3% / 5.3%          | —           | _pending_      |
| 0.1   | 6.4% / 6.4%          | —           | _pending_      |
| 0.05  | 8.4% / 8.4%          | —           | _pending_      |

**Key advantage:** torch-harmonics has vector SHT for spin-2 fields (E/B separation),
which NNhealpix lacks — it had to learn E/B from Q/U pixel patterns.

---

## Test 3: τ estimation from Q/U maps

**Problem:** Estimate the optical depth to reionization τ from CMB polarization maps,
using realistic CAMB power spectra with τ ∈ [0.03, 0.08].

**Setup:** HEALPix N_side=16, CAMB spectra (5000 pre-computed, paper approach),
other cosmological parameters fixed to Planck best-fit.
v2: 100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau, early stopping.

| Method     | Mean % error |
|------------|-------------|
| NNhealpix  | 4.0%        |
| MCMC       | 2.8%        |
| SpectralCNN v2 | _pending_ |

---

## Implementation Status

| Component                         | Status    | Notes                                              |
|-----------------------------------|-----------|----------------------------------------------------|
| HEALPix ↔ equiangular resample    | ✅ Done   | Nearest-neighbor                                   |
| Data generation (Test 1)          | ✅ Done   | data_generation.py, σ_p=5                          |
| Data generation (Test 2)          | ✅ Done   | data_generation_test2.py, spin-2 + masks           |
| Data generation (Test 3)          | ✅ Done   | data_generation_test3.py, 5000 CAMB spectra cached |
| MCMC baseline (Test 1)            | ✅ Done   | mcmc_baseline.py, 1000 maps on Popeye (CPU)        |
| SpectralCNN model                 | ✅ Done   | Multi-channel I/O, complex spectral weights        |
| Unit tests (CPU + GPU)            | ✅ Done   | 34/34 passing on Expanse V100                      |
| Training scripts v1               | ✅ Done   | 50k maps, cosine LR, 50 epochs                     |
| Training scripts v2               | ✅ Done   | 100k maps, ReduceLROnPlateau, early stopping       |
| Slurm scripts                     | ✅ Done   | slurm/ (Expanse GPU + Popeye CPU)                  |
| Test 1 v1 results                 | ✅ Done   | σ_p=3 bug, saved in results/                       |
| Test 1 v2 results                 | 🔄 Running| Expanse job 49192871, σ_n=0,5,10 done, 15 running |
| Test 2 results                    | 🔲 Next   | Scripts ready, Slurm ready                         |
| Test 3 results                    | 🔲 Planned| Scripts ready, Slurm ready, needs CAMB on Expanse  |
| Evaluation + comparison plots     | 🔲 Planned| Scatter plots, error bars vs paper                 |

---

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the HEALPix sphere", A&A, arXiv:1902.04083
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML, arXiv:2306.05420
3. Ocampo, Price, McEwen (2023), "Scalable and equivariant spherical CNNs by discrete-continuous convolutions", ICLR, arXiv:2209.13603
