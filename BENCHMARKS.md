# Benchmark Comparison: torch-harmonics-healpix vs Krachmalnicoff & Tomasi (2019)

This document tracks our reproduction of the three benchmarks from
[Krachmalnicoff & Tomasi 2019, arXiv:1902.04083](https://arxiv.org/abs/1902.04083)
(Sections 6.1.1–6.1.3) using spectral CNNs from torch-harmonics instead of NNhealpix.

---

## Test 1: ℓ_p estimation from scalar (T) maps

**Problem:** Estimate the peak multipole ℓ_p of a Gaussian-peaked power spectrum
C_ℓ = exp(-(ℓ - ℓ_p)² / (2σ²_p)) + 10⁻⁵, with σ_p=5, ℓ_p ∈ [5, 20].

**Setup:** HEALPix N_side=16 (3072 pixels), 100k train / 10k val / 1k test maps.

| Noise σ_n | NNhealpix | MCMC (paper) | MCMC (ours) | SpectralCNN (ours) |
|-----------|-----------|-------------|-------------|-------------------|
| 0         | 1.3%      | 0.7%        | 2.7%        | **1.1%**          |
| 5         | 2.9%      | 2.5%        | 3.1%        | _pending_         |
| 10        | 5.2%      | 4.8%        | 3.5%        | _pending_         |
| 15        | 8.4%      | 7.8%        | 7.2%        | _pending_         |

**Method:** Mean % error = avg(|ℓ_p_pred - ℓ_p_true| / ℓ_p_true × 100) over test set.

**Why SpectralCNN should win:**
- ℓ_p is literally a spectral parameter — SpectralConvS2 operates directly in ℓ-space
- Rotation equivariance is free (no wasted capacity learning rotated patterns)
- Modern GPU-optimized PyTorch vs 2019 TensorFlow on CPU

---

## Test 2: ℓ_Ep and ℓ_Bp estimation from tensor (Q/U) maps

**Problem:** Estimate the peak multipoles of E-mode and B-mode power spectra
from polarization Q/U maps, with varying sky fraction f_sky.

**Setup:** HEALPix N_side=16, spin-2 fields via vector SHT.

| f_sky | NNhealpix (ℓ_Ep) | NNhealpix (ℓ_Bp) | SpectralCNN (ours) |
|-------|-------------------|-------------------|---------------------|
| 1.0   | 2.7%              | 2.7%              | _pending_           |
| 0.5   | 3.9%              | 3.9%              | _pending_           |
| 0.2   | 5.3%              | 5.3%              | _pending_           |
| 0.1   | 6.4%              | 6.4%              | _pending_           |
| 0.05  | 8.4%              | 8.4%              | _pending_           |

**Key challenge:** torch-harmonics has vector SHT for spin-2 fields (E/B separation),
which NNhealpix lacks — the network had to learn E/B from Q/U. This should be
a significant advantage.

**Partial sky handling:** Mask pixels outside observed region before SHT.
SpectralConvS2 can work with masked input by zeroing unobserved pixels
(common approximation, though not mathematically exact for SHT).

---

## Test 3: τ estimation from Q/U maps

**Problem:** Estimate the optical depth to reionization τ from CMB polarization maps,
using realistic CAMB power spectra with τ ∈ [0.03, 0.08].

**Setup:** HEALPix N_side=16, CAMB spectra, other cosmological parameters
fixed to Planck best-fit.

| Method     | Mean % error |
|------------|-------------|
| NNhealpix  | 4.0%        |
| MCMC       | 2.8%        |
| SpectralCNN (ours) | _pending_ |

**Key challenge:** This is the most physically realistic test. τ imprints a
large-scale polarization signal (ℓ < 10) that is subtle and easily confused
with noise. Our approach:

1. Use vector SHT to separate E/B modes (τ signal is in E-mode)
2. SpectralConvS2 can focus on the low-ℓ modes where τ lives
3. The spectral domain representation naturally emphasizes large-scale patterns

**Stretch goal:** ≤ 3.0% (beat NNhealpix, approach MCMC)

---

## Implementation Status

| Component                    | Status      | Notes                                    |
|------------------------------|-------------|------------------------------------------|
| HEALPix ↔ equiangular resample | ✅ Done    | Nearest-neighbor, Phase 1               |
| Data generation (Test 1)     | ✅ Done     | generate_test1_data.py                   |
| MCMC baseline (Test 1)       | ✅ Done     | mcmc_baseline.py, benchmarked on Expanse |
| SpectralCNN model            | ✅ Done     | models/spectral_cnn.py, GPU tests pass   |
| Unit tests (CPU + GPU)       | ✅ Done     | 34/34 passing on Expanse V100            |
| Data generation (Test 2)     | ✅ Done     | data_generation_test2.py, spin-2 + masks |
| Data generation (Test 3)     | ✅ Done     | data_generation_test3.py, CAMB spectra   |
| Training script (Test 1)     | 🔲 Next     | Need to implement train_test1.py         |
| Training scripts (Test 2/3)  | 🔲 Planned  |                                          |
| Evaluation + comparison plots| 🔲 Planned  | Scatter plots, error bars vs paper       |

---

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the HEALPix sphere", A&A, arXiv:1902.04083
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML, arXiv:2306.05420
3. Ocampo, Price, McEwen (2023), "Scalable and equivariant spherical CNNs by discrete-continuous convolutions", ICLR, arXiv:2209.13603
