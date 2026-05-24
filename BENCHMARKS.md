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
| SpectralCNN v2 training (100k maps, ~70 epochs, batch 32) | ~70 min per noise level |
| SpectralCNN inference (1 map) | ~0.05 ms |
| MCMC inference (1 map) | ~1.7 ms |

### Popeye (CPU — SDSC)

- **Node:** gen partition, Intel Xeon Platinum 8168 @ 2.70GHz
- **Software:** Python 3.11.11, healpy 1.19.0, numpy 2.4.6, scipy 1.17.1
- **Venv:** `~/torch-hh-venv`

| Component | Time |
|-----------|------|
| MCMC baseline (1000 maps × 4 noise levels) | ~2s per noise level |
| MCMC inference (1 map) | ~1.7 ms |

---

## Summary of Key Results

| Test | Config | SpectralCNN | NNhealpix | MCMC | Winner |
|------|--------|------------|-----------|------|--------|
| T1 | σ=0 | **1.27%** | 1.3% | 2.22% | SpectralCNN ✅ |
| T1 | σ=5 | 3.58% | **2.9%** | 2.87% | NNhealpix |
| T1 | σ=10 | 6.81% | **5.2%** | 5.18% | NNhealpix |
| T1 | σ=15 | 11.98% | **8.4%** | 8.24% | NNhealpix |
| T2 | f_sky=1.0 | **1.69%/1.53%** | 2.7%/2.7% | 7.1%/6.9% | SpectralCNN ✅ |
| T2 | f_sky=0.5 | **1.95%/1.91%** | 3.9%/3.9% | 82.8%/79.2% | SpectralCNN ✅ |
| T2 | f_sky=0.2 | **2.15%/2.17%** | 5.3%/5.3% | 80.5%/76.5% | SpectralCNN ✅ |
| T2 | f_sky=0.1 | **2.56%/2.70%** | 6.4%/6.4% | 75.7%/71.3% | SpectralCNN ✅ |
| T2 | f_sky=0.05 | **3.01%/3.11%** | 8.4%/8.4% | 66.3%/63.3% | SpectralCNN ✅ |
| T3 | full sky | **3.76%** | 4.0% | 2.8% (paper) | SpectralCNN ✅ |

**Main finding:** SpectralCNN dominates for polarization estimation (Tests 2 & 3) but
underperforms for noisy scalar maps (Test 1). The spectral representation provides a
strong global prior for clean/polarized data but is sensitive to noise in scalar fields.

---

## Test 1: ℓ_p estimation from scalar (T) maps

**Problem:** Estimate the peak multipole ℓ_p of a Gaussian-peaked power spectrum
C_ℓ = exp(-(ℓ - ℓ_p)² / (2σ²_p)) + 10⁻⁵, with σ_p=5, ℓ_p ∈ [5, 20].

### v2 Results (σ_p=5.0 — paper-matching)

Setup: 100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau (patience=5, factor=0.1),
early stopping (patience=20).

| Noise σ_n | NNhealpix | MCMC (paper) | MCMC (ours) | SpectralCNN v2 | Epochs |
|-----------|-----------|-------------|-------------|----------------|--------|
| 0         | 1.3%      | 0.7%        | 2.22%       | **1.27%**      | 86     |
| 5         | **2.9%**  | 2.5%        | 2.87%       | 3.58%          | 66     |
| 10        | **5.2%**  | 4.8%        | 5.18%       | 6.81%          | 117    |
| 15        | **8.4%**  | 7.8%        | 8.24%       | 11.98%         | 48     |

**Analysis:** SpectralCNN matches NNhealpix at σ_n=0 but underperforms at higher noise.
The spectral convolution architecture captures global frequency content well in the
noiseless case, but pixel-based NNhealpix with multi-resolution pooling (Nside 16→8→4→2→1)
appears more robust to noise. The SHT spreads noise across all spectral modes, while
pixel-space convolution provides implicit low-pass filtering.

---

## Test 2: ℓ_Ep and ℓ_Bp estimation from tensor (Q/U) maps

**Problem:** Estimate the peak multipoles of E-mode and B-mode power spectra
from polarization Q/U maps, with varying sky fraction f_sky.

**Setup:** HEALPix N_side=16, spin-2 fields, Q/U+mask stacked as 3 input channels.
100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau, early stopping.
**Shared mask** across train/val/test (critical for SHT-based models — see ARCHITECTURE.md).
**Inpainting** for f_sky < 1.0 (replace masked zeros with observed-pixel mean before SHT).

| f_sky | NNhealpix (ℓ_Ep/ℓ_Bp) | MCMC (ours) | SpectralCNN v2 (ℓ_Ep/ℓ_Bp) | Epochs | Δ vs NNhealpix |
|-------|----------------------|-------------|---------------------------|--------|---------------|
| 1.0   | 2.7% / 2.7%          | 7.1% / 6.9% | **1.69% / 1.53%**        | 63     | **-37% / -43%** |
| 0.5   | 3.9% / 3.9%          | 82.8%/79.2% | **1.95% / 1.91%**        | 56     | **-50% / -51%** |
| 0.2   | 5.3% / 5.3%          | 80.5%/76.5% | **2.15% / 2.17%**        | 84     | **-59% / -59%** |
| 0.1   | 6.4% / 6.4%          | 75.7%/71.3% | **2.56% / 2.70%**        | 85     | **-60% / -58%** |
| 0.05  | 8.4% / 8.4%          | 66.3%/63.3% | **3.01% / 3.11%**        | 80     | **-64% / -63%** |

**Key finding:** SpectralCNN **dominates** NNhealpix at all sky fractions for polarization
estimation. The advantage **increases** with smaller f_sky: from +37% at full sky to +64%
at 5% sky. The spectral representation's global context is overwhelmingly beneficial for
partial-sky polarization, where E/B separation via spin-2 SHT provides a strong physical
prior that pixel-space methods must learn from scratch.

**Why MCMC fails:** Our MCMC baseline uses naive χ² fitting on full-sky power spectra,
which doesn't properly handle E/B leakage from masks. The paper's MCMC result (0.7%)
uses specialized CMB tools with proper E/B separation.

**Why shared mask matters:** The SHT is a global operation — spectral coefficients encode
the absolute position of the mask boundary. With different masks for train/val/test
(our initial bug), the model learned mask-position-specific features that didn't
generalize. Using a single shared mask (as the paper does) fixed this completely.

---

## Test 3: τ estimation from Q/U maps

**Problem:** Estimate the optical depth to reionization τ from CMB polarization maps,
using realistic CAMB power spectra with τ ∈ [0.03, 0.08].

**Setup:** HEALPix N_side=16, CAMB spectra (5000 pre-computed, paper approach),
other cosmological parameters fixed to Planck best-fit.
100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau, early stopping.

| Method         | Mean % error |
|----------------|-------------|
| MCMC (paper)   | **2.8%**    |
| SpectralCNN | **3.6%** |
| NNhealpix      | 4.0%        |
| MCMC (ours)    | 19.3%       |

SpectralCNN beats NNhealpix by ~6% on τ estimation. MCMC (paper) remains best,
but our MCMC baseline is weak because it uses naive template matching rather than
proper Bayesian inference.

---

## Test 4: Joint r/τ estimation for Simons Observatory

**Problem:** Estimate both the tensor-to-scalar ratio r and the optical depth τ
from CMB polarization Q/U maps, using realistic CAMB power spectra with
r ∈ [0, 1] and τ ∈ [0.03, 0.08].

**Setup:** HEALPix N_side=16, CAMB spectra (FITS-cached via astropy),
other cosmological parameters fixed to Planck best-fit.
100k train / 10k val / 1k test, batch 32, ReduceLROnPlateau, early stopping.

**Targets:** [log(r + 1e-4), τ] with MSE loss.
**Model:** `SpectralCNN(in_channels=3, out_channels=2, nside=16, num_blocks=3, hidden_channels=32, inpaint=f_sky<1)`

### Configurations

| Config | f_sky | Noise (μK-arcmin) | Inpaint | Model File |
|--------|-------|-------------------|---------|------------|
| T4-a   | 1.0   | 0                 | False   | `test4_fsky1.0_noise0.pt` |
| T4-b   | 1.0   | 6                 | False   | `test4_fsky1.0_noise6.pt` |
| T4-c   | 0.1   | 0                 | True    | `test4_fsky0.1_noise0.pt` |
| T4-d   | 0.1   | 6                 | True    | `test4_fsky0.1_noise6.pt` |

### Results

**Results pending — training in progress.**

| Config | Fisher forecast (r % / τ %) | SpectralCNN (r % / τ %) | MCMC (r % / τ %) |
|--------|-----------------------------|--------------------------|-------------------|
| T4-a   | TBD / TBD                   | TBD / TBD                | TBD / TBD         |
| T4-b   | TBD / TBD                   | TBD / TBD                | TBD / TBD         |
| T4-c   | TBD / TBD                   | TBD / TBD                | TBD / TBD         |
| T4-d   | TBD / TBD                   | TBD / TBD                | TBD / TBD         |

---

## Architecture Comparison

| Property | SpectralCNN | NNhealpix |
|----------|------------|-----------|
| Domain | Spectral (ℓ, m) | Pixel (HEALPix) |
| Parameters (T1) | 6,454,529 | ~80,000 |
| Parameters (T2/T3) | 9,829,634 | ~240,000 |
| Parameter scaling | O(ℓ²_max × C²) | O(filter² × C²) |
| Parameter overhead | 40-80× more | baseline |
| SHT type | RealSHT (scalar) | N/A |
| Spin-2 support | Via Q/U stacking | Learned from pixels |
| Mask handling | Inpainting required | Natural (zero contributes nothing) |
| Noise sensitivity | High (SHT spreads noise) | Low (implicit low-pass) |
| Best for | Clean/polarized data | Noisy/scalar data |

---

## Implementation Status

| Component                         | Status    | Notes                                              |
|-----------------------------------|-----------|----------------------------------------------------|
| HEALPix ↔ equiangular resample    | ✅ Done   | Nearest-neighbor                                   |
| Data generation (Test 1)          | ✅ Done   | data_generation.py, σ_p=5                          |
| Data generation (Test 2)          | ✅ Done   | data_generation_test2.py, spin-2 + masks           |
| Data generation (Test 3)          | ✅ Done   | data_generation_test3.py, 5000 CAMB spectra cached |
| MCMC baseline (Test 1)            | ✅ Done   | mcmc_baseline.py, 1000 maps on Popeye (CPU)        |
| MCMC baseline (Test 2+3)          | ✅ Done   | mcmc_baselines_test2_3.py, Popeye (CPU)            |
| SpectralCNN model                 | ✅ Done   | Multi-channel I/O, complex spectral weights        |
| Inpainting for masked pixels      | ✅ Done   | Observed-pixel mean replacement before SHT         |
| Shared mask across datasets       | ✅ Done   | Critical fix for SHT-based models                  |
| Inpainting unit tests             | ✅ Done   | test_inpainting.py (8 tests)                       |
| Unit tests (CPU + GPU)            | ✅ Done   | 34/34 passing on Expanse V100                      |
| Training scripts v2               | ✅ Done   | 100k maps, ReduceLROnPlateau, early stopping       |
| Test 1 v2 results (all noise)     | ✅ Done   | 1.27%, 3.58%, 6.81%, 11.98%                        |
| Test 2 v2 results (all f_sky)     | ✅ Done   | 1.69-3.01% (all beat NNhealpix)                    |
|| Test 3 v2 result                  | ✅ Done   | 3.76% (beats NNhealpix 4.0%)                       |
|| Data generation (Test 4)          | ✅ Done   | data_generation_test4.py, FITS CAMB cache via astropy |
|| Test 4 training (4 configs)       | 🔄 WIP    | 4 configs: f_sky∈{1.0,0.1} × noise∈{0,6}            |

---

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the HEALPix sphere", A&A, arXiv:1902.04083
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML, arXiv:2306.05420
3. Ocampo, Price, McEwen (2023), "Scalable and equivariant spherical CNNs by discrete-continuous convolutions", ICLR, arXiv:2209.13603
