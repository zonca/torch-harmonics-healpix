# Architecture Comparison: SpectralCNN vs NNhealpix

This document provides a detailed technical comparison of the two CNN architectures
for parameter estimation from spherical CMB maps, as implemented in this project
(torch-harmonics-healpix) and the reference paper (Krachmalnicoff & Tomasi 2019).

---

## Overview

| Feature | NNhealpix (Paper) | SpectralCNN (Ours) | MultiResSpectralCNN (Ours, v3) |
|---------|-------------------|-------------------|-------------------------------|
| Domain | Pixel space (HEALPix) | Harmonic space (equiangular SHT) | Multi-resolution harmonic space |
| Equivariance | None (learned) | Rotation-equivariant (by construction) | Rotation-equivariant |
| Input grid | HEALPix ring-order | Equiangular (via resampling) | Equiangular (via resampling) |
| Key operation | Pixel convolution + pooling | Spectral convolution (SHT → weights → ISHT) | Multi-resolution spectral convolution |
| Parameters | ~80k (Test 1), ~240k (Test 2/3) | 6.4M (T1), 9.8M (T2/3) | 1.5M (T1 only) |
| Multi-scale | Yes (Nside pooling) | No (fixed ℓ_max) | Yes (decreasing ℓ_max) |
| Spin-weighted SHT | No | Yes (via torch-harmonics, but too slow) | Yes (via torch-harmonics, but too slow) |
| Mask handling | Natural (zero pixels → zero contribution) | Inpainting before SHT | Inpainting before SHT |

---

## NNhealpix Architecture

### Structure
```
Input: HEALPix map (N_side=16, 3072 pixels)
  → NBB1: Conv(Nside=16) + BatchNorm + ReLU + MaxPool → Nside=8
  → NBB2: Conv(Nside=8)  + BatchNorm + ReLU + MaxPool → Nside=4
  → NBB3: Conv(Nside=4)  + BatchNorm + ReLU + MaxPool → Nside=2
  → NBB4: Conv(Nside=2)  + BatchNorm + ReLU + MaxPool → Nside=1
  → FC: 48 neurons → output
```

### Key Features
- **Pixel-space convolution**: Operates directly on HEALPix grid
- **Multi-resolution pooling**: Each NBB halves Nside (16→8→4→2→1), capturing
  features at progressively larger angular scales
- **Learned features**: No built-in equivariance; rotation invariance must be
  learned from training data augmentation
- **Compact**: ~80k parameters for Test 1 (32 filters, 4 NBBs)
- **Fast training**: ~6h on Cori CPU (Intel Xeon Phi 7250) for 100k maps

### Strengths
- Multi-scale pooling naturally captures angular scale information
- Compact model trains quickly
- Pixel-space features are robust to noise (local correlations)
- Masked pixels contribute zero to convolution — natural mask handling
- Proven benchmarks on all 3 tests from the paper

### Weaknesses
- No rotation equivariance — needs data augmentation
- Cannot exploit E/B mode separation for polarization
- Fixed to HEALPix grid (not portable to other samplings)
- Pixel-space convolution is heuristic (HEALPix pixels are not on a regular grid)

---

## SpectralCNN Architecture (v2)

### Structure
```
Input: HEALPix map (N_side=16, 3072 pixels)
  → HealpixToEquiangular (nearest-neighbor resampling)
  → [Inpainting: replace masked pixels with observed-pixel mean]  (f_sky < 1.0)
  → SpectralConvBlock1: SHT(ℓ_max=47) → learned weights → ISHT + BN + ReLU
  → SpectralConvBlock2: SHT(ℓ_max=47) → learned weights → ISHT + BN + ReLU
  → SpectralConvBlock3: SHT(ℓ_max=47) → learned weights → ISHT + BN
  → Global Average Pooling → FC(32→64→1)
```

### Key Features
- **Harmonic-space convolution**: Operates on spherical harmonic coefficients a_ℓm
- **Rotation equivariance**: Spectral weights commute with rotations by construction
- **Fixed resolution**: All blocks use same ℓ_max=47 (3×Nside-1)
- **Complex weights**: Real and imaginary parts learned separately for a_ℓm
- **Larger model**: 6.4M parameters (Test 1), 9.8M (Test 2/3)
- **Inpainting**: Differentiable mean replacement of masked pixels before SHT

### Strengths
- Rotation equivariance (no data augmentation needed)
- Native E/B mode separation for spin-2 fields (Q/U maps) — when VectorSHT is fast enough
- Grid-independent (could work with any sampling that supports SHT)
- **Dominates for polarization estimation** (Tests 2 & 3) — +37-64% better than NNhealpix
- Matches NNhealpix at zero noise (Test 1, σ=0)

### Weaknesses
- **No multi-scale**: All blocks at same ℓ_max — no progressive scale extraction
- **Over-parameterized**: 6.4M vs 80k — spectral weights grow as O(ℓ_max²)
- **Underperforms at high noise** (Test 1): SHT spreads noise across all modes
- **Resampling overhead**: HEALPix→equiangular conversion adds cost
- **Inpainting required**: Masked pixels must be handled explicitly (unlike pixel-space)

---

## Inpainting for Partial-Sky Observations

The SHT is a **global** transform — it operates on every pixel. When the sky
is partially observed (f_sky < 1), masked pixels set to zero are treated as
valid signal by the SHT, corrupting the spectral coefficients. At low f_sky
(≤ 0.2), 80-95% of pixels are zeros, making the SHT output meaningless.

**Solution:** Before the SHT, replace zero-masked pixels with the mean of
observed pixels (per channel, per map in the batch):

```
x_observed_mean = sum(x * mask) / sum(mask)
x_inpainted = x * mask + x_observed_mean * (1 - mask)
```

This is:
- **Differentiable** — the mean computation and conditional replacement
  support gradient flow through autograd
- **Simple** — ~10 lines of code, no extra parameters
- **Effective** — eliminates the sharp zero→signal discontinuity at mask boundary

**Critical: Shared Mask**
All datasets (train/val/test) must use the **exact same mask** (same center, same shape).
The SHT is a global operation — spectral coefficients encode the absolute position
of the mask boundary. With different masks per split, the model learns
mask-position-specific features that don't generalize. This was our main bug:
with random masks per split, f_sky=0.2 gave 4% val error but 17.7% test error.
With a shared mask, both converged to ~2.2%.

**NNhealpix comparison:** NNhealpix's pixel-space convolution naturally
handles masks because masked pixels contribute zero to the convolution
kernel — no inpainting needed, and random masks work fine because the
convolution is local.

---

## Performance Summary

### Test 1: ℓ_peak from T maps

| σ_n | SpectralCNN | NNhealpix | Gap |
|-----|------------|-----------|-----|
| 0   | **1.27%**  | 1.3%      | -0.03% (tied) |
| 5   | 3.58%      | **2.9%**  | +0.68% |
| 10  | 6.81%      | **5.2%**  | +1.61% |
| 15  | 11.98%     | **8.4%**  | +3.58% |

SpectralCNN matches at σ=0 but degrades with noise. The SHT spreads noise
across all spectral modes, while pixel-space convolution provides implicit
low-pass filtering.

### Test 2: ℓ_Ep/ℓ_Bp from Q/U maps — SpectralCNN DOMINATES

| f_sky | SpectralCNN | NNhealpix | Δ |
|-------|------------|-----------|---|
| 1.0   | **1.69%/1.53%** | 2.7%/2.7% | -37%/-43% |
| 0.5   | **1.95%/1.91%** | 3.9%/3.9% | -50%/-51% |
| 0.2   | **2.15%/2.17%** | 5.3%/5.3% | -59%/-59% |
| 0.1   | **2.56%/2.70%** | 6.4%/6.4% | -60%/-58% |
| 0.05  | **3.01%/3.11%** | 8.4%/8.4% | -64%/-63% |

The advantage **increases** with smaller f_sky. The spectral representation's
global context is overwhelmingly beneficial for partial-sky polarization.

### Test 3: τ estimation

| Method | τ % error |
|--------|----------|
| MCMC (paper) | **2.8%** |
| SpectralCNN | **3.76%** |
| NNhealpix | 4.0% |

SpectralCNN beats NNhealpix by ~6%.

### Test 4: Joint r/τ estimation (Simons Observatory)

**Configuration:** `SpectralCNN(in_channels=3, out_channels=2, nside=16, num_blocks=3, hidden_channels=32, inpaint=f_sky<1)`

- **Input:** Q/U/mask stacked [3, 3072]
- **Targets:** [log(r + 1e-4), τ] — tensor-to-scalar ratio r and optical depth τ
- **Loss:** MSE on both targets jointly
- **num_blocks=3** (differs from Tests 2/3 which use `num_blocks=4`)
- **4 configurations:** f_sky ∈ {1.0, 0.1} × noise ∈ {0, 6} μK-arcmin

| f_sky | Noise (μK-arcmin) | Inpaint | r % error | τ % error |
|-------|-------------------|---------|-----------|-----------|
| 1.0   | 0                 | False   | TBD       | TBD       |
| 1.0   | 6                 | False   | TBD       | TBD       |
| 0.1   | 0                 | True    | TBD       | TBD       |
| 0.1   | 6                 | True    | TBD       | TBD       |

Results pending — training in progress.

### Key Observations

1. **Polarization is the sweet spot**: SpectralCNN dominates for Q/U-based estimation
   (Tests 2 & 3), even without proper spin-2 SHT. The spectral prior captures
   global E/B polarization structure that pixel-space methods must learn from scratch.

2. **Noise sensitivity is the weakness**: SpectralCNN underperforms for noisy scalar
   maps (Test 1). The SHT is a global transform — noisy pixels contaminate all
   spectral coefficients.

3. **Shared mask is critical**: Different masks for train/test caused a 4× error
   increase at f_sky=0.2 (4% → 17.7%). This is specific to global transforms —
   NNhealpix doesn't have this issue.

4. **Multi-resolution doesn't help**: v3 (MultiResSpectralCNN) gave marginal
   improvement at σ_n=15 (11.3% vs 11.8%) but identical at σ_n=5.
   The noise gap is fundamental to global vs local operations, not multi-scale.

5. **Parameter overhead is large but justified**: SpectralCNN uses 40-80× more
   parameters than NNhealpix. For polarization tasks, the extra capacity is
   well-utilized. For noisy scalar tasks, it's wasted.

---

## Technical Details

### Spectral Convolution Mechanism

For a real scalar field f(θ,φ), the spectral convolution is:

```
f_out = ISHT[W * SHT(f_in)]
```

Where:
- SHT: Forward spherical harmonic transform → a_ℓm coefficients
- W: Learned complex weights [C_out, C_in, ℓ_max, m_max]
- ISHT: Inverse spherical harmonic transform → spatial field

This is rotation-equivariant because SHT coefficients transform predictably
under rotations (Wigner D-matrices), and the learned weights act on ℓ and m
independently of the coordinate system.

### HEALPix ↔ Equiangular Resampling

torch-harmonics requires equiangular grids for SHT. We resample HEALPix maps
using nearest-neighbor interpolation:

```
HEALPix (3072 pixels) → Equiangular (32×64 grid, nlat=2×Nside, nlon=4×Nside)
```

For Nside=16 (ℓ_max=47), the equiangular grid has sufficient resolution
to avoid significant aliasing.

### Spin-2 (Polarization) Handling

For Q/U polarization maps, torch-harmonics provides `RealVectorSHT` /
`InverseRealVectorSHT` for spin-2 fields:

```
(Q ± iU) = ISHT[±_2 a_ℓm]   (spin-2 spherical harmonics)
```

This naturally separates E-mode and B-mode power:
- E-mode: spheroidal component (coeffs[:, 0, :, :])
- B-mode: toroidal component (coeffs[:, 1, :, :])

**However**, in torch-harmonics 0.8.0, the Vector SHT implementation is
extremely slow. Our current implementation uses scalar SHT with Q/U stacked
as independent channels, which does NOT separate E/B modes. Despite this,
SpectralCNN still outperforms NNhealpix on polarization tasks — suggesting
that even scalar SHT provides a useful global spectral prior for Q/U data.

---

## MultiResSpectralCNN Architecture (v3) — Ablation

### Structure
```
Input: HEALPix map (N_side=16, 3072 pixels)
  → HealpixToEquiangular (nearest-neighbor resampling)
  → Block 1: Downsample → SHT(ℓ_max=47) → weights → ISHT + BN + ReLU
  → Block 2: Downsample → SHT(ℓ_max=23) → weights → ISHT + BN + ReLU
  → Block 3: Downsample → SHT(ℓ_max=11) → weights → ISHT + BN + ReLU
  → Block 4: Downsample → SHT(ℓ_max=5)  → weights → ISHT + BN
  → Global Average Pooling → FC(32→48→1)
```

### Results (Test 1 only — ablation)

| σ_n | v2 (fixed ℓ_max) | v3 (multi-res) | Δ |
|-----|------------------|----------------|---|
| 0   | 1.27%            | 1.5%           | +0.2% (worse) |
| 5   | 3.58%            | 3.5%           | -0.1% |
| 10  | 6.81%            | 6.7%           | -0.1% |
| 15  | 11.98%           | 11.3%          | -0.7% |

**Conclusion:** Multi-resolution spectral blocks provide marginal improvement
at high noise but don't close the gap with NNhealpix. The noise sensitivity
is fundamental to the global SHT operation, not the lack of multi-scale features.
v3 was not pursued further.

---

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the
   HEALPix sphere: a pixel-based approach to analyse cosmic microwave
   background data", A&A 624, A97, arXiv:1902.04083

2. Bonev et al. (2023), "Spherical Fourier Neural Operators: Learning
   Stable Dynamics on the Sphere", ICML, arXiv:2306.05420

3. Ocampo, Price, McEwen (2023), "Scalable and equivariant spherical CNNs
   by discrete-continuous convolutions", ICLR, arXiv:2209.13603

4. torch-harmonics: https://github.com/Philippe7427/torch-harmonics

5. NNhealpix: https://github.com/NToulis/nnhealpix
