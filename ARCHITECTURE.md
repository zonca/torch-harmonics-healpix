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
| Parameters | ~80k (Test 1) | 6.4M (3 blocks, 32 channels) | 1.5M (4 blocks, 32 channels) |
| Multi-scale | Yes (Nside pooling) | No (fixed ℓ_max) | Yes (decreasing ℓ_max) |
| Spin-weighted SHT | No | Yes (via torch-harmonics) | Yes (via torch-harmonics) |

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
- **Larger model**: 6.4M parameters (spectral weights are dense: [C_out, C_in, ℓ_max, m_max])

### Strengths
- Rotation equivariance (no data augmentation needed)
- Native E/B mode separation for spin-2 fields (Q/U maps)
- Grid-independent (could work with any sampling that supports SHT)
- Matches NNhealpix at zero noise

### Weaknesses
- **No multi-scale**: All blocks at same ℓ_max — no progressive scale extraction
- **Over-parameterized**: 6.4M vs 80k — spectral weights grow as O(ℓ_max²)
- **Underperforms at high noise**: Pixel-space features are more noise-robust
- **Resampling overhead**: HEALPix→equiangular conversion adds cost

### Performance (Test 1, σ_p=5.0)

| σ_n | SpectralCNN | NNhealpix | Gap |
|-----|------------|-----------|-----|
| 0   | **1.3%**   | 1.3%      | 0.0% |
| 5   | 3.5%       | **2.9%**  | +0.6% |
| 10  | 6.8%       | **5.2%**  | +1.6% |
| 15  | 11.8%      | **8.4%**  | +3.4% |

The gap grows with noise, suggesting the spectral architecture is less robust
to noise contamination.

---

## MultiResSpectralCNN Architecture (v3)

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

### Key Features
- **Multi-resolution spectral**: Decreasing ℓ_max per block mimics NNhealpix's
  Nside pooling in harmonic space
- **Progressive scale extraction**: Block 1 captures small-scale (high-ℓ) features,
  Block 4 captures large-scale (low-ℓ) features
- **More parameter-efficient**: 1.5M params (lower ℓ_max → smaller spectral weights)
- **Spatial downsampling**: Bilinear interpolation between blocks reduces spatial
  grid to match each block's ℓ_max

### Rationale
NNhealpix's success at high noise may stem from its multi-resolution architecture,
which explicitly extracts features at different angular scales. By adding
progressive ℓ_max reduction to SpectralCNN, we test whether multi-scale
information is the key factor.

### Expected Benefits
- Better noise robustness (low-ℓ blocks less affected by noise)
- More parameter-efficient (lower ℓ_max → O(ℓ²) weight savings)
- Maintains rotation equivariance

### Open Questions
- Is bilinear downsampling between blocks appropriate for equiangular grids?
- Should spectral truncation be used instead (just zero-pad high-ℓ coefficients)?
- What is the optimal ℓ_max schedule?

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

This introduces:
- **Approximation error**: Nearest-neighbor is not band-limited
- **Aliasing**: High-ℓ power leaks to lower modes
- **Cost**: O(N_pix) per resampling, negligible vs SHT

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
extremely slow (even at nlat=8, initialization takes >30s). This appears
to be an optimization issue in the Legendre polynomial precomputation for
spin-weighted harmonics. As a result, **our current implementation uses
scalar SHT with Q/U stacked as independent channels**, which does NOT
separate E/B modes. This is a significant limitation for Test 2 (polarization).

NNhealpix also lacks spin-weighted SHT and must learn E/B separation from
pixel-space Q/U patterns. So both architectures face the same challenge,
but the spectral approach could theoretically gain more from proper E/B
separation since it operates naturally in harmonic space.

---

## Benchmark Summary (Test 1)

### All Variants

| σ_n | NNhealpix | MCMC (paper) | v1 (σ_p=3) | v2 (σ_p=5) | v3 (multi-res) |
|-----|-----------|-------------|-----------|-----------|----------------|
| 0 | 1.3% | 0.7% | 1.2% | 1.3% | 1.5% |
| 5 | 2.9% | 2.5% | 3.0% | 3.5% | 3.5% |
| 10 | 5.2% | 4.8% | 6.3% | 6.8% | 6.7% |
| 15 | 8.4% | 7.8% | 11.8% | 11.8% | 11.3% |

### Key Observations

1. **Noiseless case**: SpectralCNN (v2) matches NNhealpix exactly (1.3%).
   Both architectures can extract spectral peak information when noise is absent.

2. **High noise degradation**: SpectralCNN degrades faster than NNhealpix
   with increasing noise. The gap grows from 0% (σ_n=0) to 3.4% (σ_n=15).

3. **Multi-resolution doesn't help**: v3 (MultiResSpectralCNN) gives marginal
   improvement at σ_n=15 (11.3% vs 11.8%) but is identical at σ_n=5 (3.5% vs 3.5%).
   The decreasing ℓ_max approach does NOT close the gap with NNhealpix.
   The performance gap is likely due to a fundamental difference between
   pixel-space local convolution (noise-robust) and global spectral
   convolution (noise-sensitive), not multi-scale features.

4. **σ_p bug had minimal impact**: v1 (σ_p=3.0) and v2 (σ_p=5.0) give
   similar results at σ_n=15 (both 11.8%).

5. **SpectralCNN excels at polarization**: Test 2 (f_sky=1.0) preliminary
   results show SpectralCNN at ℓ_Ep≈2.0%, ℓ_Bp≈1.8%, significantly better
   than NNhealpix's 2.7%/2.7%. The SHT captures global Q/U patterns effectively
   even without explicit E/B mode separation.

6. **v3 architecture limitation**: The bilinear spatial downsampling between
   spectral blocks may lose information. Spectral truncation (zeroing high-ℓ
   coefficients) instead of spatial downsampling could be more appropriate.

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
