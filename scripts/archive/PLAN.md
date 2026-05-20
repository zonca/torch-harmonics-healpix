# torch-harmonics-healpix: Overall Plan

## Vision

A differentiable PyTorch library that enables spherical CNNs operating directly on HEALPix maps, powered by spherical harmonic transforms via [torch-harmonics](https://github.com/NVIDIA/torch-harmonics). The goal is to make it trivial to build, train, and deploy rotation-equivariant neural networks on cosmological (and other) spherical data stored in HEALPix format.

## Why this project?

Current CNN frameworks for HEALPix maps are stale and unmaintained:

- **NNhealpix** (Krachmalnicoff & Tomasi 2019, arXiv:1902.04083): pixel-based 1D convolution. Simple but no rotation equivariance. Last meaningful update 2021. TensorFlow-only.
- **DeepSphere** (Perraudin et al. 2019, arXiv:1810.12186): graph-based Chebyshev convolution. Rotation-equivariant with radial filters. Last update 2022. Broken `pygsp` dependency.

Both are several years old, unmaintained, and lack modern PyTorch support. Meanwhile, NVIDIA's **torch-harmonics** provides an actively maintained (as of 2026), GPU-optimized, rotation-equivariant spherical CNN framework — but it works on equiangular/Legendre-Gauss grids, not HEALPix.

**This project bridges that gap.**

## Key insight: HEALPix is iso-latitude

HEALPix maps in ring ordering are organized as iso-latitude rings with equidistant pixels per ring. This is exactly the grid type that `ducc0`'s spherical harmonic transform (the same SHT used by healpy) is optimized for. So there is no fundamental incompatibility — we just need the right interface layer.

## Architecture: Three phases

### Phase 1: HEALPix ↔ Equiangular Resampling (quick prototype)

Use torch-harmonics as-is with a resampling layer that converts HEALPix maps to equiangular grids.

**Pipeline:**
```
HEALPix map (Nside) → ResampleS2 → equiangular [nlat, nlon] → torch-harmonics Conv layers → ResampleS2 → HEALPix map
```

**Pros:** Works today with no torch-harmonics modifications. Simple.
**Cons:** Resampling is lossy (bilinear interpolation). Not ideal for preserving exact angular power spectra.

**Goal:** Validate that torch-harmonics convolutions work for CMB parameter estimation. Establish baselines.

### Phase 2: HEALPix-native SHT via ducc0 (production quality)

Wrap `ducc0`'s SHT (which natively supports HEALPix ring grids) as a differentiable PyTorch autograd layer.

**Pipeline:**
```
HEALPix map (ring order) → HealpixSHT (via ducc0) → a_lm → SpectralConv / DISCO Conv → HealpixISHT (via ducc0) → HEALPix map
```

**How the autograd works:**
- Forward: `ducc0.sht.map2alm` / `ducc0.sht.alm2map`
- Backward: use the adjoint (ducc0 provides this — `map2alm` is the adjoint of `alm2map` and vice versa)
- The Legendre transform is precomputed and cached for each grid/lmax combination

**Pros:** Exact SHT on HEALPix (no interpolation). Same library healpy uses. Best accuracy for CMB.
**Cons:** Requires writing custom PyTorch C++/CUDA extensions that call ducc0. Medium-high effort.

### Phase 3: CMB-specific models and Voyager optimization

Build production-ready models optimized for SDSC's Voyager supercomputer (AMD MI250X GPUs):

- **HealpixSFNO**: Spherical Fourier Neural Operator on HEALPix maps — for map-to-map tasks (component separation, inpainting, map generation)
- **HealpixS2UNet**: encoder-decoder on HEALPix — for segmentation, source detection
- **HealpixSpectralCNN**: spectral convolution classifier/regressor — for parameter estimation
- All with proper support for:
  - Partial sky / masks (critical for CMB)
  - Spin-2 fields (polarization Q/U) via vector SHT
  - Multi-GPU / multi-node training on Voyager
  - Mixed precision (FP16/BF16 on MI250X)

## Two convolution approaches in torch-harmonics

### SpectralConvS2 (spectral / global convolution)

```
Input [nlat, nlon] → RealSHT → [lmax, mmax] coefficients → multiply by learned weights per-(l,m) → InverseRealSHT → Output [nlat, nlon]
```

- **Global** convolution: the entire sphere is transformed to harmonic space
- Rotation-equivariant by construction
- Used in SFNO (Spherical Fourier Neural Operator)
- Best for: parameter estimation, map generation — tasks where you want to operate in ℓ-space

### DiscreteContinuousConvS2 (DISCO / local convolution)

```
Input [nlat, nlon] → local kernel (like standard CNN, but on the sphere) → Output [nlat, nlon]
```

- **Local** convolution: more similar to standard CNNs, but mathematically rigorous on the sphere
- Rotation-equivariant via spherical quadrature for kernel sampling
- Used in S2UNet
- Best for: map-to-map tasks, segmentation — tasks where local patterns matter

## Target benchmarks (from NNhealpix paper)

All use HEALPix N_side=16 (3072 pixels).

### Test 1: ℓ_p estimation from scalar (T) maps

Power spectrum model: C_ℓ = exp(-(ℓ-ℓ_p)² / (2σ²_p)) + 10⁻⁵, with σ_p=5, ℓ_p ∈ [5, 20]

| Noise level | NNhealpix | MCMC | Our target |
|---|---|---|---|
| No noise | 1.3% | 0.7% | ≤ 1.3% |
| S/N=1 (σ_n=5) | 2.9% | 2.5% | ≤ 2.5% |
| S/N=1/2 (σ_n=10) | 5.2% | 4.8% | ≤ 4.8% |
| S/N=1/3 (σ_n=15) | 8.4% | 7.8% | ≤ 7.8% |

### Test 2: ℓ_Ep, ℓ_Bp estimation from tensor (Q/U) maps

Full sky + partial sky (f_sky = 1, 0.5, 0.2, 0.1, 0.05)

| f_sky | NNhealpix | MCMC | Our target |
|---|---|---|---|
| 1.0 | 2.7% | 0.7% | ≤ 1.0% |
| 0.5 | 3.9% | — | — |
| 0.2 | 5.3% | — | — |
| 0.1 | 6.4% | — | — |
| 0.05 | 8.4% | — | — |

### Test 3: τ estimation from Q/U maps

CAMB spectra with τ ∈ [0.03, 0.08], other params fixed to Planck best-fit.

| Method | Mean % error |
|---|---|
| NNhealpix | 4.0% |
| MCMC | 2.8% |
| Our target | ≤ 3.0% |

## Why torch-harmonics should beat NNhealpix

1. **Rotation equivariance**: spectral convolutions are equivariant by construction. NNhealpix wastes capacity learning rotated versions of the same pattern.
2. **Spectral domain operation**: for ℓ_p estimation, the parameter is literally defined in ℓ-space. SpectralConvS2 operates directly there.
3. **Vector SHT**: for polarization (Test 2/3), torch-harmonics' vector SHT properly separates E/B modes instead of forcing the network to learn this from Q/U.
4. **Modern optimization**: torch-harmonics has CUDA kernels, mixed precision support, distributed SHT. NNhealpix ran on CPU with TensorFlow 1.x.

## Dependencies

- `torch` (>= 2.0)
- `torch-harmonics` (latest from PyPI/NVIDIA)
- `healpy` (for data I/O and HEALPix operations)
- `ducc0` (for Phase 2 native HEALPix SHT)
- `numpy`, `scipy` (for data generation)
- `camb` (for Test 3 τ estimation spectra)

## Development environment

- **Expanse** (SDSC): GPU development and initial benchmarks
- **Voyager** (SDSC): AMD MI250X — production training and optimization (Phase 3)

## People

- **Andrea Zonca**: implementation, architecture design
- **AI/CNN colleague** (SDSC): feedback, Voyager optimization, GPU performance tuning
- **Giuseppe Puglisi** (University of Catania): domain expertise, CMB test cases, PySM integration

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the HEALPix sphere", A&A, arXiv:1902.04083
2. Perraudin et al. (2019), "DeepSphere: Efficient spherical Convolutional Neural Network with HEALPix sampling", Astronomy & Computing, arXiv:1810.12186
3. Bonev et al. (2023), "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere", ICML, arXiv:2306.05420
4. Ocampo, Price, McEwen (2023), "Scalable and equivariant spherical CNNs by discrete-continuous convolutions", ICLR, arXiv:2209.13603
5. Górski et al. (2005), "HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere", ApJ
