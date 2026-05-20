# Alternatives Considered

## Approach alternatives

### 1. Port NNhealpix to modern PyTorch (pixel-based 1D convolution)

**How it works:** Unroll each HEALPix pixel + 8 neighbors into a flat vector, apply standard 1D convolution with stride 9. Pooling = degrade N_side with stride 4 in nested ordering.

**Pros:**
- Dead simple algorithm — trivially understandable
- Uses standard 1D conv — gets all GPU optimizations for free
- Directly operates on HEALPix (no grid conversion)
- O(N_pix) complexity

**Cons:**
- **No rotation equivariance** — the main weakness. Network wastes capacity learning rotated versions of the same pattern
- Only first-order neighbors (3×3-equivalent kernel) — limited receptive field
- The 24 pixels with 7 neighbors (instead of 8) cause distortions, especially at low N_side
- No inherent connection to spherical harmonics
- Small community (18 GitHub stars), stale

**Why not chosen:** Lack of rotation equivariance is a fundamental limitation for CMB work. The whole point of this project is to leverage spherical harmonics.

### 2. Rebuild DeepSphere on PyTorch Geometric (graph-based Chebyshev convolution)

**How it works:** Build a graph from HEALPix pixel adjacency, compute the graph Laplacian, use Chebyshev polynomial spectral graph convolution via PyG's `ChebConv`. Pooling uses HEALPix hierarchical structure.

**Pros:**
- Rotation-equivariant (with radial filters)
- O(N_pix) complexity
- PyTorch Geometric is actively maintained (23,773 stars)
- Handles partial sky naturally (sub-graphs)
- Proven approach — DeepSphere has been used in cosmology

**Cons:**
- **Still no spherical harmonics** — graph Laplacian is an approximation of the spherical Laplacian
- Graph operations (sparse matmul) are less GPU-friendly than standard convolutions or FFTs
- No direct path to spectral-domain operations (a_lm coefficients)
- More complex implementation than pixel-based or spectral approaches
- Chebyshev approximation introduces its own approximation errors

**Why not chosen:** Giuseppe's group is specifically interested in spherical harmonics. A graph-based approach doesn't give us a_lm's as first-class citizens. Also, the spectral approach (torch-harmonics) is mathematically more rigorous and better maintained.

### 3. DLWP-HPX face-based 2D convolution

**How it works:** Treats the 12 HEALPix base pixels as 12 square faces, reshapes the map into [batch, 12, H, W], and uses standard 2D Conv2d on each face. Custom `HEALPixPadding` layer handles cross-face boundaries with rotations/flips.

**Pros:**
- Uses standard 2D Conv2d — maximum GPU throughput
- U-Net architecture (encoder-decoder)
- Well-tested in weather forecasting
- Pure PyTorch, no exotic dependencies

**Cons:**
- Face boundary handling is approximate — rotations at corners introduce artifacts
- Designed for weather, not CMB — CMB cares more about angular power spectrum accuracy
- Not rotation-equivariant
- No spherical harmonics
- 12-face representation means pixels near boundaries get less accurate convolutions

**Why not chosen:** CMB analysis requires preserving exact angular power spectra. The face boundary approximations would be problematic. Also, no spherical harmonics.

### 4. Google spherical-cnn (JAX, spectral)

**How it works:** Spherical CNN using Generalized FFT for spectral-domain convolutions. Rotation-equivariant. Implements spin-weighted spherical CNNs.

**Pros:**
- State-of-the-art rotation equivariance
- Actively maintained by Google (last push May 2026)
- Supports spin-weighted fields (relevant for polarization)
- Published at ICML 2023 (Scaling Spherical CNNs)

**Cons:**
- **JAX, not PyTorch** — completely different ecosystem
- Not HEALPix-native (uses its own grid)
- Would need to translate all PyTorch infrastructure to JAX
- Less community support in cosmology (which is PyTorch-heavy)

**Why not chosen:** JAX ecosystem doesn't align with the project's needs. PyTorch is standard in the cosmology ML community and on SDSC systems.

### 5. s2cnn (older PyTorch spectral CNN)

**How it works:** PyTorch implementation of rotation-equivariant spherical CNNs using spherical harmonic transforms. 973 stars.

**Cons:**
- **Explicitly marked as outdated by the authors** — "This code is old and does not support the last versions of PyTorch! Especially since the change in the FFT interface."
- Requires Python 3.6, CUDA 9, PyTorch 0.4
- Abandonware

**Why not chosen:** Dead code. torch-harmonics is its spiritual successor.

### 6. NVIDIA torch-harmonics (CHOSEN)

**How it works:** Differentiable spherical harmonic transforms in PyTorch. Two convolution modes: SpectralConvS2 (global spectral convolution) and DiscreteContinuousConvS2 (local DISCO convolution). Both rotation-equivariant.

**Pros:**
- **Spherical harmonics as first-class citizens** — a_lm coefficients are directly accessible
- Actively maintained by NVIDIA (last push May 2026)
- GPU-optimized with CUDA kernels
- Supports distributed computation (multi-GPU SHT)
- Both spectral (global) and DISCO (local) convolution approaches
- Vector SHT for spin-2 fields (polarization)
- 670 stars, well-documented, production-quality
- Used for Spherical Fourier Neural Operators (SFNO)
- Pure PyTorch — integrates with existing cosmology ML workflows

**Cons:**
- Not HEALPix-native — uses equiangular/Legendre-Gauss grids internally
- Requires HEALPix ↔ equiangular resampling (Phase 1) or custom SHT wrapper (Phase 2)
- Spectral approach is O(N_pix log N_pix), not O(N_pix) like graph/pixel methods
- Less intuitive for people used to pixel-space operations

**Why chosen:** The spherical harmonics integration is the decisive advantage. For CMB work, operating in a_lm space is natural. The rotation equivariance comes for free. The library is actively maintained and GPU-optimized. Phase 1 (resampling) gives a quick prototype; Phase 2 (ducc0 wrapper) gives production quality.

## SHT implementation alternatives

### ducc0 (CHOSEN for Phase 2)

- Used by healpy under the hood
- Supports HEALPix ring grids natively for SHT
- Supports arbitrary grids via NUFFT
- Multi-threaded, SIMD-optimized C++17
- Provides both forward and adjoint operations (needed for autograd)

### libsharp (predecessor to ducc0)

- ducc0 is the successor to libsharp
- libsharp is no longer maintained
- ducc0 has significantly better performance

### healpy SHT

- Uses ducc0 internally, but the Python interface doesn't expose gradients
- Not differentiable
- Could be used for MCMC baselines but not for NN training

### torch-harmonics SHT

- Differentiable, GPU-optimized
- But only supports equiangular/Legendre-Gauss/Lobatto grids — not HEALPix ring grids
- This is why we need the resampling bridge (Phase 1) or ducc0 wrapper (Phase 2)

## Grid conversion alternatives

### Bilinear interpolation via torch-harmonics ResampleS2

- Built into torch-harmonics
- Differentiable
- Moderate accuracy loss for coarse N_side

### Nearest-neighbor via healpy ang2pix/pix2ang

- Simple, fast
- Pre-computable as index tensors
- More lossy than bilinear
- But might be sufficient for N_side=16

### SHT-based conversion

- HEALPix → SHT → a_lm → ISHT → equiangular
- Exact (up to bandlimiting)
- But this IS the SHT itself — if we can do this, we don't need the equiangular grid at all
- This is the Phase 2 approach

## Summary decision matrix

| Criterion | NNhealpix port | DeepSphere+PyG | DLWP-HPX | Google s-cnn | torch-harmonics |
|---|---|---|---|---|---|
| Rotation equivariance | ✗ | ✓ | ✗ | ✓ | ✓ |
| Spherical harmonics | ✗ | ✗ | ✗ | ✓ | ✓ |
| PyTorch | needs port | ✓ | ✓ | ✗ (JAX) | ✓ |
| HEALPix native | ✓ | ✓ | ✓ | ✗ | ✗ (bridge needed) |
| Actively maintained | ✗ | ✓ (PyG) | ~ | ✓ | ✓ |
| GPU optimized | ✓ (1D conv) | ~ (sparse) | ✓ (2D conv) | ✓ | ✓ |
| Vector SHT (Q/U) | ✗ | ✗ | ✗ | ✓ | ✓ |
| Complexity | O(N) | O(N) | O(N) | O(N log N) | O(N log N) |
