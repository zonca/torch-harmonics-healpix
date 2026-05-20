# torch-harmonics-healpix

Bridge HEALPix to [torch-harmonics](https://github.com/PhilChodrow/torch-harmonics) for spherical CNNs on CMB data.

Reproduces and extends benchmarks from **Krachmalnicoff & Tomasi (2019)**, ["Convolutional Neural Networks on the HEALPix sphere"](https://arxiv.org/abs/1902.04083), A&A — replacing their pixel-based NNhealpix CNN with spectral convolution via the Spherical Harmonic Transform.

## Quick Start

```bash
pip install -e ".[dev]"
```

Requires: PyTorch ≥2.0, torch-harmonics, healpy, numpy, scipy.

**⚠️ torch-harmonics must be pinned to 0.8.0 with `--no-deps`** on V100 GPUs — version ≥0.9.0 has a C++ ABI incompatibility with `disco_helpers`.

## Architecture

**SpectralCNN** — spectral convolution on the sphere via SHT:

1. **HEALPix → Equiangular** resampling (nearest-neighbor)
2. **Forward SHT** (`RealSHT`) — pixel space → harmonic space
3. **Spectral convolution** — learned complex-valued weights multiply harmonic coefficients
4. **Inverse SHT** (`InverseRealSHT`) — harmonic space → pixel space
5. **ReLU + residual connection**
6. Stack 3 blocks, then FC head (64 neurons, dropout 0.2)

**Key advantages over pixel-based CNNs (NNhealpix):**
- **Rotation equivariance** — spectral convolution is inherently equivariant
- **Direct ℓ-space operation** — spectral parameters like ℓ_p are estimated where they live
- **GPU-optimized** — SHT + matrix multiply vs pixel-neighbor gather

**Multi-channel support:** `in_channels` / `out_channels` params allow:
- Test 1: 1→1 (scalar T map → ℓ_p)
- Test 2: 3→2 (Q, U, mask → ℓ_Ep, ℓ_Bp)
- Test 3: 2→1 (Q, U → τ)

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full comparison tables, hardware specs, and runtime analysis.

### Test 1: ℓ_p estimation from scalar (T) maps

| Noise σ_n | NNhealpix | MCMC (paper) | SpectralCNN (v1) |
|-----------|-----------|-------------|------------------|
| 0         | 1.3%      | 0.7%        | **1.2%** ✅      |
| 5         | **2.9%**  | 2.5%        | 3.0%             |
| 10        | **5.2%**  | 4.8%        | 6.3%             |
| 15        | **8.4%**  | 7.8%        | 11.8%            |

**v1 setup:** 50k train, batch 64, cosine LR, 50 epochs.

At σ_n=0, SpectralCNN beats NNhealpix. At higher noise, the 50k training set is insufficient — a paper-matching run (100k train, ReduceLROnPlateau, batch 32, early stopping) is in progress.

### Test 2: ℓ_Ep / ℓ_Bp from polarization Q/U maps — *pending*
### Test 3: τ estimation from Q/U maps — *pending*

## Project Structure

```
src/torch_harmonics_healpix/
├── __init__.py
├── resample.py              # HEALPix ↔ equiangular resampling
├── models/
│   └── spectral_cnn.py      # SpectralCNN with multi-channel I/O
├── data_generation.py       # Test 1: Gaussian peak power spectra
├── data_generation_test2.py # Test 2: Q/U polarization + masks
├── data_generation_test3.py # Test 3: CAMB spectra + τ
└── mcmc_baseline.py         # χ² MCMC baseline (hp.anafast + minimize_scalar)
scripts/
├── train_test1.py           # Training loop for Test 1
├── train_test2.py           # Training loop for Test 2
└── train_test3.py           # Training loop for Test 3
slurm/                       # Slurm submission scripts
├── run_tests.slurm                  # Expanse GPU: test suite
├── run_train_test1.slurm            # Expanse GPU: Test 1 training
└── run_mcmc_benchmark_popeye.slurm  # Popeye CPU: MCMC benchmark
results/                     # JSON result files
tests/
└── test_resample.py         # Unit tests (34 tests, CPU + GPU)
```

## Remote Compute

| Platform | Use case | Partition | Access |
|----------|----------|-----------|--------|
| **Expanse** (SDSC) | GPU training/testing | `gpu-shared`, account `sds166` | `ssh expanse` |
| **Popeye** (SDSC) | CPU benchmarks (MCMC, data gen) | `gen` | `ssh popeye` (key auth) |

Code sync: `git push` from local → `git pull` on remote. **No scp/rsync.**

See [AGENTS.md](AGENTS.md) for detailed platform specs, software versions, and Slurm script documentation.

## Key Technical Notes

- **`RealSHT`/`InverseRealSHT`**: `lmax` and `mmax` are **dimension sizes** (not max indices). Output is complex.
- **`hp.synfast()`**: No `seed` kwarg in healpy ≥1.18 — use `np.random.default_rng()` or `np.random.seed()`.
- **Complex weights**: `SpectralConvBlock` stores `weight_real` and `weight_imag` as separate `nn.Parameter`, combined via `torch.complex`.
- **PYTHONUNBUFFERED=1**: Required in Slurm for real-time Python output.

## References

1. Krachmalnicoff & Tomasi (2019), A&A, arXiv:1902.04083
2. torch-harmonics: https://github.com/PhilChodrow/torch-harmonics
3. NNhealpix: https://github.com/ai4cmb/NNhealpix
