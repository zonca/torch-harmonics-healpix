---
license: mit
tags:
  - pytorch
  - spherical-cnn
  - cmb
  - healpix
  - astronomy
  - cosmology
library_name: pytorch
---

> **⚠️ Pipeline notice (2026-10, v4b):** all polarization weights trained
> before October 2026 (Tests 2, 3, 4) used healpy `synfast` with 6 spectra
> in the wrong order (EE in the TE slot) and are **invalid** — kept in this
> repo for provenance only. **Current weights: `models/v4b/*.pt`** — the
> final budget-matched campaign (100k training maps = KT19 budget, 3 seeds
> per config). Test 1 (scalar) weights are unaffected by that bug.

# torch-harmonics-healpix

Spectral CNN models for CMB parameter estimation on the HEALPix sphere, bridging [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) with HEALPix maps.

These models reproduce and improve upon the benchmarks from [Krachmalnicoff & Tomasi (2019)](https://arxiv.org/abs/1902.04083), which originally used the pixel-space [NNhealpix](https://github.com/ai4cmb/NNhealpix) architecture.

**Source code:** `https://github.com/zonca/torch-harmonics-healpix`
**Manuscript:** `https://github.com/zonca/torch-harmonics-healpix-paper`

## Model Summary

Current weights are the **v4b campaign**: 100k training maps (the KT19
budget), 3 seeds per configuration, torch 2.11.0 / torch-harmonics 0.9.2,
trained on an A100. Errors below are mean ± std across the 3 seeds.

| Model | Files | Class | Task | Input | Output | Test % error | Status |
|-------|-------|-------|------|-------|--------|-------------|--------|
| T1 (v4b) | `models/v4b/test1_v4b_noise{0,5,10,15}_seed{42,123,777}.pt` (12) | `MultiResSpectralCNN` | ℓ_peak estimation | T map | ℓ_peak | 1.61±0.22 / 3.41±0.04 / 6.54±0.01 / 11.09±0.08 | **current** |
| T2 (v4b) | `models/v4b/test2_v4b_fsky{1.0,0.5,0.2,0.1,0.05}_seed{42,123,777}.pt` (15) | `SpectralCNN` | ℓ_Ep / ℓ_Bp estimation | Q, U, mask | [ℓ_Ep, ℓ_Bp] | 2.29±0.04 / 2.27±0.10 (full sky) → 4.99±0.30 / 4.89±0.24 (f_sky=0.05) | **current** |
| T3 (v4b) | `models/v4b/test3_v4b_seed{42,123,777}.pt` (3) | `SpectralCNN` | τ estimation | Q, U, mask | τ | 3.39±0.02 | **current** |
| T1 (v2) | `models/test1_v2_fix_noise0.pt` | `SpectralCNN` (3 blocks) | ℓ_peak | T map | ℓ_peak | 1.27 | valid (scalar, unaffected) |
| T2/T3 (pre-v4) | `models/test2_v2_fix_*.pt`, `models/test3_v3.pt` | — | — | — | — | — | superseded (synfast ordering bug) |
| T4 | *(archived in the code repo)* | — | r/τ estimation | — | — | — | removed from suite (pre-v4 data) |

Model sizes: T1 `MultiResSpectralCNN` 1.53M parameters; T2/T3 `SpectralCNN` 6.9M parameters.
Full tables (including exact-likelihood MCMC and NaMaster pseudo-Cℓ baselines) are in the code repo's [BENCHMARKS.md](https://github.com/zonca/torch-harmonics-healpix/blob/main/BENCHMARKS.md).

## Architecture

**SpectralCNN** performs convolution in harmonic space instead of pixel space:

1. **HEALPix → Equiangular** resampling (nearest-neighbor interpolation)
2. **SHT** (Spherical Harmonic Transform) via torch-harmonics
3. **Learned spectral weights** — learned complex-valued spectral weights via einsum on (ℓ, m) coefficients
4. **ISHT** (Inverse SHT) back to pixel space
5. **Equiangular → HEALPix** resampling

Two variants ship in the code repo:

- **`SpectralCNN`** — every block operates at the full ℓ_max; used for Tests 2 and 3 (`num_blocks=4`, 32 hidden channels).
- **`MultiResSpectralCNN`** — each block halves ℓ_max (47 → 23 → 11 → 5), a spectral pyramid; used for Test 1 (`num_blocks=4`).

The network stacks multiple `SpectralConvBlock` layers (SHT → learned weights → ISHT + residual) followed by global average pooling and a linear head.

**Key advantage over pixel-space CNNs:** The spectral prior enforces physical smoothness in harmonic space, which is especially powerful for polarization estimation on cut skies, where classical pseudo-Cℓ estimators need NaMaster-grade machinery to control E/B leakage.

### Design Decisions

- **Inpainting for partial sky:** Masked pixels are replaced with the observed-pixel mean before SHT to prevent mode-coupling artifacts
- **Shared mask:** Train/val/test use the same mask geometry; different masks corrupt spectral coefficients
- **Scalar SHT with Q/U stacking:** torch-harmonics VectorSHT is slow, so Q/U are stacked as independent channels

See [ARCHITECTURE.md](https://github.com/zonca/torch-harmonics-healpix/blob/main/ARCHITECTURE.md) for the full comparison with NNhealpix.

## Benchmark Results (v4b campaign)

### Test 2 — Polarization ℓ_Ep / ℓ_Bp (headline: CNN dominates at every f_sky)

| f_sky | SpectralCNN (ℓ_Ep / ℓ_Bp) | NNhealpix | Improvement | NaMaster pseudo-Cℓ |
|-------|---------------------------|-----------|-------------|--------------------|
| 1.0   | **2.29 ± 0.04 / 2.27 ± 0.10** | 2.7% / 2.7% | −15% | 22.1% |
| 0.5   | **2.71 ± 0.15 / 2.69 ± 0.12** | 3.9% / 3.9% | −31% | 30.9% |
| 0.2   | **3.34 ± 0.05 / 3.20 ± 0.20** | 5.3% / 5.3% | −38% | 37.4% |
| 0.1   | **3.99 ± 0.06 / 3.68 ± 0.07** | 6.4% / 6.4% | −40% | 41.2% |
| 0.05  | **4.99 ± 0.30 / 4.89 ± 0.24** | 8.4% / 8.4% | −41% | 43.7% |

NaMaster numbers are the purified pseudo-Cℓ baseline (1000 maps, nlb=2 binning);
the naive spectrum-fit MCMC collapses on partial skies (medians 13–25%, E/B leakage).

### Test 3 — Optical depth τ (full sky)

| Method | τ % error |
|--------|----------|
| Exact-likelihood MCMC (this work) | **3.07%** |
| MCMC (KT19) | 2.8% |
| **SpectralCNN (v4b, 3 seeds)** | **3.39 ± 0.02%** |
| NNhealpix (KT19) | 4.0% |

### Test 1 — Scalar ℓ_peak (honest negative)

| σ_n | SpectralCNN (v4b) | NNhealpix | MCMC exact (this work) |
|-----|-------------------|-----------|------------------------|
| 0   | **1.61 ± 0.22%**  | 1.3%      | 0.66% |
| 5   | 3.41 ± 0.04%      | **2.9%**  | 2.43% |
| 10  | 6.54 ± 0.01%      | **4.8%**  | 4.41% |
| 15  | 11.09 ± 0.08%     | **7.8%**  | 7.20% |

SpectralCNN matches NNhealpix on noise-free maps but trails at higher noise:
SHT spreads local noise globally, while pixel-space convolution naturally
filters it. The paper reports this as an honest negative — the spectral
architecture's advantage is polarization on masked skies (Test 2), not
noisy scalar maps.

See [BENCHMARKS.md](https://github.com/zonca/torch-harmonics-healpix/blob/main/BENCHMARKS.md) for full tables including MCMC baselines.

## Usage

### Installation

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install torch-harmonics==0.9.2
uv pip install healpy astropy scipy huggingface_hub
uv pip install -e "git+https://github.com/zonca/torch-harmonics-healpix#egg=torch-harmonics-healpix"
```

### Download and Load

The class and constructor arguments differ per test — this is the exact
configuration the v4b weights were trained with:

- **Test 1:** `MultiResSpectralCNN(nside=16, hidden_channels=32, num_blocks=4, in_channels=1, out_channels=1)`
- **Test 2:** `SpectralCNN(nside=16, hidden_channels=32, num_blocks=4, in_channels=3, out_channels=2, inpaint=(f_sky < 1))`
- **Test 3:** `SpectralCNN(nside=16, hidden_channels=32, num_blocks=4, in_channels=3, out_channels=1, inpaint=False)`

```python
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from torch_harmonics_healpix.models import SpectralCNN

# Download model weights (any of the 30 v4b checkpoints)
model_path = hf_hub_download(
    repo_id="zonca/torch-harmonics-healpix",
    filename="models/v4b/test2_v4b_fsky1.0_seed42.pt",
)

# Create model with matching architecture (Test 2, full sky)
model = SpectralCNN(
    in_channels=3,       # Q, U, mask
    out_channels=2,      # [ℓ_Ep, ℓ_Bp]
    nside=16,
    hidden_channels=32,
    num_blocks=4,
    inpaint=False,       # True for f_sky < 1.0
)

# Load weights
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Run inference on a HEALPix Nside=16 map (3072 pixels)
# Stack [Q, U, mask] as 3 channels
input_tensor = torch.from_numpy(
    np.stack([q_map, u_map, mask], axis=0).astype(np.float32)
).unsqueeze(0)  # [1, 3, 3072]

with torch.no_grad():
    prediction = model(input_tensor)

print(f"Predicted parameters: {prediction[0].tolist()}")
```

For Test 1, import `MultiResSpectralCNN` from
`torch_harmonics_healpix.models.multires_spectral_cnn` and construct it
with `in_channels=1, out_channels=1` (no `inpaint` argument).

### Test 4 — REMOVED

Test 4 (joint r/τ) weights are deprecated: they were trained before the
healpy `synfast` spectra-ordering fix (v4) and are retained for
provenance only. Test 4 is out of scope for the benchmark suite.

### Retrain yourself

```bash
# Test 1: ℓ_peak from T maps (MultiResSpectralCNN)
python scripts/train_test1_v3.py --noise_std 0 --output results_v4b/test1.json

# Test 2: ℓ_Ep/ℓ_Bp from Q/U maps
python scripts/train_test2_v2.py --f_sky 0.5 --output results_v4b/test2.json

# Test 3: τ estimation (requires: pip install camb)
python scripts/train_test3_v2.py --f_sky 1.0 --output results_v4b/test3.json
```

Each script saves both `results*/*.json` (metrics) and `*.pt` (model weights);
see `slurm/train_v4b_popeye.slurm` in the code repo for the full campaign setup.

## Limitations

- **HEALPix Nside=16 only** (3072 pixels) — not tested at higher resolutions
- **Scalar SHT with stacked Q/U channels** — no explicit E/B separation; relies on the spectral prior to learn E/B structure implicitly
- **Noise sensitivity (Test 1)** — SHT spreads local noise globally; pixel-space CNNs are more robust for high-noise scalar maps
- **Partial-sky models are mask-specific** — training and inference use the same mask geometry; a different mask corrupts spectral coefficients

## Citation

If you use these models, please cite:

```bibtex
@article{krachmalnicoff2019,
  title={Convolutional Neural Networks on the {HEALPix} sphere: a pixel-based approach for CMB data analysis},
  author={Krachmalnicoff, N. and Tomasi, M.},
  journal={Astronomy \& Astrophysics},
  volume={624},
  pages={A97},
  year={2019},
  doi={10.1051/0004-6361/201834952},
  url={https://arxiv.org/abs/1902.04083}
}
```

and the torch-harmonics-healpix manuscript (Astronomy and Computing,
2026; see the `torch-harmonics-healpix-paper` repository above).

## License

MIT
