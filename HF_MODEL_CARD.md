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

> **⚠️ Pipeline notice (2026-07):** the **Test 4** weights below were trained
> with the v2 pipeline, which generated maps from CAMB D_ℓ instead of C_ℓ
> amplitudes (missing `raw_cl=True`); they reproduce the superseded v2 numbers
> only. **Test 3 now has a corrected v3 checkpoint** (`models/test3_v3.pt`,
> τ error 2.18%); `models/test3_v2_fix.pt` is kept for provenance only.
> Test 1 and Test 2 weights do not use CAMB and are unaffected.

# torch-harmonics-healpix

Spectral CNN models for CMB parameter estimation on the HEALPix sphere, bridging [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) with HEALPix maps.

These models reproduce and improve upon the benchmarks from [Krachmalnicoff & Tomasi (2019)](https://arxiv.org/abs/1902.04083), which originally used the pixel-space [NNhealpix](https://github.com/ai4cmb/NNhealpix) architecture.

**Source code:** `https://github.com/zonca/torch-harmonics-healpix`

## Model Summary

| Model | File | Task | Input | Output | Error | Params |
|-------|------|------|-------|--------|-------|--------|
| SpectralCNN T1 | `models/test1_v2_fix_noise0.pt` | ℓ_peak estimation | T map | ℓ_peak | 1.27% | 6.4M |
| SpectralCNN T2 | `models/test2_v2_fix_fsky1.0.pt` | ℓ_Ep / ℓ_Bp estimation | Q, U, mask | [ℓ_Ep, ℓ_Bp] | 1.69% / 1.53% | 9.8M |
| **SpectralCNN T3 (v3)** | `models/test3_v3.pt` | τ estimation | Q, U, mask | τ | **2.18%** | 9.8M |
| SpectralCNN T3 (v2, superseded) | `models/test3_v2_fix.pt` | τ estimation | Q, U, mask | τ | 3.76% (D_ℓ bug) | 9.8M |
| SpectralCNN T4 | `models/test4_fsky1.0_noise0.pt` | r/τ estimation (f_sky=1.0, σ=0) | Q, U, mask | [log(r+1e-4), τ] | TBD | 9.8M |
| SpectralCNN T4 | `models/test4_fsky1.0_noise6.pt` | r/τ estimation (f_sky=1.0, σ=6) | Q, U, mask | [log(r+1e-4), τ] | TBD | 9.8M |
| SpectralCNN T4 | `models/test4_fsky0.1_noise0.pt` | r/τ estimation (f_sky=0.1, σ=0) | Q, U, mask | [log(r+1e-4), τ] | TBD | 9.8M |
| SpectralCNN T4 | `models/test4_fsky0.1_noise6.pt` | r/τ estimation (f_sky=0.1, σ=6) | Q, U, mask | [log(r+1e-4), τ] | TBD | 9.8M |

## Architecture

**SpectralCNN** performs convolution in harmonic space instead of pixel space:

1. **HEALPix → Equiangular** resampling (nearest-neighbor interpolation)
2. **SHT** (Spherical Harmonic Transform) via torch-harmonics
3. **Learned spectral weights** — learned complex-valued spectral weights via einsum on (ℓ, m) coefficients
4. **ISHT** (Inverse SHT) back to pixel space
5. **Equiangular → HEALPix** resampling

The network stacks multiple `SpectralConvBlock` layers (SHT → learned weights → ISHT + residual) followed by global average pooling and a linear head.

**Key advantage over pixel-space CNNs:** The spectral prior enforces physical smoothness in harmonic space, which is especially powerful for polarization estimation where E/B modes have characteristic spectral signatures.

### Design Decisions

- **Inpainting for partial sky:** Masked pixels are replaced with the observed-pixel mean before SHT to prevent mode-coupling artifacts
- **Shared mask:** Train/val/test use the same mask geometry; different masks corrupt spectral coefficients
- **Scalar SHT with Q/U stacking:** torch-harmonics v0.8.0 VectorSHT is slow, so Q/U are stacked as independent channels

See [ARCHITECTURE.md](https://github.com/zonca/torch-harmonics-healpix/blob/main/ARCHITECTURE.md) for the full comparison with NNhealpix.

## Benchmark Results

### Test 2 — Polarization (SpectralCNN dominates)

| f_sky | SpectralCNN (ℓ_Ep / ℓ_Bp) | NNhealpix | Improvement |
|-------|---------------------------|-----------|-------------|
| 1.0   | **1.69% / 1.53%**        | 2.7% / 2.7% | 37% / 43% |
| 0.5   | **1.95% / 1.91%**        | 3.9% / 3.9% | 50% / 51% |
| 0.2   | **2.15% / 2.17%**        | 5.3% / 5.3% | 59% / 59% |
| 0.1   | **2.56% / 2.70%**        | 6.4% / 6.4% | 60% / 58% |
| 0.05  | **3.01% / 3.11%**        | 8.4% / 8.4% | 64% / 63% |

### Test 3 — Optical depth τ

| Method | τ % error |
|--------|----------|
| MCMC (paper) | 2.8% |
| **SpectralCNN** | **3.76%** |
| NNhealpix | 4.0% |

### Test 1 — Scalar maps (noise-free only)

| σ_n | SpectralCNN | NNhealpix |
|-----|------------|-----------|
| 0   | **1.27%**  | 1.3%      |
| 5   | 3.58%      | **2.9%**  |

SpectralCNN wins for noise-free data but loses at high noise because SHT spreads local noise globally, while pixel-space convolution naturally filters it.

See [BENCHMARKS.md](https://github.com/zonca/torch-harmonics-healpix/blob/main/BENCHMARKS.md) for full tables including MCMC baselines.

## Usage

### Installation

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-harmonics==0.8.0 --no-deps
uv pip install healpy astropy scipy huggingface_hub
uv pip install -e "git+https://github.com/zonca/torch-harmonics-healpix#egg=torch-harmonics-healpix"
```

### Download and Load

```python
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from torch_harmonics_healpix.models import SpectralCNN

# Download model weights
model_path = hf_hub_download(
    repo_id="zonca/torch-harmonics-healpix",
    filename="models/test2_v2_fix_fsky1.0.pt",
)

# Create model with matching architecture
model = SpectralCNN(
    in_channels=3,       # Test 1: 1, Test 2/3: 3 (Q, U, mask)
    out_channels=1,      # Test 1/3: 1, Test 2: 2
    nside=16,
    hidden_channels=32,
    num_blocks=3,
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

print(f"Predicted parameter: {prediction[0, 0].item():.4f}")
```

### Test 4 — Joint r/τ estimation

```python
# Test 4: Joint r/τ estimation (Simons Observatory)
model = SpectralCNN(
    in_channels=3,       # Q, U, mask
    out_channels=2,      # [log(r + 1e-4), τ]
    nside=16,
    hidden_channels=32,
    num_blocks=3,        # Note: 3 blocks (not 4 like Tests 2/3)
    inpaint=True,        # True for f_sky < 1.0
)

model_path = hf_hub_download(
    repo_id="zonca/torch-harmonics-healpix",
    filename="models/test4_fsky0.1_noise6.pt",
)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Run inference
with torch.no_grad():
    prediction = model(input_tensor)  # shape: [1, 2]

import numpy as np
log_r = prediction[0, 0].item()
tau = prediction[0, 1].item()
r_estimate = np.exp(log_r) - 1e-4
print(f"Predicted r: {r_estimate:.6f}, τ: {tau:.4f}")
```

## Training

To retrain from scratch (e.g., for different noise levels or f_sky values):

```bash
# Test 1: ℓ_peak from T maps
python scripts/train_test1_v2.py --noise_std 0 --output results/test1_noise0.json

# Test 2: ℓ_Ep/ℓ_Bp from Q/U maps
python scripts/train_test2_v2.py --f_sky 0.5 --output results/test2_fsky0.5.json

# Test 3: τ estimation (requires: pip install camb)
python scripts/train_test3_v2.py --f_sky 1.0 --output results/test3.json
```

Each script saves both `results/*.json` (metrics) and `results/*.pt` (model weights).

## Limitations

- **HEALPix Nside=16 only** (3072 pixels) — not tested at higher resolutions
- **torch-harmonics v0.8.0** — VectorSHT too slow; uses scalar SHT with stacked Q/U channels
- **No explicit E/B separation** — relies on spectral prior to learn E/B structure implicitly
- **Noise sensitivity** — SHT spreads local noise globally; pixel-space CNNs are more robust for high-noise scalar maps
- **Full-sky pre-trained models** — partial-sky models require retraining with `inpaint=True`

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

## License

MIT
