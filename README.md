# torch-harmonics-healpix

Bridge HEALPix to [torch-harmonics](https://github.com/Philippe7427/torch-harmonics) for spherical CNNs on CMB data.

Reproduces and benchmarks all 4 tests from Krachmalnicoff & Tomasi (2019) [[arXiv:1902.04083](https://arxiv.org/abs/1902.04083)] and adds a Simons Observatory parameter estimation challenge using spectral convolution networks instead of pixel-based NNhealpix.

## Quick Results

### Test 2 (Polarization) — SpectralCNN dominates

| f_sky | SpectralCNN (ℓ_Ep / ℓ_Bp) | NNhealpix | Δ vs NNhealpix |
|-------|---------------------------|-----------|----------------|
| 1.0   | **1.69% / 1.53%**         | 2.7% / 2.7% | −37% / −43%  |
| 0.5   | **1.95% / 1.91%**         | 3.9% / 3.9% | −50% / −51%  |
| 0.2   | **2.15% / 2.17%**         | 5.3% / 5.3% | −59% / −59%  |
| 0.1   | **2.56% / 2.70%**         | 6.4% / 6.4% | −60% / −58%  |
| 0.05  | **3.01% / 3.11%**         | 8.4% / 8.4% | −64% / −63%  |

### Test 3 (τ estimation) — SpectralCNN wins

| Method         | τ % error |
|----------------|-----------|
| MCMC (paper)   | 2.8%      |
| **SpectralCNN** | **3.6%** |
| NNhealpix      | 4.0%      |

### Test 1 (Scalar maps) — NNhealpix better at high noise

| σ_n | SpectralCNN | NNhealpix | Winner      |
|-----|-------------|-----------|-------------|
| 0   | **1.27%**   | 1.3%      | SpectralCNN |
| 5   | 3.58%       | **2.9%**  | NNhealpix   |
| 10  | 6.81%       | **5.2%**  | NNhealpix   |
| 15  | 11.98%      | **8.4%**  | NNhealpix   |

### Test 4 (Joint r/τ estimation) — CNN approaches Fisher bound

**NSIDE=16** (9.8M params, fiducial-point evaluation):

Fiducial-point evaluation at (r=0.003, τ=0.054), 1000 noise realizations.
RMSE = √(bias² + σ²) vs Fisher σ (Cramér-Rao lower bound for unbiased estimators).

| Config              | CNN RMSE(r) / Fisher σ(r) | CNN RMSE(τ) / Fisher σ(τ) |
|---------------------|---------------------------|---------------------------|
| f_sky=1.0, noise=0  | 1.11×                     | 1.34×                     |
| f_sky=1.0, noise=6  | 1.06×                     | 0.33×                     |
| f_sky=0.1, noise=0  | 0.39×                     | 1.02×                     |
| f_sky=0.1, noise=6  | 0.38×                     | 0.56×                     |

**NSIDE=128** (422M params, fiducial-point evaluation):

| Config              | Fisher σ(r) % | CNN RMSE/Fisher (r) | CNN RMSE/Fisher (τ) |
|---------------------|---------------|---------------------|---------------------|
| f_sky=1.0, noise=0  | 7.5%          | 5.21×               | 2.90×               |
| f_sky=1.0, noise=6  | 7.7%          | 8.16×               | 1.23×               |
| f_sky=0.1, noise=0  | 23.8%         | 2.13×               | 1.66×               |
| f_sky=0.1, noise=6  | 24.2%         | 2.39×               | **0.53×**           |

> NSIDE=128 underperforms NSIDE=16 due to τ divergence at epoch 11 limiting training to 5-10 epochs. Best result: τ at f_sky=0.1/noise=6 beats Fisher bound (0.53×). Longer training or τ-stable architecture needed.

**Main finding:** SpectralCNN dominates for polarization estimation (Tests 2, 3, 4) — the spectral prior provides a strong global physical prior. For noisy scalar maps (Test 1), pixel-space convolution is more robust due to implicit low-pass filtering.

See [BENCHMARKS.md](BENCHMARKS.md) for full results and [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details.

## Pre-trained Models

Trained model weights are available on Hugging Face:

<https://huggingface.co/zonca/torch-harmonics-healpix>

| Model           | File                                     | Task              | Error          | Parameters |
|-----------------|------------------------------------------|-------------------|----------------|------------|
| SpectralCNN T1  | `models/test1_v2_fix_noise0.pt`          | ℓ_peak from T map | 1.27%          | 6.4M       |
| SpectralCNN T2  | `models/test2_v2_fix_fsky1.0.pt`         | ℓ_Ep/ℓ_Bp from Q/U | 1.69%/1.53% | 9.8M       |
| SpectralCNN T3  | `models/test3_v2_fix.pt`                 | τ from Q/U        | 3.6%           | 9.8M       |
| SpectralCNN T4  | `models/test4_fsky1.0_noise0.pt`         | r, τ from Q/U     | See Test 4     | 9.8M       |
| SpectralCNN T4  | `models/test4_fsky1.0_noise6.pt`         | r, τ from Q/U     | See Test 4     | 9.8M       |
| SpectralCNN T4  | `models/test4_fsky0.1_noise0.pt`         | r, τ from Q/U     | See Test 4     | 9.8M       |
| SpectralCNN T4  | `models/test4_fsky0.1_noise6.pt`         | r, τ from Q/U     | See Test 4     | 9.8M       |

### Downloading Weights

**Option 1: Using huggingface_hub**

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="zonca/torch-harmonics-healpix",
    filename="models/test2_v2_fix_fsky1.0.pt",
)
```

**Option 2: Direct URL**

```bash
wget https://huggingface.co/zonca/torch-harmonics-healpix/resolve/main/models/test2_v2_fix_fsky1.0.pt
```

### Loading and Using Weights

```python
import torch
import numpy as np
import healpy as hp
from torch_harmonics_healpix.models import SpectralCNN

# 1. Create model with matching architecture
model = SpectralCNN(
    in_channels=3,       # Test 1: 1 (T only), Test 2/3/4: 3 (Q, U, mask)
    out_channels=1,      # Test 1: 1, Test 2: 2 (ℓ_Ep, ℓ_Bp), Test 3: 1, Test 4: 2
    nside=16,            # HEALPix resolution
    hidden_channels=32,  # spectral convolution channels
    num_blocks=3,        # number of SpectralConvBlocks
    inpaint=False,       # True for f_sky < 1.0, False for full sky
)

# 2. Load trained weights
state_dict = torch.load("test2_v2_fix_fsky1.0.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 3. Prepare input map (HEALPix NSIDE=16, 3072 pixels)
# Test 1: shape [1, 1, 3072]  (just T map)
# Test 2/3/4: shape [1, 3, 3072] (Q, U, mask stacked)
q_map = hp.read_map("my_q_map.fits")
u_map = hp.read_map("my_u_map.fits")
mask = np.ones_like(q_map)  # 1.0 = observed, 0.0 = masked

input_map = np.stack([q_map, u_map, mask], axis=0).astype(np.float32)
input_tensor = torch.from_numpy(input_map).unsqueeze(0)  # [1, 3, 3072]

# 4. Run inference
with torch.no_grad():
    prediction = model(input_tensor)

# Test 1:  prediction[0, 0].item() → ℓ_peak
# Test 2:  prediction[0, 0].item() → ℓ_Ep,  prediction[0, 1].item() → ℓ_Bp
# Test 3:  prediction[0, 0].item() → τ
# Test 4:  prediction[0, 0].item() → log(r+1e-4),  prediction[0, 1].item() → τ
print(f"Predicted parameter: {prediction[0, 0].item():.4f}")
```

### Architecture Details per Model

**Test 1** (`in_channels=1`, `out_channels=1`, `inpaint=False`):
- Input: single T map → Output: ℓ_peak scalar
- 6.4M parameters

**Test 2** (`in_channels=3`, `out_channels=2`):
- Input: Q/U/mask → Output: [ℓ_Ep, ℓ_Bp]
- `inpaint=True` for f_sky < 1.0, `False` for full sky
- 9.8M parameters

**Test 3** (`in_channels=3`, `out_channels=1`, `inpaint=False`):
- Input: Q/U/mask → Output: τ scalar
- 9.8M parameters

**Test 4** (`in_channels=3`, `out_channels=2`, `inpaint=f_sky<1`):
- Input: Q/U/mask → Output: [log(r + 1e-4), τ]
- Joint r/τ estimation for Simons Observatory challenge
- 4 configs: f_sky ∈ {1.0, 0.1} × noise ∈ {0, 6} μK-arcmin
- Loss: MSE on [log(r + 1e-4), τ]
- 9.8M parameters (NSIDE=16), 422M parameters (NSIDE=128)

> **Note:** Pre-trained models correspond to the specific configurations above. For different noise levels, f_sky values, or architectures, retrain using the scripts in `scripts/`.

## Setup

```bash
# Create venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-harmonics==0.8.0 --no-deps
uv pip install healpy astropy scipy

# Install in dev mode
uv pip install -e .
```

For Test 3/4 (τ/r estimation), also install CAMB:

```bash
uv pip install camb
```

For downloading pre-trained models:

```bash
uv pip install huggingface_hub
```

## Project Structure

```
torch-harmonics-healpix/
├── src/torch_harmonics_healpix/
│   ├── __init__.py
│   ├── healpix_resample.py          # HEALPix ↔ equiangular resampling
│   ├── data_generation.py           # Test 1: scalar power spectrum maps
│   ├── data_generation_test2.py     # Test 2: polarization Q/U maps
│   ├── data_generation_test3.py     # Test 3: CAMB spectra + τ maps
│   ├── data_generation_test4.py     # Test 4: Simons Obs r/τ maps
│   ├── mcmc_baseline.py             # MCMC ℓ_p estimation baseline
│   ├── mcmc_baselines_test2_3.py    # MCMC baselines for Tests 2 & 3
│   └── models/
│       ├── spectral_cnn.py          # SpectralCNN (fixed ℓ_max)
│       └── multires_spectral_cnn.py # MultiResSpectralCNN (decreasing ℓ_max)
├── scripts/
│   ├── train_test1_v2.py            # Test 1 training (4 noise levels)
│   ├── train_test2_v2.py            # Test 2 training (5 f_sky values)
│   ├── train_test3_v2.py            # Test 3 training (τ estimation)
│   ├── train_test4.py               # Test 4 training (r/τ estimation)
│   ├── fisher_forecast.py           # Fisher matrix forecast
│   ├── run_mcmc_test4.py            # MCMC baseline for Test 4
│   └── eval_test4_at_fiducial.py    # Fiducial-point CNN evaluation
├── slurm/                           # Slurm scripts (Expanse + Popeye)
├── results/                         # JSON result files + .pt model weights
├── BENCHMARKS.md                    # Full benchmark comparison
├── ARCHITECTURE.md                  # Architecture comparison
└── GPU_USAGE.md                     # GPU hours tally
```

## Key Design Decisions

### 1. Inpainting for Partial-Sky

The SHT is a global transform — masked pixels set to zero corrupt spectral coefficients. We replace masked pixels with the observed-pixel mean before SHT:

```python
x_inpainted = x * mask + x_observed_mean * (1 - mask)
```

### 2. Shared Mask Across Datasets

Train/val/test must use the **same mask** (same center, same shape). The SHT encodes the absolute mask position in spectral coefficients. Different masks per split caused val/test discrepancy (4% vs 17.7% at f_sky=0.2).

### 3. Scalar SHT with Q/U Stacking

torch-harmonics VectorSHT (spin-2) is too slow in v0.8.0. We stack Q/U as independent channels with scalar SHT. Despite lacking explicit E/B separation, SpectralCNN still outperforms NNhealpix on polarization — the spectral prior captures global Q/U structure effectively.

## Retraining Models

Each training script saves both JSON results and `.pt` model weights:

```bash
# Test 1: ℓ_peak from T maps (4 noise levels)
python scripts/train_test1_v2.py --noise_std 0 --output results/test1_noise0.json
python scripts/train_test1_v2.py --noise_std 15 --output results/test1_noise15.json

# Test 2: ℓ_Ep/ℓ_Bp from Q/U maps (5 f_sky values)
python scripts/train_test2_v2.py --f_sky 1.0 --output results/test2_fsky1.0.json
python scripts/train_test2_v2.py --f_sky 0.5 --output results/test2_fsky0.5.json

# Test 3: τ estimation (requires CAMB)
python scripts/train_test3_v2.py --f_sky 1.0 --output results/test3.json

# Test 4: Joint r/τ estimation (requires CAMB)
python scripts/train_test4.py --f_sky 1.0 --noise_std 0 --output results/test4_fsky1.0_noise0.json
python scripts/train_test4.py --f_sky 0.1 --noise_std 6 --output results/test4_fsky0.1_noise6.json

# Test 4 at NSIDE=128 (higher resolution, longer training)
python scripts/train_test4.py --nside 128 --lmax 383 --f_sky 1.0 --noise_std 0 \
    --n_train 5000 --n_val 1000 --batch_size 16 \
    --output results/test4_nside128_fsky1.0_noise0.json
```

Each script outputs:
- `results/testN_*.json` — metrics, hyperparameters, comparison with baselines
- `results/testN_*.pt` — best model `state_dict` (based on validation loss)

## Running on SDSC Clusters

**Expanse (GPU):**

```bash
sbatch -p gpu-shared -A sds166 -N 1 -n 1 --gpus=1 --mem=64G -t 12:00:00 slurm/run_test4_expanse.slurm
```

**Popeye (CPU, MCMC/Fisher baselines):**

```bash
ssh popeye "cd ~/torch-harmonics-healpix && sbatch slurm/run_test4_popeye.slurm"
```

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional Neural Networks on the HEALPix sphere", A&A 624, A97. [arXiv:1902.04083](https://arxiv.org/abs/1902.04083)
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML. [arXiv:2306.05420](https://arxiv.org/abs/2306.05420)
3. [torch-harmonics](https://github.com/Philippe7427/torch-harmonics) — Spherical Harmonics in PyTorch
4. [NNhealpix](https://github.com/NToulis/nnhealpix) — Pixel-space CNN on HEALPix
