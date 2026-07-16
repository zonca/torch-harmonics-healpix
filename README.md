# torch-harmonics-healpix

Bridge HEALPix to [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) for spherical CNNs on CMB data.

Reproduces the three CMB benchmarks of Krachmalnicoff & Tomasi (2019)
[[A&A 628, A129](https://doi.org/10.1051/0004-6361/201935211), arXiv:1902.04083]
with harmonic-space (spectral) convolutions instead of the pixel-based NNhealpix,
and adds a fourth benchmark: joint tensor-to-scalar ratio r / optical depth τ
estimation relevant to the Simons Observatory, benchmarked against Fisher
(Cramér–Rao) forecasts.

> **Pipeline versions.** Results labeled **v3** use the corrected pipeline
> (CAMB `raw_cl=True` C_ℓ fix + Huber τ loss + fixed-accuracy Fisher bounds,
> `lmax_calc=500`). Earlier v1/v2 Test 3/4 results were generated with D_ℓ
> amplitudes (unit bug) and are kept only for provenance in
> [BENCHMARKS.md](BENCHMARKS.md). Tests 1–2 do not use CAMB and are
> unaffected.

## Key Results

### Test 2 (Polarization peaks) — SpectralCNN dominates

Mean error on (ℓ_Ep, ℓ_Bp) from Q/U maps, NSIDE=16:

| f_sky | SpectralCNN (ℓ_Ep / ℓ_Bp) | NNhealpix | Δ vs NNhealpix |
|-------|---------------------------|-----------|----------------|
| 1.0   | **1.69% / 1.53%**         | 2.7%      | −40%           |
| 0.5   | **1.95% / 1.91%**         | 3.9%      | −51%           |
| 0.2   | **2.15% / 2.17%**         | 5.3%      | −59%           |
| 0.1   | **2.56% / 2.70%**         | 6.4%      | −59%           |
| 0.05  | **3.01% / 3.11%**         | 8.4%      | −64%           |

The advantage *grows* as the sky fraction shrinks — the spectral prior handles
aggressive masking far better than pixel-space convolution.

### Test 1 (Scalar maps) — NNhealpix better at high noise

| σ_n (μK) | SpectralCNN | NNhealpix | Winner      |
|-----|-------------|-----------|-------------|
| 0   | **1.27%**   | 1.3%      | SpectralCNN |
| 5   | 3.58%       | **2.9%**  | NNhealpix   |
| 10  | 6.81%       | **5.2%**  | NNhealpix   |
| 15  | 11.98%      | **8.4%**  | NNhealpix   |

The global SHT spreads local noise across all modes; pixel-space pooling
filters it.

### Test 3 (τ estimation) — SpectralCNN best overall (v3)

| Method | τ error |
|--------|---------|
| **SpectralCNN (v3)** | **2.18%** |
| MCMC (KT19) | 2.8% |
| NNhealpix (KT19) | 4.0% |

With the corrected C_ℓ pipeline the SpectralCNN beats both the pixel-space
network and the published spectrum-based fit (single training run; the
superseded v2 number was 3.76% due to the D_ℓ bug).

### Test 4 (Joint r/τ, Simons Observatory) — v3 corrected

Fisher (Cramér–Rao) bounds at fiducial (r=0.003, τ=0.054), fixed CAMB
accuracy (`lmax_calc=500`):

| NSIDE | lmax | σ(r)/r, fsky=1.0/noise=0 | fsky=1.0/noise=6 | fsky=0.1/noise=0 | fsky=0.1/noise=6 |
|-------|------|--------------------------|------------------|------------------|------------------|
| 16    | 47   | 11.5%                    | 23.5%            | 36.5%            | 74.4%            |
| 32    | 95   | 8.6%                     | 19.6%            | 27.3%            | 62.1%            |
| 128   | 383  | 8.2%                     | 19.0%            | 26.0%            | 60.0%            |

SpectralCNN (range-averaged r error over the test range, v3):

- **NSIDE=16 (6.7M params): a calibrated estimator.** Fiducial-point RMSE is
  0.74–2.06× the Fisher bound on r, and the multi-fiducial response
  ⟨r̂⟩ vs r_true is linear with slope 1.02.
- **NSIDE≥32 (26.5M–422M params): the r channel collapses.** The network
  outputs r̂ ≈ 0.0011 *independently of the input* (σ ≈ 8×10⁻⁵ at
  r_true = 0.001/0.003/0.006). The apparent "55–59% error plateau" and the
  consistent +0.0007 near-zero-r bias are artifacts of that constant.
  A 16× capacity increase (26.5M → 409M params) does not fix it; the
  suspected bottleneck is training-set diversity (5000 distinct CAMB
  spectra reused ~20× each) and the scalar MSE-on-log-r objective.
- **τ keeps a real but shrunk response** at both resolutions
  (slope ≈ 0.3–0.4 across τ ∈ [0.04, 0.07]).

Full fiducial-point RMSE vs Fisher tables and the multi-fiducial response
data are in [BENCHMARKS.md](BENCHMARKS.md).

**Main finding:** spectral convolution is the right inductive bias for CMB
polarization (Tests 2–4 at low resolution), while noisy scalar maps favor
pixel-space convolution (Test 1). Scaling map-based r/τ inference to high
resolution is currently limited by simulation diversity, not architecture.

See [BENCHMARKS.md](BENCHMARKS.md) for all numbers,
[ARCHITECTURE.md](ARCHITECTURE.md) for architecture details, and the paper
repository ([torch-harmonics-healpix-paper](https://github.com/zonca/torch-harmonics-healpix-paper))
for the manuscript.

## Pre-trained Models

Trained model weights are on Hugging Face:
<https://huggingface.co/zonca/torch-harmonics-healpix>

> **Warning:** the Test 3/4 weights currently on Hugging Face were trained
> before the C_ℓ fix (v2 pipeline) and should only be used to reproduce the
> superseded v2 numbers. v3 checkpoints live on the compute clusters
> (`results_v3/*.pt`) and will replace the HF weights once the v3 campaign is
> complete. Test 1/2 weights are unaffected.

**Loading example** (Test 2, full sky):

```python
import torch
import numpy as np
import healpy as hp
from torch_harmonics_healpix.models import SpectralCNN

model = SpectralCNN(
    in_channels=3,       # Test 1: 1 (T only); Tests 2/3/4: 3 (Q, U, mask)
    out_channels=2,      # Test 1: 1; Test 2: 2; Test 3: 1; Test 4: 2
    nside=16,
    hidden_channels=32,
    num_blocks=4,        # Tests 2/3 use 4; Test 4 uses 3
    inpaint=False,       # True for f_sky < 1.0
)
model.load_state_dict(torch.load("test2_v2_fix_fsky1.0.pt", map_location="cpu"))
model.eval()

q = hp.read_map("q.fits"); u = hp.read_map("u.fits")
mask = np.ones_like(q)
x = torch.from_numpy(np.stack([q, u, mask]).astype(np.float32)).unsqueeze(0)
with torch.no_grad():
    ell_ep, ell_bp = model(x)[0].tolist()
```

Output conventions: Test 1 → ℓ_peak; Test 2 → (ℓ_Ep, ℓ_Bp); Test 3 → τ;
Test 4 → (log(r + 1e-4), τ).

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-harmonics==0.8.0 --no-deps   # >=0.9.0 has a C++ ABI issue on V100
uv pip install healpy astropy scipy h5py
uv pip install -e .

# Tests 3/4 (CAMB spectra):
uv pip install camb
```

## Project Structure

```
torch-harmonics-healpix/
├── src/torch_harmonics_healpix/
│   ├── healpix_resample.py          # HEALPix ↔ equiangular resampling
│   ├── data_generation.py           # Test 1: scalar power-spectrum maps
│   ├── data_generation_test2.py     # Test 2: polarization Q/U maps + masks
│   ├── data_generation_test3.py     # Test 3: CAMB τ spectra (raw_cl=True)
│   ├── data_generation_test4.py     # Test 4: CAMB r/τ spectra (raw_cl=True)
│   ├── mcmc_baseline.py             # χ² spectrum-fit baseline (Test 1)
│   ├── mcmc_baselines_test2_3.py    # baselines for Tests 2 & 3
│   └── models/
│       ├── spectral_cnn.py          # SpectralCNN (fixed ℓ_max)
│       └── multires_spectral_cnn.py # MultiResSpectralCNN (ablation)
├── scripts/
│   ├── train_test{1,2,3}_v2.py      # Tests 1–3 training
│   ├── train_test4.py               # Test 4 training (on-the-fly or HDF5)
│   ├── generate_test4_hdf5.py       # pre-generate NSIDE=128 datasets
│   ├── precompute_test3_camb.py     # parallel CAMB cache (CPU cluster)
│   ├── fisher_forecast.py           # Fisher bounds (fixed lmax_calc=500)
│   ├── run_fisher_mcmc_test4.py     # Fisher + MH-MCMC baseline
│   ├── eval_test4_at_fiducial.py    # fiducial-point RMSE evaluation
│   ├── compare_cnn_fisher.py        # CNN vs Fisher summary tables
│   ├── monitoring/                  # cron/agent training monitors
│   └── archive/                     # superseded one-off scripts
├── slurm/                           # Slurm jobs (Expanse GPU + Popeye CPU)
│   └── archive/                     # superseded v1/v2 jobs
├── results/                         # v1/v2 JSONs + model weights (provenance)
├── results_v3/                      # v3 (corrected) results — current
├── tests/                           # unit tests (CPU + GPU)
├── BENCHMARKS.md                    # all benchmark tables
├── ARCHITECTURE.md                  # architecture notes and comparisons
├── AGENTS.md                        # cluster workflows and pitfalls
└── GPU_USAGE.md                     # GPU-hours tally
```

## Reproducing

Each training script writes a JSON with metrics and a `.pt` checkpoint:

```bash
# Test 1 (per noise level)
python scripts/train_test1_v2.py --noise_std 0 --output results/test1_noise0.json

# Test 2 (per sky fraction)
python scripts/train_test2_v2.py --f_sky 0.5 --output results/test2_fsky0.5.json

# Test 3 (uses corrected CAMB cache; ~2h of CAMB calls without one)
python scripts/precompute_test3_camb.py --output camb_cache_test3_v3.fits
python scripts/train_test3_v2.py --camb_cache camb_cache_test3_v3.fits \
    --output results_v3/test3_v3.json

# Test 4 (per config)
python scripts/train_test4.py --nside 32 --f_sky 0.1 --noise_std 6 \
    --output results_v3/test4_nside32_fsky0.1_noise6.json

# Fisher bounds + fiducial-point evaluation
python scripts/fisher_forecast.py --nside 32 --f_sky 0.1 --noise_std 6 \
    --output results_v3/test4_fisher_nside32_fsky0.1_noise6.json
python scripts/eval_test4_at_fiducial.py --nside 32 --hidden_channels 32 \
    --results_dir results_v3
```

## Running on Clusters

- **Expanse (GPU, Slurm):** training and fiducial evaluation.
  `sbatch slurm/eval_fiducial_v3_expanse.slurm`, `sbatch slurm/train_test3_v3_expanse.slurm`, ...
- **Popeye (CPU, Slurm):** CAMB caches, Fisher forecasts, MCMC baselines.
  `sbatch slurm/fisher_multifid_popeye.slurm`, `sbatch slurm/precompute_test3_camb_popeye.slurm`, ...

Workflow: `git commit && git push` locally, `git pull` on the cluster, submit
from the repo clone. See [AGENTS.md](AGENTS.md) for accounts, environments,
and the full list of operational pitfalls (Lustre striping, HDF5 locking,
CAMB conventions, τ-divergence, shared masks).

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional neural networks on the HEALPix sphere", A&A 628, A129. [arXiv:1902.04083](https://arxiv.org/abs/1902.04083)
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML. [arXiv:2306.03838](https://arxiv.org/abs/2306.03838)
3. [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) — differentiable spherical harmonics in PyTorch
4. [NNhealpix](https://github.com/ai4cmb/NNhealpix) — pixel-space CNNs on HEALPix
