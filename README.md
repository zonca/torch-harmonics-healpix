# torch-harmonics-healpix

Bridge HEALPix to [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) for spherical CNNs on CMB data.

Reproduces the three CMB benchmarks of Krachmalnicoff & Tomasi (2019)
[[A&A 628, A129](https://doi.org/10.1051/0004-6361/201935211), arXiv:1902.04083]
with harmonic-space (spectral) convolutions instead of the pixel-based NNhealpix,
and adds a fourth benchmark: joint tensor-to-scalar ratio r / optical depth τ
estimation relevant to the Simons Observatory, benchmarked against Fisher
(Cramér–Rao) forecasts.

> **Pipeline versions.** Results labeled **v4** are current: healpy
> `synfast` spectra-ordering fix (`new=True`) for all polarization maps +
> exact-likelihood MCMC baselines, run on the v4 stack (torch 2.11+cu130,
> torch-harmonics 0.9.2, healpy 1.20.1). **All polarization results
> (Tests 2–4) from v1–v3 are invalid** — the generators passed spectra in
> diagonal order while healpy's default expects row order, putting the EE
> spectrum in the TE slot (integrity notice in
> [BENCHMARKS.md](BENCHMARKS.md)). Test 1 (scalar) is unaffected. v3
> fixed the CAMB `raw_cl=True` C_ℓ unit bug and τ-divergence (Huber
> loss); v1/v2 Test 3/4 numbers predate those fixes and are provenance
> only.

## Key Results

> **Status (2026-10-08): v4 results are FINAL.** All numbers below now come
> from the corrected pipeline (`results_v4/`, Popeye A100 + CPU). Summary:
> Test 1 CNN 1.78/3.46/6.62/10.98% (MCMC 0.66/2.43/4.41/7.20% — MCMC
> dominates); Test 2 CNN parity at full sky (2.61/2.56% vs NNhealpix 2.7%)
> and −23…−31% better at partial sky, where the naive MCMC collapses
> (13–25% medians); Test 3 CNN 3.73% vs MCMC 3.07% and NNhealpix 4.0%.
> See [BENCHMARKS.md](BENCHMARKS.md).

### Test 2 (Polarization peaks) — v4 final: parity at full sky, clear win at partial sky

Mean error on (ℓ_Ep, ℓ_Bp) from Q/U maps, NSIDE=16, corrected data:

| f_sky | SpectralCNN v4 (ℓ_Ep / ℓ_Bp) | NNhealpix | Δ vs NNhealpix |
|-------|---------------------------|-----------|----------------|
| 1.0   | 2.61% / 2.56%             | 2.7%      | −4% (parity)   |
| 0.5   | **2.92% / 2.90%**         | 3.9%      | −25%           |
| 0.2   | **4.07% / 3.90%**         | 5.3%      | −25%           |
| 0.1   | **5.02% / 4.78%**         | 6.4%      | −23%           |
| 0.05  | **5.83% / 5.63%**         | 8.4%      | −31%           |

The advantage at partial sky is real and grows as the sky fraction
shrinks; at full sky the two architectures are at parity. The old
"−40% at full sky" claim was an artifact of the synfast bug.

### Test 1 (Scalar maps) — v4 final

| σ_n (μK) | SpectralCNN v4 | NNhealpix | MCMC v4 |
|-----|-------------|-----------|---------|
| 0   | 1.78%       | **1.3%**  | **0.66%** |
| 5   | 3.46%       | **2.9%**  | **2.43%** |
| 10  | 6.62%       | **5.2%**  | **4.41%** |
| 15  | 10.98%      | **8.4%**  | **7.20%** |

The global SHT spreads local noise across all modes; pixel-space pooling
filters it. The exact-likelihood MCMC dominates everywhere.

### Test 3 (τ estimation) — v4 final

| Method | τ error |
|--------|---------|
| **MCMC v4** | **3.07% mean / 2.61% median** |
| MCMC (KT19) | 2.8% |
| SpectralCNN v4 | 3.73% |
| NNhealpix (KT19) | 4.0% |

With honest data the CNN sits between the spectrum-fit MCMC and
NNhealpix; the earlier "CNN beats the MCMC" number (2.18%, v3) was an
artifact of the synfast ordering bug.

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
  Controlled experiments eliminate all three simple explanations: a 16×
  capacity increase (26.5M → 409M params), a 4× spectral-library increase,
  and a linear-head ablation all leave the collapse unchanged — the linear
  head collapses to exactly E[r] of the training prior, proving the learned
  representation carries no r information. Side finding: τ error improves
  3× (16–23% → 6.1%) when the log-r loss term is removed (shared-trunk
  gradient interference).
- **τ keeps a real but shrunk response** at both resolutions
  (slope ≈ 0.3–0.4 across τ ∈ [0.04, 0.07]).

Full fiducial-point RMSE vs Fisher tables and the multi-fiducial response
data are in [BENCHMARKS.md](BENCHMARKS.md).

**Main finding:** spectral convolution is the right inductive bias for CMB
polarization (Tests 2–4 at low resolution), while noisy scalar maps favor
pixel-space convolution (Test 1). Scaling map-based r/τ inference to high
resolution fails at the representation level under scalar point-estimate
regression — not for lack of capacity, data diversity, or a better output
transform; distributional objectives and harmonic-space readouts are the
indicated remedies.

See [BENCHMARKS.md](BENCHMARKS.md) for all numbers,
[ARCHITECTURE.md](ARCHITECTURE.md) for architecture details, and the paper
repository ([torch-harmonics-healpix-paper](https://github.com/zonca/torch-harmonics-healpix-paper))
for the manuscript.

## Pre-trained Models

Trained model weights are on Hugging Face:
<https://huggingface.co/zonca/torch-harmonics-healpix>

> **v3 checkpoints on HF:** `models/test3_v3.pt` (τ 2.18%) and the four
> calibrated NSIDE=16 Test 4 models `models/test4_v3_nside16_*.pt`
> (hidden_channels=32, num_blocks=3). The old v2 Test 3 weight
> (`test3_v2_fix.pt` on HF) and v2 Test 4 weights (in this repo's
> `results/test4_fsky*.pt`) predate the C_ℓ fix and are kept only to
> reproduce the superseded v2 numbers. Higher-resolution v3 models
> are deliberately not published (collapsed r response — see BENCHMARKS).
> Test 1/2 weights are unaffected by the fix.

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

# v4 stack (torch-harmonics 0.9.2 requires torch >=2.11)
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install torch-harmonics==0.9.2
uv pip install healpy astropy scipy h5py numpy camb pytest
uv pip install -e .
```

> The old `torch-harmonics==0.8.0 --no-deps` pin (V100 ABI issue) is
> obsolete on the v4 stack; 0.9.2 is installed normally. cu130 wheels
> require sm_80+ (A100) — for V100 use the cu124 build with 0.8.0.

## Project Structure

```
torch-harmonics-healpix/
├── src/torch_harmonics_healpix/
│   ├── healpix_resample.py          # HEALPix ↔ equiangular resampling
│   ├── data_generation.py           # Test 1: scalar power-spectrum maps
│   ├── data_generation_test2.py     # Test 2: polarization Q/U maps + masks
│   ├── data_generation_test3.py     # Test 3: CAMB τ spectra (raw_cl=True)
│   ├── data_generation_test4.py     # Test 4: CAMB r/τ spectra (raw_cl=True)
│   ├── mcmc_baseline.py             # exact-likelihood spectrum fit (Test 1)
│   ├── mcmc_baselines_test2_3.py    # baselines for Tests 2 & 3 (exact likelihood)
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
├── results_v3/                      # v3 results (superseded for Tests 2-4)
├── results_v4/                      # v4 (synfast fix + exact-likelihood) — current
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

- **Popeye (Slurm, account `soap`):** CPU jobs on `gen` (MCMC baselines,
  Fisher, CAMB caches); GPU jobs on `gpupreempt` (A100-40GB on
  `pcn-16-06`; preemptible — the v4 scripts checkpoint every epoch).
  v4 jobs: `sbatch slurm/run_mcmc_v4_popeye.slurm`,
  `sbatch slurm/train_v4_popeye.slurm`. Venv:
  `~/torch-hh-v4-venv` (build with `scripts/setup_popeye_venv_v4.sh`).
- **Expanse (GPU, Slurm):** legacy v2/v3 training jobs.
  `sbatch slurm/eval_fiducial_v3_expanse.slurm`, `sbatch slurm/train_test3_v3_expanse.slurm`, ...

Workflow: `git commit && git push` locally, `git pull` on the cluster, submit
from the repo clone. See [AGENTS.md](AGENTS.md) for accounts, environments,
and the full list of operational pitfalls (Lustre striping, HDF5 locking,
CAMB conventions, τ-divergence, shared masks).

## References

1. Krachmalnicoff & Tomasi (2019), "Convolutional neural networks on the HEALPix sphere", A&A 628, A129. [arXiv:1902.04083](https://arxiv.org/abs/1902.04083)
2. Bonev et al. (2023), "Spherical Fourier Neural Operators", ICML. [arXiv:2306.03838](https://arxiv.org/abs/2306.03838)
3. [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) — differentiable spherical harmonics in PyTorch
4. [NNhealpix](https://github.com/ai4cmb/NNhealpix) — pixel-space CNNs on HEALPix
