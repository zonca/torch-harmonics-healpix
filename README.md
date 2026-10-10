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

### Test 4 (Joint r/τ) — removed from the suite

Test 4 (joint r/τ estimation from Q/U maps, Fisher-benchmarked) was an
exploratory extension. Its results predate the synfast spectra-ordering
fix, and it is **out of scope for the benchmark suite**: regenerating it
is deliberately deferred. Code and historical results stay in the repo
for provenance (`data_generation_test4.py`, `scripts/train_test4.py`,
`results_v3/` — including the instructive NSIDE≥32 r-channel collapse,
see git history). The paper covers Tests 1–3 only.

**Main finding:** spectral convolution is the right inductive bias for
CMB polarization on masked skies (Test 2), while noisy scalar maps favor
pixel-space convolution (Test 1) and the exact-likelihood MCMC remains
the reference at full sky (Test 3).

See [BENCHMARKS.md](BENCHMARKS.md) for all numbers,
[ARCHITECTURE.md](ARCHITECTURE.md) for architecture details, and the paper
repository ([torch-harmonics-healpix-paper](https://github.com/zonca/torch-harmonics-healpix-paper))
for the manuscript.

## Pre-trained Models

Trained model weights are on Hugging Face:
<https://huggingface.co/zonca/torch-harmonics-healpix>

> **v4 checkpoints (current):** `models/test2_v4_fsky*.pt`,
> `models/test3_v4.pt`, `models/test1_v4_noise*.pt` — trained on the
> corrected data (20k maps). The v4b campaign (100k maps, 3 seeds) will
> supersede these. Pre-v4 weights on HF (`models/test3_v3.pt`,
> `models/test4_v3_nside16_*.pt`) predate the synfast spectra-ordering
> fix and are kept for provenance only; Test 1 weights are unaffected.

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
│   ├── data_generation_test4.py     # (archived) Test 4 r/τ spectra — out of suite scope
│   ├── mcmc_baseline.py             # exact-likelihood spectrum fit (Test 1)
│   ├── mcmc_baselines_test2_3.py    # baselines for Tests 2 & 3 (exact likelihood)
│   └── models/
│       ├── spectral_cnn.py          # SpectralCNN (fixed ℓ_max)
│       └── multires_spectral_cnn.py # MultiResSpectralCNN (ablation)
├── scripts/
│   ├── train_test{1,2,3}_v2.py      # Tests 1–3 training
│   ├── train_test4.py               # (archived) Test 4 training — out of suite scope
│   ├── generate_test4_hdf5.py       # pre-generate NSIDE=128 datasets
│   ├── precompute_test3_camb.py     # parallel CAMB cache (CPU cluster)
│   ├── fisher_forecast.py           # Fisher bounds (fixed lmax_calc=500)
│   ├── run_fisher_mcmc_test4.py     # (archived) Test 4 Fisher + MH-MCMC
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
python scripts/train_test1_v3.py --noise_std 0 --n_train 100000 --skip_mcmc --output results_v4b/test1_noise0.json

# Test 2 (per sky fraction)
python scripts/train_test2_v2.py --f_sky 0.5 --output results/test2_fsky0.5.json

# Test 3 (uses corrected CAMB cache; ~2h of CAMB calls without one)
python scripts/precompute_test3_camb.py --output camb_cache_test3_v3.fits
python scripts/train_test3_v2.py --camb_cache camb_cache_test3_v3.fits \
    --output results_v3/test3_v3.json

# Multi-seed resumable training (v4b campaign pattern)
python scripts/train_test2_v2.py --f_sky 0.5 --n_train 100000 \
    --seed 42 --resume --checkpoint ckpt.pt --output results_v4b/test2_fsky0.5_seed42.json
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
