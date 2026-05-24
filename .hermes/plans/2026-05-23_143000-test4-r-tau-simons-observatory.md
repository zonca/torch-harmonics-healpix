# Plan: Test 4 — Joint r + τ Estimation from Q/U Maps

## Goal

Test whether SpectralCNN can jointly estimate r (tensor-to-scalar ratio) and τ (optical depth) from polarization maps. This is Simons Observatory's core science problem and directly extends our Test 3 (τ-only) to 2-parameter estimation with primordial B-modes.

## Why It's Interesting

- **SO's #1 goal**: constrain r from B-mode polarization
- **r–τ degeneracy**: these two parameters are correlated in BB/EE spectra — can the CNN break it?
- **SpectralCNN advantage**: the degeneracy lives in harmonic space where our network operates
- **Novel vs paper**: original tests all use r=0; we add primordial B-modes

## Design (Simple)

### Data
- **r ∈ [0, 0.01]**, log(r + 1e-4) as target (handles r=0 boundary)
- **τ ∈ [0.03, 0.08]** (same as Test 3)
- Other params: Planck 2018 best-fit, fixed
- Nside=16, ℓ_max=47 (same as T1-T3)
- Pre-compute 5000 CAMB spectra, share across datasets
- White noise only: σ ∈ {0, 6} μK-arcmin (noise-free + SO SAT 145 GHz goal)
- f_sky ∈ {1.0, 0.1} (full sky + SO SAT deep field)

### Model
- **Single-frequency**: 3 input channels [Q, U, mask], same as T2/T3
- **2 outputs**: [log(r + 1e-4), τ]
- Same SpectralCNN architecture (in_channels=3, out_channels=2)
- Inpainting for f_sky < 1.0

### Training
- Adapt from `train_test3_v2.py`
- Save `.pt` model weights (like T1-T3)
- Evaluation: r % error (r > 0.001), r bias (r ≈ 0), τ % error

### Baselines
- MCMC: χ² grid search over (r, τ) using C_ℓ^EE + C_ℓ^BB
- Fisher forecast: theoretical lower bound on σ(r), σ(τ)

### Configs (4 total)

| # | f_sky | σ [μK-arcmin] | Purpose |
|---|-------|---------------|---------|
| 1 | 1.0 | 0 | Noise-free baseline |
| 2 | 1.0 | 6 | SO noise, full sky |
| 3 | 0.1 | 0 | Partial sky, clean |
| 4 | 0.1 | 6 | SO-realistic (main result) |

## Implementation

### New Files

| File | Description |
|------|-------------|
| `src/torch_harmonics_healpix/data_generation_test4.py` | CAMB r/τ spectra, map generation, RTauDataset |
| `scripts/train_test4.py` | Training script (saves .pt + .json) |
| `scripts/fisher_forecast.py` | Fisher matrix → σ(r), σ(τ) lower bounds |
| `slurm/run_test4_expanse.slurm` | GPU training (all 4 configs) |
| `slurm/run_test4_mcmc_popeye.slurm` | MCMC baselines on CPU |

### Modified Files

| File | Change |
|------|--------|
| `mcmc_baselines_test2_3.py` | Add `mcmc_baseline_r_tau()` |
| `BENCHMARKS.md` | Add Test 4 results |
| `ARCHITECTURE.md` | Add 2-output design notes |
| `README.md` | Add SO context + Test 4 |
| `HF_MODEL_CARD.md` | Add T4 models |

## Benchmarking & Comparison

### No paper baseline — our own baselines:

| Method | Metric | What it tests |
|--------|--------|---------------|
| Fisher forecast | σ(r), σ(τ) lower bounds | Theoretical optimal power-spectrum estimator — CNN cannot beat this |
| **SpectralCNN** | σ(r), σ(τ) | Our method |
| MCMC (C_ℓ fit) | σ(r), σ(τ) | Traditional power-spectrum approach (same as T1-T3 baselines) |

### Metrics per config:
- **r % error**: mean |r_pred - r_true| / r_true (for r > 0.001); r bias (for r ≈ 0)
- **τ % error**: same as Test 3
- **2D error ellipse area**: captures r–τ degeneracy breaking (compare to Fisher ellipse)

### Narrative:
- "SpectralCNN approaches Fisher-optimal at full sky and outperforms MCMC at partial sky" — or whatever the data shows
- No NNhealpix comparison (paper doesn't test r estimation)

## Validation

- CNN error ≥ Fisher lower bound (can't beat theory)
- CNN τ error ≈ Test 3 at f_sky=1.0 (consistency check)
- r=0 maps → r_pred ≈ 0 (no false positives)
- Partial-sky advantage matches Test 2 trend
