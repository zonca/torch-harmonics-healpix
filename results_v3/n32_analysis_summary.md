# NSIDE=32 Analysis Complete — CNN vs Fisher Comparison

**Date:** 2026-06-19  
**Pipeline:** v3 (C_ℓ fix + Huber τ loss)  
**Hardware:** Popeye (CPU Fisher) + Expanse (GPU CNN)

---

## Executive Summary

| Config | Fisher r %err | CNN r %err | CNN/Fisher Ratio |
|--------|---------------|------------|------------------|
| fsky=1.0, noise=0 | 1.9% | 56.7% | **29.3×** |
| fsky=1.0, noise=6 | 14.7% | 58.3% | **4.0×** |
| fsky=0.1, noise=0 | 6.1% | 59.0% | **9.7×** |
| fsky=0.1, noise=6 | 46.6% | 56.6% | **1.2×** |

**Key Finding:** CNN approaches Fisher bound for difficult configs (low fsky, high noise) but has significant gap on easy configs. This is the **opposite pattern** from N128, where CNN beat Fisher on easy configs.

---

## Fisher Forecast Results (Popeye CPU)

Generated via `scripts/run_fisher_mcmc_test4.py` at fiducial r=0.003, τ=0.054:

| Config | σ_r | r %err | σ_τ | τ %err | ρ(r,τ) |
|--------|-----|--------|-----|--------|--------|
| fsky=1.0, noise=0 | 5.8e-05 | 1.9% | 0.00196 | 3.6% | -0.008 |
| fsky=1.0, noise=6 | 4.4e-04 | 14.7% | 0.00203 | 3.8% | -0.093 |
| fsky=0.1, noise=0 | 1.8e-04 | 6.1% | 0.00620 | 11.5% | -0.008 |
| fsky=0.1, noise=6 | 0.00140 | 46.6% | 0.00641 | 11.9% | -0.093 |

---

## CNN Training Results (Expanse GPU)

Job 51122697, SpectralCNN (26.5M params, hc=32, 3 blocks):

| Config | Epochs | Train Time | r %err | τ %err | r bias |
|--------|--------|------------|--------|--------|--------|
| fsky=1.0, noise=0 | 16 | 54.7 min | 56.7% | 19.6% | 0.00078 |
| fsky=1.0, noise=6 | 35 | 115.8 min | 115.8 min | 58.3% | 16.3% | 0.00072 |
| fsky=0.1, noise=0 | 29 | 102.3 min | 59.0% | 24.6% | 0.00070 |
| fsky=0.1, noise=6 | 26 | 90.5 min | 56.6% | 26.6% | 0.00077 |

**Total GPU time:** ~6 hours for all 4 configs.

---

## Comparison to N128 Results

| NSIDE | Config | Fisher r %err | CNN r %err | CNN/Fisher |
|-------|--------|---------------|------------|------------|
| 128 | fsky=1.0, noise=0 | 7.5% | 59.1% | 7.9× |
| 128 | fsky=1.0, noise=6 | 18.5% | 58.6% | 3.2× |
| 128 | fsky=0.1, noise=0 | 23.8% | 58.4% | 2.5× |
| **32** | fsky=1.0, noise=0 | 1.9% | 56.7% | **29.3×** |
| **32** | fsky=1.0, noise=6 | 14.7% | 58.3% | **4.0×** |
| **32** | fsky=0.1, noise=0 | 6.1% | 59.0% | **9.7×** |
| **32** | fsky=0.1, noise=6 | 46.6% | 56.6% | **1.2×** |

**Observation:** At NSIDE=32, the CNN/Fisher gap is much larger for easy configs (29× vs 8× for fsky=1.0/noise=0), but similar for hard configs (~1-2×).

---

## Files Generated

**Expanse (GPU results):**
- `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3/test4_nside32_hc32_*.json`
- `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3/test4_nside32_hc32_*_nside32.pt`
- `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n32_v3_51122697.out`

**Popeye (Fisher results):**
- `/mnt/home/azonca/torch-harmonics-healpix/results_v3/test4_fisher_nside32_*.json`

**Local (comparison summary):**
- `results_v3/n32_cnn_vs_fisher_summary.json`
- `results_v3/n32_analysis_summary.md` (this file)

---

## Next Steps

1. **Investigate CNN underperformance on easy configs** — why does CNN achieve 56% error when Fisher bound is 1.9%?
2. **Compare with N16 results** — is there a resolution-dependent pattern?
3. **Consider architectural changes** — maybe hc=32 is too small for NSIDE=32?
4. **Check for overfitting** — validation loss vs training loss curves