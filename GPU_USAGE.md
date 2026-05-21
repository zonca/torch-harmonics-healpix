# GPU Usage & Model Parameters

## GPU Usage Tally (Expanse, V100-SMX2-32GB, gpu-shared partition)

| Category | Jobs | GPU-hours | Notes |
|----------|------|-----------|-------|
| Setup & Debug | 10 | 0.31 | test-gpu, debug-sht, bash |
| torch-hh-test | 7 | 0.11 | SHT integration tests |
| T1 v1 Training | 6 | 2.41 | First attempt, many failures |
| T1 v2 Training | 1 | 4.55 | Paper-matching SpectralCNN v2 ✅ |
| T1 v3 Training | 3 | 4.21 | MultiResSpectralCNN (ablation) ✅ |
| T2/T3 No-inpaint | 1 | 8.18 | ❌ Different masks per dataset (bug) |
| T2/T3 Inpaint (bug) | 1 | 13.75 | ❌ Shared mask not fixed yet |
| T2/T3 Full fix re-run | 1 | 5.83+ | ✅ Running (job 49283390) |
| **TOTAL** | **30** | **39.35+** | |

### Wasted GPU Time

| Job | Hours | Reason |
|-----|-------|--------|
| T2/T3 No-inpaint (49224438) | 8.18 | Different masks per dataset — test eval meaningless |
| T2/T3 Inpaint (49229707) | 13.75 | Same mask bug + CAMB error; cancelled after 13.75h |
| **Total wasted** | **21.93** | ~56% of total GPU time |

### Estimated Final Total

- Current: 39.35h (with job 49283390 at 5.83h and running)
- Job 49283390 estimated total: ~20h
- **Estimated final total: ~53-54 GPU-hours**
- **Productive GPU-hours: ~31-32** (excluding wasted)

## Popeye CPU Usage

| Job | Purpose | Walltime | Status |
|-----|---------|----------|--------|
| 2426199 | MCMC Test 1 only (first attempt) | ~2 min | ✅ Done |
| 2426200 | MCMC Tests 1+2+3 (all) | ~10 min | ✅ Done |

## Model Parameters

| Architecture | Test 1 | Test 2 | Notes |
|-------------|--------|--------|-------|
| **SpectralCNN v2** | 6,454,529 | 9,829,634 | 3 spectral blocks, 32 channels |
| **MultiResSpectralCNN v3** | 1,545,601 | — | Multi-resolution ablation |
| **NNhealpix** (paper) | ~80,000 | ~240,000 | Pixel-space CNN, O(filter²) |
| **MCMC** (baseline) | 0 | 0 | Classical, no trainable params |

### Parameter Scaling

SpectralCNN parameters scale as O(ℓ_max² × n_channels²) per layer because spectral
weights are full (ℓ, m) matrices. NNhealpix scales as O(filter_size² × n_channels²).
This gives SpectralCNN a **40-80× parameter overhead** over NNhealpix.

Despite this, SpectralCNN excels at full-sky polarization (Test 2 f_sky=1.0:
1.5%/1.6% vs 2.7%/2.7%) but struggles with noise (Test 1 σ≥5) and partial sky
without inpainting. The spectral representation provides a strong prior for
clean, full-sky data but is sensitive to artifacts that corrupt high-ℓ modes.
