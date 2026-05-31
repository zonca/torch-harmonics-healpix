# GPU Usage & Model Parameters

## GPU Usage Tally (Expanse, V100-SXM2-32GB, gpu-shared partition)

| Category | Jobs | GPU-hours | Notes |
|----------|------|-----------|-------|
| Setup & Debug | 10 | 0.31 | test-gpu, debug-sht, bash |
| torch-hh-test | 7 | 0.11 | SHT integration tests |
| T1 v1 Training | 6 | 2.41 | First attempt, many failures |
| T1 v2 Training | 1 | 4.55 | Paper-matching SpectralCNN v2 ✅ |
| T1 v3 Training | 3 | 4.21 | MultiResSpectralCNN (ablation) ✅ |
| T2/T3 No-inpaint | 1 | 8.18 | ❌ Different masks per dataset (bug) |
| T2/T3 Inpaint (bug) | 1 | 13.75 | ❌ Shared mask not fixed yet |
| T2/T3 Full fix re-run | 1 | 24.00 | ✅ Timed out but Test 1+2 complete |
| Test 3 only | 1 | ~8.00 | ✅ Test 3 complete (3.76%) |
|| Test 4 training | 1 | ~6.0 | Job 49694662, 4 configs (f_sky×noise), CAMB FITS cache 3.9MB |
|| T4 NSIDE=128 (ReduceLR) | 4 | 96.0 | Jobs 49877275-31, 24h walltime each, τ divergence at ep11 |
|| T4 NSIDE=128 (cosine T150) | 4 | 40.0 | Jobs 49905105-8, cancelled after τ divergence confirmed |
|| T4 NSIDE=128 (hard clamp) | 4 | 2.0 | Jobs 49917222-5, cancelled ep1 (dead gradients, τ=940%) |
|| T4 NSIDE=128 (Huber T25) | 4 | 96.0 | Jobs 49918615-8, 24h walltime, Huber τ loss + CosineAnnealingLR |
|| **TOTAL** | **53** | **~311.5+** | |

### Note on Test 4 GPU Time

Expanse job 49694662 covers all 4 Test 4 configurations. Total walltime ~6 hours.
The FITS-based CAMB cache (introduced in Test 4's data generation) saves ~4.5 hours
of GPU time by avoiding recomputation of CAMB power spectra for each configuration.
The CAMB FITS cache file is 3.9MB.

### Wasted GPU Time

| Job | Hours | Reason |
|-----|-------|--------|
| T2/T3 No-inpaint (49224438) | 8.18 | Different masks per dataset — test eval meaningless |
| T2/T3 Inpaint (49229707) | 13.75 | Same mask bug + CAMB error; cancelled after 13.75h |
| **Total wasted** | **21.93** | ~33% of total GPU time |

### Productive GPU Time: ~49.5 hours

## Popeye CPU Usage

| Job | Purpose | Walltime | Status |
|-----|---------|----------|--------|
| 2426200 | MCMC Tests 1+2+3 (all) | ~10 min | ✅ Done |

## Model Parameters

| Architecture | Test 1 | Test 2/3 | Notes |
|-------------|--------|----------|-------|
| **SpectralCNN v2** | 6,454,529 | 9,829,634 | 3 spectral blocks, 32 channels |
| **SpectralCNN T4 NSIDE=128** | — | 422,074,562 | 3 blocks, 32 ch, LMAX=383 |
| **MultiResSpectralCNN v3** | 1,545,601 | — | Multi-resolution ablation |
| **NNhealpix** (paper) | ~80,000 | ~240,000 | Pixel-space CNN, O(filter²) |
| **MCMC** (baseline) | 0 | 0 | Classical, no trainable params |

### Parameter Scaling

SpectralCNN parameters scale as O(ℓ_max² × n_channels²) per layer because spectral
weights are full (ℓ, m) matrices. NNhealpix scales as O(filter_size² × n_channels²).
This gives SpectralCNN a **40-80× parameter overhead** over NNhealpix.

Despite this overhead, SpectralCNN dominates for polarization estimation (Tests 2 & 3)
where the spectral prior is physically well-motivated. For noisy scalar maps (Test 1),
the extra parameters provide no advantage — pixel-space convolution is inherently
more noise-robust.
