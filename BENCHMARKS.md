# Benchmarks: torch-harmonics-healpix vs Krachmalnicoff & Tomasi (2019)

Reproduction of the three CMB benchmarks of
[Krachmalnicoff & Tomasi 2019, A&A 628, A129](https://arxiv.org/abs/1902.04083)
(KT19) with spectral CNNs, benchmarked against exact-likelihood MCMC
baselines and KT19's published NNhealpix numbers.

**Pipeline versions**

| Version | Status | Notes |
|---------|--------|-------|
| v4      | **current** | synfast `new=True` spectra-ordering fix (Tests 2–4 data regenerated), exact-likelihood MCMC baselines matching KT19 |
| v3      | superseded (Tests 2–4) | CAMB `raw_cl=True` (C_ℓ fix), Huber τ loss, Fisher at fixed `lmax_calc=500` — **polarization maps still carried the synfast ordering bug** |
| v2      | superseded | Test 3/4 maps generated with D_ℓ amplitudes (unit bug); Tests 1–2 unaffected and still current |
| v1      | superseded | early runs, per-split masks bug |

All v1/v2 JSONs remain in `results/` for provenance; v3 results are in
`results_v3/`; v4 results land in `results_v4/`.

**⚠ v4 data-integrity notice (2026-10).** Every polarization map generated
before v4 (Tests 2, 3, 4 — CNN training *and* MCMC evaluation) is
statistically wrong: `hp.synfast` was called with 6 spectra in diagonal
order `[TT,EE,BB,TE,EB,TB]` but healpy's default (`new=False`) expects row
order `[TT,TE,TB,EE,EB,BB]`, so the EE spectrum landed in the TE slot (BB in
TB) and the clipped Cholesky produced maps with observed C_EE ≈ C_EE²/C_TT
and Q/U as rescaled copies of a single T realization instead of independent
E/B fields. **All pre-v4 SpectralCNN results for Tests 2–3 are not
comparable to KT19** (the old Test 2/3 tables are kept below for provenance
only). Test 1 (scalar) is unaffected. Fixed in commit 6331690. The **v4b
campaign** (100k maps = KT19 budget, 3 seeds per config, A100) is re-running
every config on the corrected data (`slurm/train_v4b_popeye.slurm`); the
20k-map v4 numbers below are interim.

---

## Hardware & Runtime

### Popeye (SDSC/Flatiron — CPU + GPU)

- CPU: Intel Xeon Platinum 8168, 48 cores; partition `gen`, account `soap`.
  Fisher forecast: seconds per config. v4 MCMC baselines: ~0.05–0.06 s/map
  (1000 maps ≈ 1 min per config).
- GPU: partition `gpupreempt` (preemptible), account `soap` —
  `pcn-16-06` 6× A100-PCIE-40GB, `pcn-16-01/03/04/05` V100S-32GB.
  **torch 2.11+cu130 runs on the A100s only** (sm_80; V100 sm_70 unsupported
  by cu130).
- v4 stack: Python 3.11 (module), torch 2.11.0+cu130, torch-harmonics
  0.9.2 (the ≥0.9 ABI break does not affect A100/sm_80), healpy 1.20.1,
  numpy/scipy current, camb. Venv: `~/torch-hh-v4-venv`
  (`scripts/setup_popeye_venv_v4.sh`).
- CAMB spectral grids: cached at
  `/mnt/sdceph/users/zonca/torch-harmonics-healpix/camb_cache`
  (`THH_CAMB_CACHE_DIR`), shared between CPU and GPU jobs.

### Expanse (GPU — superseded for v4)

- NVIDIA Tesla V100-SXM2-32GB, CUDA 12.0; Python 3.11, PyTorch 2.6.0+cu124,
  torch-harmonics 0.8.0 (V100 ABI pin), healpy 1.19.0
- Test 1–3 training (100k maps): ~70 min/config
- Test 4 NSIDE=32 training: ~1–2 h/config; NSIDE=128 (422M params, HDF5 on
  striped Lustre): ~60 min/epoch (I/O-dominated)

---

## Test 1: ℓ_p from scalar (T) maps — NSIDE=16

Estimate the peak multipole of C_ℓ = exp(−(ℓ−ℓ_p)²/2σ_p²)+10⁻⁵, σ_p=5,
ℓ_p ∈ [5, 20]. 100k train / 10k val / 1k test, four noise levels.
No CAMB, scalar maps → **unaffected by the v4 polarization fix**; v2 CNN
numbers remain current. The MCMC baseline was rebuilt in v4.

### v4 MCMC baseline (exact Gaussian likelihood)

The v4 baseline replaces the χ² fit with the exact Wishart likelihood over
ℓ ≤ 2·N_side. **Production numbers** (1000 maps per config, seed 42,
Popeye job 2483914, `results_v4/mcmc_test1_v4.json`):

| σ_n (μK) | MCMC v4 | MCMC (KT19) | χ² fit (v2, superseded) |
|----------|---------|-------------|--------------------------|
| 0        | **0.66%** | 0.7%      | 2.22%                    |
| 5        | **2.43%** | 2.5%      | 2.87%                    |
| 10       | **4.41%** | 4.8%      | 5.18%                    |
| 15       | **7.20%** | 7.8%      | 8.24%                    |

The v4 baseline reproduces — or slightly beats — KT19's MCMC at every
noise level (medians are lower still: 0.54/1.62/3.13/5.07%). Design
notes: **no pixel window in the model** — healpy ≥1.15 `synfast` defaults
to `pixwin=False` (verified by a flat-floor test; convolving or
deconvolving biases the fit badly, 0.65%→6.5%); the fit range stops at
2·N_side because the 2·N_side < ℓ ≤ 3·N_side−1 band of the pseudo-C_ℓ is
aliasing-dominated.

### SpectralCNN vs baselines (v4, FINAL — 20k maps, corrected data, new stack)

| σ_n (μK) | SpectralCNN v4 | SpectralCNN v2 (100k) | NNhealpix (KT19) | MCMC v4 |
|----------|----------------|------------------------|------------------|---------|
| 0        | 1.78%          | 1.27%                  | **1.3%**         | **0.66%** |
| 5        | 3.46%          | 3.58%                  | **2.9%**         | **2.43%** |
| 10       | 6.62%          | 6.81%                  | **5.2%**         | **4.41%** |
| 15       | 10.98%         | 11.98%                 | **8.4%**         | **7.20%** |

Test 1 is scalar (unaffected by the polarization bug): the v4 retrain on
20k maps with the new stack (torch 2.11/torch-harmonics 0.9.2) matches or
beats the 100k-map v2 numbers at every noisy level. The MCMC baseline
dominates everywhere; NNhealpix beats the CNN at all σ_n>0. The SHT
spreads white noise into every (ℓ, m) mode; pixel-space pooling
low-passes it — parity without noise, growing deficit with noise.

## Test 2: ℓ_Ep/ℓ_Bp from Q/U maps — NSIDE=16

3 input channels (Q, U, mask), 2 outputs. **Shared mask** across splits
(mandatory — see ARCHITECTURE.md), zeros masking outside the cap
(KT19 methodology).

**⚠ v4: all pre-v4 CNN numbers below are INVALID** — the maps were
generated with the synfast ordering bug (EE in the TE slot), so the old
"SpectralCNN beats KT19 by 40–64%" tables are provenance only. The v4
MCMC baseline is trustworthy; the v4 CNN retrain is in flight.

### v4 MCMC baseline (exact Gaussian likelihood, corrected data)

**Production numbers** (1000 maps per f_sky, seed 42, Popeye job 2483914,
`results_v4/mcmc_test2_v4.json`). Means are heavy-tailed at partial sky —
medians shown alongside:

| f_sky | ℓ_Ep mean | ℓ_Bp mean | ℓ_Ep med | ℓ_Bp med | note |
|-------|-----------|-----------|----------|----------|------|
| 1.0   | **0.64%** | **0.65%** | 0.49%    | 0.55%    | matches KT19 full-sky MCMC ~0.7% |
| 0.5   | 31.9%     | 29.9%     | 13.3%    | 13.1%    | E/B leakage from the mask, uncorrected |
| 0.2   | 38.4%     | 36.4%     | 18.7%    | 18.9%    | same |
| 0.1   | 42.3%     | 40.0%     | 21.7%    | 22.3%    | same |
| 0.05  | 45.8%     | 43.5%     | 25.4%    | 24.8%    | same |

Design notes: polarization fits run over ℓ ∈ [2, 2·N_side] — `anafast`
returns exact zeros for E/B at ℓ=0,1 (a log(0) hazard for the exact
likelihood) and the ℓ>2·N_side band is aliasing noise. For partial sky the
f_sky pseudo-C_ℓ correction (C_obs/f_sky, modes (2ℓ+1)·f_sky) is applied
but **E/B mixing from the mask is NOT undone** — that needs a full
pseudo-C_ℓ framework (e.g. NaMaster). KT19 report no partial-sky MCMC
number; the strong degradation of the naive estimator at f_sky≤0.5 is
itself the paper's motivation for pixel-space methods, and is exactly
where a CNN operating on pixels should be compared.

### v4 SpectralCNN results (FINAL — corrected data, 20k maps, new stack)

Popeye job 2483915, `results_v4/test2_v4_fsky*.json`. First
polarization-CNN results on data with the correct E/B content:

| f_sky | SpectralCNN v4 (ℓ_Ep / ℓ_Bp) | NNhealpix (KT19) | Δ mean vs KT19 |
|-------|------------------------------|------------------|----------------|
| 1.0   | 2.61% / 2.56%                | 2.7%             | −4% (parity)   |
| 0.5   | **2.92% / 2.90%**            | 3.9%             | −25%           |
| 0.2   | **4.07% / 3.90%**            | 5.3%             | −25%           |
| 0.1   | **5.02% / 4.78%**            | 6.4%             | −23%           |
| 0.05  | **5.83% / 5.63%**            | 8.4%             | −31%           |

**The corrected story**: full-sky parity with the pixel-space network
(the old −40% was an artifact of the synfast bug — E-power arriving
through the TE slot made the task easier), but a **genuine, growing
advantage at partial sky: −23…−31%**, and the naive spectrum-fit MCMC
collapses there (medians 13–25%, means 30–46%) because of uncorrected
E/B leakage. Pixel-space inpainting + the CNN's spectral prior is the
right combination for masked polarization — exactly the regime KT19
motivate their pixel-based method for, and where the map-based CNN now
beats both published baselines.

### Superseded: v2 SpectralCNN vs KT19 (invalid data — provenance only)

| f_sky | SpectralCNN (ℓ_Ep/ℓ_Bp) | NNhealpix (KT19) | Δ mean vs KT19 |
|-------|--------------------------|------------------|----------------|
| 1.0   | **1.69% / 1.53%**        | 2.7%             | −40%           |
| 0.5   | **1.95% / 1.91%**        | 3.9%             | −51%           |
| 0.2   | **2.15% / 2.17%**        | 5.3%             | −59%           |
| 0.1   | **2.56% / 2.70%**        | 6.4%             | −59%           |
| 0.05  | **3.01% / 3.11%**        | 8.4%             | −64%           |

KT19's accuracy scales as f_sky^−0.36. (Numbers generated from distorted
maps — see the v4 integrity notice above.)

## Test 3: τ from Q/U maps — NSIDE=16

CAMB EE spectra, τ ∈ [0.03, 0.08], full sky, noiseless.

**⚠ v4: pre-v4 CNN numbers are INVALID** (synfast ordering bug — see the
integrity notice). The v2/v3 CNN rows below are provenance only. Two
additional v4 fixes affect this test: the MCMC's τ labels now come from
the same grid spectrum that generated each map (previously drawn
independently — pure label noise), and the 5000-spectrum CAMB grid is
disk-cached + process-parallel (`~/.cache/thh_camb`, `THH_CAMB_CACHE_DIR`).

### v4 MCMC baseline (exact Gaussian likelihood, corrected labels)

**Production numbers** (1000 maps, full 5000-template grid, seed 42,
Popeye job 2483914, `results_v4/mcmc_test3_v4.json`):

| Method | τ error | note |
|--------|---------|------|
| **MCMC v4** | **3.07% mean / 2.61% median** | 5000-template grid, ~4 ms/map |
| MCMC (KT19) | 2.8% | paper value |
| SpectralCNN v4 | 3.73% | corrected data, 20k maps, 34 epochs |
| NNhealpix (KT19) | 4.0% | paper value |

**v4 verdict**: with honest data the CNN no longer beats the spectrum-fit
MCMC (3.73% vs 3.07%); it sits between the MCMC and NNhealpix. The v3
"CNN wins" number (2.18%) was measured on maps whose E-power arrived
through the TE slot — an easier, physically wrong problem.

### Superseded: v2/v3 SpectralCNN (invalid data — provenance only)

| Method | τ error | Pipeline |
|--------|---------|----------|
| SpectralCNN (superseded) | 2.18% (2.23 ± 0.06% over 3 seeds) | v3 (NRP GPU, 2026-07-16) |
| SpectralCNN (superseded) | 3.76% | v2 (D_ℓ bug) |

The v3 "CNN beats the MCMC" reading does not survive the v4 data fix: those
maps had E-power injected through the TE slot, a different (and likely
easier) statistical problem. The v4 retrain
(`slurm/train_v4_popeye.slurm`) settles the comparison on corrected data.

---

## Test 4 (joint r/τ) — REMOVED from the suite (2026-10-10)

Test 4 was an exploratory extension (joint r/τ estimation from Q/U maps
at NSIDE 16/32/128, benchmarked against Fisher Cramér–Rao bounds). It is
**out of scope for this project's benchmark suite**: its v1–v3 results
were produced with the pre-v4 synfast spectra-ordering bug (like all
polarization data before 2026-10), and regenerating it was deliberately
deferred. The code, result JSONs, and analysis remain in the repository
for provenance:

- Data/models: `src/torch_harmonics_healpix/data_generation_test4.py`,
  `scripts/train_test4.py`, `scripts/run_fisher_mcmc_test4.py`
- Results: `results/` (v2), `results_v3/` (v3, incl. the multi-fiducial
  collapse analysis), Fisher bounds `results_v3/fisher_multifid/`
- Historical note: the NSIDE≥32 r-channel collapse (r̂ constant ≈ 0.0011
  independent of input, robust to 16× capacity, 4× spectral diversity,
  and head parameterisation) remains an instructive negative result; see
  git history of this file for the full tables.

## Architecture comparison

| Property | SpectralCNN | NNhealpix |
|----------|-------------|-----------|
| Domain | spectral (ℓ, m) | pixel (HEALPix) |
| Parameters (T1 / T2-3) | 6.5M / 9.8M | ~0.08M / ~0.24M |
| Parameter scaling | O(ℓ²max·C²) | O(k²·C²) |
| Mask handling | mean-inpainting before SHT | natural (zeros) |
| Noise robustness | low (SHT spreads noise) | high (pooling low-pass) |
| Best for | polarization, global structure | noisy scalar maps |

## References

1. Krachmalnicoff & Tomasi (2019), A&A 628, A129, arXiv:1902.04083
2. Bonev et al. (2023), ICML, arXiv:2306.03838
3. Ocampo, Price & McEwen (2023), ICLR, arXiv:2209.13603
