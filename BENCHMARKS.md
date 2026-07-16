# Benchmarks: torch-harmonics-healpix vs Krachmalnicoff & Tomasi (2019)

Reproduction of the three CMB benchmarks of
[Krachmalnicoff & Tomasi 2019, A&A 628, A129](https://arxiv.org/abs/1902.04083)
(KT19) with spectral CNNs, plus the new joint r/τ benchmark (Test 4)
compared against Fisher (Cramér–Rao) bounds.

**Pipeline versions**

| Version | Status | Notes |
|---------|--------|-------|
| v3      | **current** | CAMB `raw_cl=True` (C_ℓ fix), Huber τ loss, Fisher at fixed `lmax_calc=500` |
| v2      | superseded | Test 3/4 maps generated with D_ℓ amplitudes (unit bug); Tests 1–2 unaffected and still current |
| v1      | superseded | early runs, per-split masks bug |

All v1/v2 JSONs remain in `results/` for provenance; v3 results are in
`results_v3/`.

---

## Hardware & Runtime

### Expanse (GPU — SDSC, allocation sds166)

- NVIDIA Tesla V100-SXM2-32GB, CUDA 12.0; Python 3.11, PyTorch 2.6.0+cu124,
  torch-harmonics 0.8.0, healpy 1.19.0
- Test 1–3 training (100k maps): ~70 min/config
- Test 4 NSIDE=32 training: ~1–2 h/config; NSIDE=128 (422M params, HDF5 on
  striped Lustre): ~60 min/epoch (I/O-dominated)

### Popeye (CPU — SDSC/Flatiron)

- Intel Xeon Platinum 8168, 48 cores; Python 3.11 via module + `~/torch-hh-venv`
- Fisher forecast: seconds per config; 5000-spectra CAMB cache: minutes
  (48-way parallel, `scripts/precompute_test3_camb.py`)

---

## Test 1: ℓ_p from scalar (T) maps — NSIDE=16

Estimate the peak multipole of C_ℓ = exp(−(ℓ−ℓ_p)²/2σ_p²)+10⁻⁵, σ_p=5,
ℓ_p ∈ [5, 20]. 100k train / 10k val / 1k test, four noise levels.
No CAMB → v2 results current.

| σ_n (μK) | SpectralCNN | NNhealpix (KT19) | MCMC (KT19) | χ² fit (ours) |
|----------|-------------|------------------|-------------|---------------|
| 0        | **1.27%**   | 1.3%             | 0.7%        | 2.22%         |
| 5        | 3.58%       | **2.9%**         | 2.5%        | 2.87%         |
| 10       | 6.81%       | **5.2%**         | 4.8%        | 5.18%         |
| 15       | 11.98%      | **8.4%**         | 7.8%        | 8.24%         |

The SHT spreads white noise into every (ℓ, m) mode; pixel-space pooling
low-passes it. Parity without noise, growing deficit with noise.

## Test 2: ℓ_Ep/ℓ_Bp from Q/U maps — NSIDE=16

3 input channels (Q, U, mask), 2 outputs. **Shared mask** across splits
(mandatory — see ARCHITECTURE.md), mean-inpainting for f_sky<1.
No CAMB → v2 results current.

| f_sky | SpectralCNN (ℓ_Ep/ℓ_Bp) | NNhealpix (KT19) | Δ mean vs KT19 |
|-------|--------------------------|------------------|----------------|
| 1.0   | **1.69% / 1.53%**        | 2.7%             | −40%           |
| 0.5   | **1.95% / 1.91%**        | 3.9%             | −51%           |
| 0.2   | **2.15% / 2.17%**        | 5.3%             | −59%           |
| 0.1   | **2.56% / 2.70%**        | 6.4%             | −59%           |
| 0.05  | **3.01% / 3.11%**        | 8.4%             | −64%           |

KT19's accuracy scales as f_sky^−0.36; the SpectralCNN degrades much more
slowly (1.6% → 3.1% over the same range). KT19's full-sky MCMC reference:
0.7%.

## Test 3: τ from Q/U maps — NSIDE=16

CAMB EE spectra, τ ∈ [0.03, 0.08], full sky, noiseless.

| Method | τ error | Pipeline |
|--------|---------|----------|
| **SpectralCNN** | **2.18%** | v3 (77 epochs, NRP GPU, 2026-07-16) |
| MCMC (KT19) | 2.8% | — |
| NNhealpix (KT19) | 4.0% | — |
| SpectralCNN (superseded) | 3.76% | v2 (D_ℓ bug — not comparable) |

The corrected pipeline nearly halves the CNN error relative to v2 and puts
the map-based spectral network ahead of the published spectrum fit
(single run; read as parity-or-better). Retrained on NRP
(`nrp/examples/train-test3-v3.yaml`) with the corrected CAMB cache
(`results_v3/camb_cache_test3_v3.fits`).

---

## Test 4: joint r/τ (Simons Observatory challenge) — v3

r ∈ [0, 0.01], τ ∈ [0.03, 0.08]; targets [log(r+10⁻⁴), τ];
loss MSE(log r) + Huber(τ, δ=0.01). Four configs:
f_sky ∈ {1.0, 0.1} × noise ∈ {0, 6} μK-arcmin.

### Fisher (Cramér–Rao) bounds — fixed lmax_calc=500

At fiducial (r=0.003, τ=0.054). **Bug fixed 2026-06:** CAMB's
`set_for_lmax()` changes internal accuracy with the requested lmax, which
made bounds non-monotonic across NSIDE; all spectra are now computed at
lmax=500 and truncated (`results_v3/fisher_fixed_lmax_verification.json`).

| NSIDE | f_sky | noise | σ(r) | σ(r)/r | σ(τ) | σ(τ)/τ |
|-------|-------|-------|----------|--------|---------|--------|
| 16  | 1.0 | 0 | 0.000346 | 11.5% | 0.00194 | 3.6%  |
| 16  | 1.0 | 6 | 0.000706 | 23.5% | 0.00205 | 3.8%  |
| 16  | 0.1 | 0 | 0.001095 | 36.5% | 0.00613 | 11.3% |
| 16  | 0.1 | 6 | 0.002232 | 74.4% | 0.00648 | 12.0% |
| 32  | 1.0 | 0 | 0.000259 | 8.6%  | 0.00185 | 3.4%  |
| 32  | 1.0 | 6 | 0.000589 | 19.6% | 0.00197 | 3.7%  |
| 32  | 0.1 | 0 | 0.000820 | 27.3% | 0.00586 | 10.9% |
| 32  | 0.1 | 6 | 0.001863 | 62.1% | 0.00624 | 11.6% |
| 128 | 1.0 | 0 | 0.000247 | 8.2%  | 0.00110 | 2.0%  |
| 128 | 1.0 | 6 | 0.000569 | 19.0% | 0.00134 | 2.5%  |
| 128 | 0.1 | 0 | 0.000781 | 26.0% | 0.00347 | 6.4%  |
| 128 | 0.1 | 6 | 0.001801 | 60.0% | 0.00422 | 7.8%  |

Off-center fiducial bounds (r ∈ {0.001, 0.006}, τ ∈ {0.04, 0.07}) for the
multi-fiducial evaluation are in `results_v3/fisher_multifid/`.

### SpectralCNN range-averaged errors (v3)

Mean |Δθ/θ| over the test range (r > 0.001 for the r error) — the training
diagnostic, not directly a fiducial-point σ:

| NSIDE | params | f_sky=1.0/n=0 | f_sky=1.0/n=6 | f_sky=0.1/n=0 | f_sky=0.1/n=6 |
|-------|--------|---------------|---------------|---------------|---------------|
| 16  | 6.7M  | r 21.9% / τ 15.1% | r 32.7% / τ 20.0% | r 57.6% / τ 26.9% | r 56.3% / τ 28.4% |
| 32  | 26.5M | r 56.7% / τ 19.6% | r 58.3% / τ 16.3% | r 59.0% / τ 24.6% | r 56.6% / τ 26.6% |
| 128 (hc=8) | 29.9M | r 59.1% / τ 21.7% | r 58.6% / τ 21.1% | r 58.4% / τ 24.3% | — |

**The plateau:** at NSIDE≥32 the r error sits at 55–59% in every
configuration while the Fisher bound tightens to 8–27% — the network does
not exploit the additional small-scale information.

### Capacity scaling (NSIDE=32, f_sky=1.0, noise=0)

| hidden ch. | params | r error | τ error | epochs |
|------------|--------|---------|---------|--------|
| 32  | 26.5M  | 56.7% | 19.6% | 16 |
| 64  | 103.5M | 54.8% | 24.4% | 17 |
| 128 | 409M   | 58.1%* | 32.2%* | 4* (walltime) |

\* partial run. A 16× capacity increase does not move the plateau →
**the bottleneck is not model capacity.** Training loss keeps decreasing
while validation error stalls, pointing at training-set diversity
(5000 distinct CAMB spectra reused ~20× each across 100k maps).

### Fiducial-point evaluation (RMSE vs Fisher) — v3

1000 signal+noise realizations at (r=0.003, τ=0.054) per config;
RMSE² = σ² + bias². Produced on Popeye CPU (jobs 2450946 N16/N32,
2451069 N128; torch 2.12 CPU). Expanse GPU job 52109433 left queued as an
optional GPU/HDF5-mask cross-check.

| NSIDE | f_sky | noise | σ(r) | bias(r) | RMSE(r) | RMSE/Fisher (r) | RMSE/Fisher (τ) |
|-------|-------|-------|------|---------|---------|-----------------|-----------------|
| 16 | 1.0 | 0 | 0.000708 | −0.000090 | 0.000713 | 2.06× | 2.24× |
| 16 | 1.0 | 6 | 0.000995 | −0.000771 | 0.001259 | 1.78× | 3.25× |
| 16 | 0.1 | 0 | 0.000130 | −0.001386 | 0.001392 | 1.27× | 1.12× |
| 16 | 0.1 | 6 | 0.000180 | −0.001636 | 0.001646 | **0.74×** | 3.30× |
| 32 | 1.0 | 0 | 0.000076 | −0.001898 | 0.001899 | 7.33× | 1.88× |
| 32 | 1.0 | 6 | 0.000015 | −0.001948 | 0.001948 | 3.31× | 2.76× |
| 32 | 0.1 | 0 | 0.000007 | −0.001976 | 0.001976 | 2.41× | **0.29×** |
| 32 | 0.1 | 6 | 0.000074 | −0.001961 | 0.001963 | 1.05× | 1.03× |
| 128 | 1.0 | 0 | 0.000003 | −0.001947 | 0.001947 | 7.89× | 5.26× |
| 128 | 1.0 | 6 | 0.000000 | −0.001681 | 0.001681 | 2.95× | 6.51× |
| 128 | 0.1 | 0 | 0.000000 | −0.001744 | 0.001744 | 2.23× | 4.18× |
| 128 | 0.1 | 6 | 0.000000 | −0.001891 | 0.001891 | 1.05× | **0.20×** |

At NSIDE=128 (422M params, 2–5 effective training epochs) the collapse is
total: σ(r) ≤ 3×10⁻⁶ and σ(τ) numerically zero in 3 of 4 configs — the
network is a deterministic constant. The **0.20× τ ratio is a pure
coincidence** (collapsed τ̂ = 0.0548 lands on the fiducial 0.054), the
starkest illustration of why sub-Fisher ratios need a response measurement.

Sub-unity ratios are bias-dominated shrinkage or constant-coincidence
(N16 fsky=0.1/noise=6: |bias| ≈ 9σ), **not** super-efficiency — see the
multi-fiducial test below.

### Multi-fiducial response — N16 calibrated, N32 collapsed

Recovered ⟨r̂⟩ at three true r values (f_sky=1.0, noise=0, 1000
realizations each):

| NSIDE | r_true=0.001 | r_true=0.003 | r_true=0.006 | response |
|-------|--------------|--------------|--------------|----------|
| 16 | 0.00085 | 0.00291 | 0.00597 | **linear, slope 1.02 — calibrated** |
| 32 | 0.00110 | 0.00110 | 0.00110 | **constant — r channel collapsed** |

The N32 network outputs r̂ ≈ 0.0011 with σ ≈ 8×10⁻⁵ *independently of the
input*. Consequences:

- The "55–59% r error plateau" at NSIDE≥32 is the error of a near-constant
  predictor, and the consistent near-zero-r bias of +0.0007 is that constant
  evaluated at r_true < 0.001. This **supersedes the Jensen's-inequality
  interpretation** in `results_v3/r_bias_analysis.md` (the Jensen mechanism
  fits the bias magnitude but predicts a prediction spread the multi-fiducial
  data rule out).
- A constant predictor "beats Fisher" (0.67×) at r_true=0.001 where the
  constant happens to land close — the textbook reason sub-Fisher ratios
  need a response measurement before being believed.
- τ keeps a real but shrunk response at both resolutions (slope ≈ 0.37 at
  N16, ≈ 0.29 at N32 between τ=0.04 and 0.07).

Off-center CNN evals: `results_v3/test4_cnn_fiducial_nside*_r*.json`;
matching Fisher bounds: `results_v3/fisher_multifid/`.

### Controlled experiments (2026-07-16, NRP)

**Seed replicas** — Test 3 retrained with 3 independent inits:
2.18 / 2.20 / 2.30% → **2.23 ± 0.06%**, robustly below the KT19 MCMC (2.8%).

**Training-diversity experiment** — two N32 hc32 fsky=1.0/noise=0 runs
identical except the spectral library (on-the-fly, seed-matched protocol):

| Arm | Spectra | r %err | near-zero-r bias | epochs |
|-----|---------|--------|------------------|--------|
| control | 5000 (seed 142) | 58.7% | +0.0009 | 16 |
| treatment | 20000 (seed 777) | 62.3% | +0.0007 | 16 |

**NULL result**: 4× spectral diversity does not restore the r response —
the collapse is unchanged. Together with the capacity null, this
eliminates the two resource explanations; the scalar log-r point-estimate
objective is the leading suspect. JSONs:
`results_v3/test4_nside32_div{5k,20k}_fsky1.0_noise0.json`; response
measurements in `results_v3/test4_cnn_fiducial_div*` (Popeye eval).

### MCMC baseline — negative result

A pseudo-C_ℓ Metropolis–Hastings fit (r, τ, A_lens; 50×50 interpolated CAMB
grid) cannot constrain r from single-map `anafast` spectra: chains drift to
the prior boundary (r→0.01, τ→0.08) in all configs
(`results_v3/test4_mcmc_mh_*.json`, r errors ≈ 200–230%). Single-realization
pseudo-C_ℓ estimates are too noisy for this simple likelihood. The Fisher
bound is therefore the classical reference for Test 4.

---

## Architecture comparison

| Property | SpectralCNN | NNhealpix |
|----------|-------------|-----------|
| Domain | spectral (ℓ, m) | pixel (HEALPix) |
| Parameters (T1 / T2-3 / T4-N16) | 6.5M / 9.8M / 6.7M | ~0.08M / ~0.24M |
| Parameter scaling | O(ℓ²max·C²) | O(k²·C²) |
| Mask handling | mean-inpainting before SHT | natural (zeros) |
| Noise robustness | low (SHT spreads noise) | high (pooling low-pass) |
| Best for | polarization, global structure | noisy scalar maps |

## Superseded results (v2 — D_ℓ bug, provenance only)

<details>
<summary>Click to expand the pre-fix Test 4 numbers</summary>

The v2 Test 4 campaign (NSIDE=16 fiducial evals in `results/`,
`test4_cnn_fiducial_*.json`) reported RMSE/Fisher ratios of 0.38–1.34 and
was the basis for early "CNN beats the Fisher bound" claims. Two bugs
invalidated the comparison:

1. Training/eval maps used D_ℓ instead of C_ℓ amplitudes (`raw_cl` missing).
2. Fisher bounds mixed CAMB accuracy settings across NSIDE
   (`set_for_lmax`), making the NSIDE=16 bound ~4.5× too loose
   (σ_r/r 52.1% instead of 11.5%).

With both fixed, the v3 comparison above replaces these numbers.

</details>

## References

1. Krachmalnicoff & Tomasi (2019), A&A 628, A129, arXiv:1902.04083
2. Bonev et al. (2023), ICML, arXiv:2306.03838
3. Ocampo, Price & McEwen (2023), ICLR, arXiv:2209.13603
