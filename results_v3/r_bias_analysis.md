# Systematic r_bias Analysis

**Date:** 2026-06-21
**Pipeline:** v3 (C_ℓ fix + Huber τ loss + Fisher lmax fix)

---

## Summary

The SpectralCNN exhibits a systematic positive bias in r predictions of ~0.0007 (23% of fiducial r=0.003) across all NSIDE=32 and NSIDE=128 configurations. This bias is a direct consequence of the log-r parameterization and Jensen's inequality.

---

## Observed r_bias Across NSIDEs

| NSIDE | Config               | r %err   | r_bias    | bias/r_fid |
|-------|----------------------|----------|-----------|------------|
| 16    | fsky=1.0, noise=0    | 21.9%    | 0.000111  | 3.7%       |
| 16    | fsky=1.0, noise=6    | 32.7%    | 0.000235  | 7.8%       |
| 16    | fsky=0.1, noise=0    | 57.6%    | 0.000918  | 30.6%      |
| 16    | fsky=0.1, noise=6    | 56.3%    | 0.000897  | 29.9%      |
| 32    | fsky=1.0, noise=0    | 56.7%    | 0.000778  | 25.9%      |
| 32    | fsky=1.0, noise=6    | 58.3%    | 0.000724  | 24.1%      |
| 32    | fsky=0.1, noise=0    | 59.0%    | 0.000701  | 23.4%      |
| 32    | fsky=0.1, noise=6    | 56.6%    | 0.000771  | 25.7%      |
| 128   | fsky=1.0, noise=0    | 59.1%    | 0.000704  | 23.5%      |
| 128   | fsky=1.0, noise=6    | 58.6%    | 0.000717  | 23.9%      |
| 128   | fsky=0.1, noise=0    | 58.4%    | 0.000722  | 24.1%      |

**Key pattern:** r_bias is remarkably consistent (~0.0007) across all N32/N128 configs, suggesting a parameterization-level systematic, not a config-specific issue.

---

## Root Cause: Jensen's Inequality

The CNN predicts log(r + ε) where ε = 10⁻⁴, then converts back:
```
r_pred = exp(pred[:, 0]) - R_LOG_EPSILON
```

By Jensen's inequality, if the log(r + ε) predictions have variance σ²_log:
```
E[r_pred] = (r_true + ε) × exp(σ²_log / 2) - ε
r_bias = (r_true + ε) × (exp(σ²_log / 2) - 1)
```

### Implied σ_log from Observed r_bias

For r_fid = 0.003, ε = 0.0001:

| Config                   | r_bias    | σ_log  | σ_r     | σ_r/r_fid |
|--------------------------|-----------|--------|---------|-----------|
| N32 fsky=1.0, noise=0    | 0.000778  | 0.6690 | 0.00170 | 56.7%     |
| N32 fsky=1.0, noise=6    | 0.000724  | 0.6480 | 0.00175 | 58.3%     |
| N32 fsky=0.1, noise=0    | 0.000701  | 0.6385 | 0.00177 | 59.0%     |
| N32 fsky=0.1, noise=6    | 0.000771  | 0.6666 | 0.00170 | 56.6%     |

The implied σ_log ≈ 0.64–0.67 is consistent across configs, confirming the Jensen's inequality explanation.

---

## Bias Correction Impact

If we correct for the systematic r_bias (r_corrected = r_pred - r_bias):

| Config                   | σ_r      | r_bias   | RMSE     | RMSE_corr | Improvement |
|--------------------------|----------|----------|----------|-----------|-------------|
| N32 fsky=1.0, noise=0    | 0.001700 | 0.000778 | 0.001869 | 0.001700  | 9.1%        |
| N32 fsky=1.0, noise=6    | 0.001749 | 0.000724 | 0.001893 | 0.001749  | 7.6%        |
| N32 fsky=0.1, noise=0    | 0.001771 | 0.000701 | 0.001905 | 0.001771  | 7.0%        |
| N32 fsky=0.1, noise=6    | 0.001698 | 0.000771 | 0.001865 | 0.001698  | 9.0%        |

Bias correction improves total RMSE by ~7–9%, but the dominant error component is variance (σ_r), not bias.

---

## Comparison: N16 vs N32/N128

| NSIDE | fsky=1.0 bias/r_fid | fsky=0.1 bias/r_fid |
|-------|---------------------|---------------------|
| 16    | 3.7–7.8%            | 29.9–30.6%          |
| 32    | 24.1–25.9%          | 23.4–25.7%          |
| 128   | 23.5–23.9%          | 24.1%               |

**N16 fsky=1.0 has much smaller bias** (3.7–7.8%) because the CNN achieves much lower r error (21.9%) at NSIDE=16 with full sky. The smaller variance in log(r) predictions leads to a smaller Jensen's inequality bias.

**N16 fsky=0.1 has similar bias to N32/N128** (~30%) because the partial sky makes r estimation harder, increasing the variance and thus the bias.

---

## Recommendations

1. **Apply bias correction in post-processing**: Subtract the empirically measured r_bias (~0.0007) from all CNN predictions. This improves RMSE by ~7–9%.

2. **Address the variance bottleneck**: The CNN's σ_r is 3–7× larger than the Fisher bound. This is the main limitation, not the bias. Possible approaches:
   - Increase model capacity (hc=64, hc=128)
   - Use more training data (more CAMB spectra, more realizations)
   - Improve loss function (likelihood-based instead of MSE on log-r)

3. **Consider alternative parameterizations**: The log-r parameterization introduces Jensen's inequality bias. Alternative approaches:
   - Direct r prediction (no log transform)
   - Bias-corrected log-r: r_pred = exp(pred - σ²_log/2) - ε
   - Quantile regression to capture full posterior

---

## Files

- `results_v3/r_bias_analysis.md` — This report
- `results_v3/cnn_vs_fisher_corrected.json` — Corrected comparison data
- `results_v3/fisher_fixed_lmax_verification.json` — Corrected Fisher bounds
