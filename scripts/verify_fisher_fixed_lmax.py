#!/usr/bin/env python3
"""Verify Fisher matrix with fixed-lmax CAMB spectra.

After the bug fix (computing all CAMB spectra at lmax=500 then truncating),
the Fisher information should increase monotonically with lmax:
  lmax=47 (N16)  <  lmax=95 (N32)  <  lmax=383 (N128)

This script computes Fisher bounds for all three NSIDEs and all 4 configs,
printing a clean comparison table.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, 'src')

from fisher_forecast import compute_fisher_matrix, LMAX_CALC_FIXED

# Configurations: (fsky, noise_arcmin, label)
configs = [
    (1.0, 0, "fsky1.0_noise0"),
    (1.0, 6, "fsky1.0_noise6"),
    (0.1, 0, "fsky0.1_noise0"),
    (0.1, 6, "fsky0.1_noise6"),
]

# NSIDEs to test
nsides = [16, 32, 128]

r_fid = 0.003
tau_fid = 0.054

print("=" * 80)
print("Fisher Forecast Verification — Fixed lmax CAMB")
print(f"  LMAX_CALC_FIXED = {LMAX_CALC_FIXED}")
print(f"  Fiducial: r={r_fid}, tau={tau_fid}")
print("=" * 80)

results = {}

for nside in nsides:
    lmax = 3 * nside - 1
    npix = 12 * nside ** 2
    pixel_area_arcmin2 = (4.0 * np.pi / npix) * (180.0 * 60.0 / np.pi) ** 2

    for fsky, noise_arcmin, label in configs:
        # Convert noise: μK-arcmin → μK per pixel
        noise_uK_per_pix = noise_arcmin / np.sqrt(pixel_area_arcmin2) if noise_arcmin > 0 else 0.0

        # Compute Fisher matrix
        F = compute_fisher_matrix(
            r=r_fid,
            tau=tau_fid,
            lmax=lmax,
            nside=nside,
            noise_std_uK=noise_uK_per_pix,
            f_sky=fsky,
        )

        # Invert to get covariance (Cramér-Rao bound)
        cov = np.linalg.inv(F)
        sigma_r = np.sqrt(cov[0, 0])
        sigma_tau = np.sqrt(cov[1, 1])
        corr = cov[0, 1] / (sigma_r * sigma_tau)

        r_pct = sigma_r / r_fid * 100
        tau_pct = sigma_tau / tau_fid * 100

        key = f"nside{nside}_{label}"
        results[key] = {
            "nside": nside,
            "lmax": lmax,
            "config": label,
            "f_sky": fsky,
            "noise_arcmin": noise_arcmin,
            "noise_uK_per_pixel": float(noise_uK_per_pix),
            "r_fiducial": r_fid,
            "tau_fiducial": tau_fid,
            "sigma_r": float(sigma_r),
            "sigma_tau": float(sigma_tau),
            "r_pct_error": float(r_pct),
            "tau_pct_error": float(tau_pct),
            "correlation_r_tau": float(corr),
            "fisher_matrix": F.tolist(),
            "lmax_calc_fixed": LMAX_CALC_FIXED,
        }

        print(f"NSIDE={nside:3d} lmax={lmax:3d} {label:20s}: "
              f"σ_r={sigma_r:.6f} ({r_pct:5.1f}%), "
              f"σ_τ={sigma_tau:.6f} ({tau_pct:5.1f}%), "
              f"ρ={corr:+.4f}")

    print()

# Verify monotonicity: σ_r should decrease (improve) with higher NSIDE
print("=" * 80)
print("Monotonicity Check (σ_r should decrease with NSIDE)")
print("=" * 80)
for fsky, noise_arcmin, label in configs:
    sigmas = []
    for nside in nsides:
        key = f"nside{nside}_{label}"
        sigmas.append(results[key]["sigma_r"])
    trend = "✓ monotonic" if all(sigmas[i] > sigmas[i+1] for i in range(len(sigmas)-1)) else "✗ NOT monotonic"
    print(f"  {label:20s}: σ_r = {sigmas[0]:.6f} → {sigmas[1]:.6f} → {sigmas[2]:.6f}  {trend}")

# Save results
output_path = os.path.join(os.path.dirname(__file__), '..', 'results_v3', 'fisher_fixed_lmax_verification.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
