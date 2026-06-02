#!/usr/bin/env python3
"""Debug: check chi-squared with D_ℓ conversion applied."""
import numpy as np
import healpy as hp
import sys
sys.path.insert(0, '/mnt/home/azonca/torch-harmonics-healpix/src')
from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau, generate_r_tau_map, R_MAX
)

nside = 128
lmax = 383
r_true = 0.003
tau_true = 0.054

print("Generating map at (r=0.003, τ=0.054), f_sky=1.0, noise=0...")
rng = np.random.default_rng(42)
q, u, mask = generate_r_tau_map(r_true, tau_true, nside, lmax, noise_std=0.0, f_sky=1.0, rng=rng)

maps_in = np.array([np.zeros_like(q), q, u])
cl_obs = hp.anafast(maps_in, lmax=lmax, pol=True)

# Convert to D_ℓ
ell = np.arange(lmax + 1)
ell_factor = np.zeros(lmax + 1)
ell_factor[1:] = ell[1:] * (ell[1:] + 1) / (2 * np.pi)
cl_ee_obs = cl_obs[1] * ell_factor  # D_ℓ
cl_bb_obs = cl_obs[2] * ell_factor  # D_ℓ

# Model at true point (already D_ℓ from CAMB)
cl_ee_true, cl_bb_true = generate_camb_spectra_r_tau(r_true, tau_true, lmax)

print("\n=== Observed D_ℓ vs CAMB D_ℓ at true (r,τ) ===")
for ell_i in [2, 5, 10, 20, 50, 100, 200, 300]:
    ee_ratio = cl_ee_obs[ell_i] / cl_ee_true[ell_i] if cl_ee_true[ell_i] > 0 else 0
    print(f"ℓ={ell_i:3d}: D_ℓ obs={cl_ee_obs[ell_i]:.6e} model={cl_ee_true[ell_i]:.6e} ratio={ee_ratio:.4f}")
    if ell_i <= 10:
        bb_ratio = cl_bb_obs[ell_i] / cl_bb_true[ell_i] if cl_bb_true[ell_i] > 0 else 0
        print(f"        BB D_ℓ obs={cl_bb_obs[ell_i]:.6e} model={cl_bb_true[ell_i]:.6e} ratio={bb_ratio:.4f}")

# Chi-squared with lmax_EE=30 (where τ sensitivity lives)
print("\n=== Chi-squared with D_ℓ conversion, lmax_EE=30, lmax_BB=10 ===")
lmax_ee = 30
lmax_bb = 10
ell_ee = np.arange(lmax_ee + 1)
ell_bb = np.arange(lmax_bb + 1)

for r_try in [0.0, 0.001, 0.003, 0.005, 0.01]:
    for tau_try in [0.03, 0.04, 0.054, 0.06, 0.08]:
        cl_ee_m, cl_bb_m = generate_camb_spectra_r_tau(r_try, tau_try, lmax)
        
        sigma_ee2 = (2.0 / (2 * ell_ee + 1)) * cl_ee_m[:lmax_ee+1] ** 2
        sigma_ee2 = np.maximum(sigma_ee2, 1e-30)
        chi2_ee = np.sum((cl_ee_obs[:lmax_ee+1] - cl_ee_m[:lmax_ee+1]) ** 2 / sigma_ee2)
        
        sigma_bb2 = (2.0 / (2 * ell_bb + 1)) * cl_bb_m[:lmax_bb+1] ** 2
        sigma_bb2 = np.maximum(sigma_bb2, 1e-30)
        chi2_bb = np.sum((cl_bb_obs[:lmax_bb+1] - cl_bb_m[:lmax_bb+1]) ** 2 / sigma_bb2)
        
        chi2 = chi2_ee + chi2_bb
        marker = " <-- true" if r_try == r_true and tau_try == tau_true else ""
        print(f"  r={r_try:.3f}, τ={tau_try:.3f}: χ²={chi2:8.1f} (EE={chi2_ee:8.1f}, BB={chi2_bb:6.1f}){marker}")
