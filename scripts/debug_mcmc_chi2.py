#!/usr/bin/env python3
"""Debug: check chi-squared landscape for a single map.

Investigates why the MCMC always hits grid boundaries at NSIDE=128.
Hypothesis: EE chi-squared (384 multipoles) overwhelms BB (11 multipoles
with lmax_bb=10), and EE at high-ℓ is the same for all τ values, so
the minimum always falls at the r boundary.

This script computes the chi-squared with different ℓ ranges for EE
to find which weighting gives a sensible minimum.
"""
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
cl_ee_obs = cl_obs[1]
cl_bb_obs = cl_obs[2]

# Model at true point
cl_ee_true, cl_bb_true = generate_camb_spectra_r_tau(r_true, tau_true, lmax)

print("\n=== Observed vs Model C_ℓ at true (r,τ) ===")
for ell in [2, 5, 10, 20, 50, 100, 200, 300]:
    ee_ratio = cl_ee_obs[ell] / cl_ee_true[ell] if cl_ee_true[ell] > 0 else 0
    print(f"ℓ={ell:3d}: EE obs/model={ee_ratio:.4f}")
    if ell <= 10:
        bb_ratio = cl_bb_obs[ell] / cl_bb_true[ell] if cl_bb_true[ell] > 0 else 0
        print(f"        BB obs/model={bb_ratio:.4f}")

# Test different EE ℓ ranges
print("\n=== Chi-squared with different EE ℓ ranges ===")
for lmax_ee in [10, 30, 50, 100, 200, 383]:
    lmax_bb = min(10, lmax_ee)
    ell_ee = np.arange(lmax_ee + 1)
    ell_bb = np.arange(lmax_bb + 1)
    
    print(f"\nlmax_EE={lmax_ee}, lmax_BB={lmax_bb}:")
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
