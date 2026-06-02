#!/usr/bin/env python3
"""Debug: check chi-squared landscape for a single map."""
import numpy as np
import healpy as hp
import sys
sys.path.insert(0, '/mnt/home/azonca/torch-harmonics-healpix/src')
from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau, generate_r_tau_map, R_MAX
)
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask

nside = 128
lmax = 383
r_true = 0.003
tau_true = 0.054

# Generate a noise-free, full-sky map
rng = np.random.default_rng(42)
q, u, mask = generate_r_tau_map(r_true, tau_true, nside, lmax, noise_std=0.0, f_sky=1.0, rng=rng)

# Get observed C_ℓ
maps_in = np.array([np.zeros_like(q), q, u])
cl_obs = hp.anafast(maps_in, lmax=lmax, pol=True)
cl_ee_obs = cl_obs[1]
cl_bb_obs = cl_obs[2]

# Model at true point
cl_ee_true, cl_bb_true = generate_camb_spectra_r_tau(r_true, tau_true, lmax)

print("=== Observed vs Model at true (r,τ) ===")
for ell in [2, 5, 10, 50, 100, 200]:
    print(f"ℓ={ell:3d}: EE obs={cl_ee_obs[ell]:.6e} model={cl_ee_true[ell]:.6e} ratio={cl_ee_obs[ell]/cl_ee_true[ell]:.4f}")
    if ell <= 10:
        print(f"        BB obs={cl_bb_obs[ell]:.6e} model={cl_bb_true[ell]:.6e} ratio={cl_bb_obs[ell]/cl_bb_true[ell]:.4f}")

# Now compute chi-squared at a few grid points
print("\n=== Chi-squared landscape (f_sky=1.0, noise=0) ===")
ell = np.arange(lmax + 1)
lmax_bb = 10

for r_try in [0.0, 0.001, 0.003, 0.005, 0.01]:
    for tau_try in [0.03, 0.04, 0.054, 0.06, 0.08]:
        cl_ee_m, cl_bb_m = generate_camb_spectra_r_tau(r_try, tau_try, lmax)
        f_sky = 1.0
        noise_cl = 0.0
        
        sigma_ee2 = (2.0 / (2 * ell + 1)) * (cl_ee_m + noise_cl) ** 2 / f_sky
        sigma_ee2 = np.maximum(sigma_ee2, 1e-30)
        chi2_ee = np.sum((cl_ee_obs - cl_ee_m) ** 2 / sigma_ee2)
        
        ell_bb = np.arange(lmax_bb + 1)
        sigma_bb2 = (2.0 / (2 * ell_bb + 1)) * (cl_bb_m[:lmax_bb+1] + noise_cl) ** 2 / f_sky
        sigma_bb2 = np.maximum(sigma_bb2, 1e-30)
        chi2_bb = np.sum((cl_bb_obs[:lmax_bb+1] - cl_bb_m[:lmax_bb+1]) ** 2 / sigma_bb2)
        
        chi2 = chi2_ee + chi2_bb
        marker = " <-- true" if r_try == r_true and tau_try == tau_true else ""
        print(f"  r={r_try:.3f}, τ={tau_try:.3f}: χ²={chi2:.1f} (EE={chi2_ee:.1f}, BB={chi2_bb:.1f}){marker}")
