#!/usr/bin/env python3
"""Debug: check CAMB output values directly."""
import numpy as np
import sys
sys.path.insert(0, '/mnt/home/azonca/torch-harmonics-healpix/src')

import camb

# Standard Planck cosmology
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, tau=0.054)
pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0.003)
pars.WantTensors = True
pars.set_for_lmax(383)

results = camb.get_results(pars)

# Default: D_ℓ
cls_dl = results.get_total_cls(383, CMB_unit="muK")
# Raw C_ℓ
cls_cl = results.get_total_cls(383, CMB_unit="muK", raw_cl=True)

print("=== CAMB D_ℓ (default) vs C_ℓ (raw_cl=True) ===")
for ell in [2, 5, 10, 50, 100, 200]:
    print(f"ℓ={ell:3d}: TT D_ℓ={cls_dl[ell,0]:.2e}, EE D_ℓ={cls_dl[ell,1]:.2e}, BB D_ℓ={cls_dl[ell,2]:.2e}")
    print(f"        TT C_ℓ={cls_cl[ell,0]:.2e}, EE C_ℓ={cls_cl[ell,1]:.2e}, BB C_ℓ={cls_cl[ell,2]:.2e}")
    print(f"        D_ℓ/C_ℓ ratio EE: {cls_dl[ell,1]/cls_cl[ell,1]:.2f}, expected ℓ(ℓ+1)/(2π): {ell*(ell+1)/(2*np.pi):.2f}")

# Now check: does hp.synfast expect C_ℓ or D_ℓ?
import healpy as hp
nside = 128

# Build cl_full from CAMB raw C_ℓ
cl_tt = cls_cl[:, 0]
cl_ee = cls_cl[:, 1]
cl_bb = cls_cl[:, 2]
cl_te = cls_cl[:, 3]
cl_eb = np.zeros(384)
cl_tb = np.zeros(384)
cl_full = np.array([cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb])

# synfast with CAMB C_ℓ
np.random.seed(42)
maps_from_cl = hp.synfast(cl_full, nside=nside, lmax=383)
cl_obs_from_cl = hp.anafast(maps_from_cl, lmax=383, pol=True)

print("\n=== hp.anafast vs CAMB C_ℓ (using raw_cl=True for synfast) ===")
for ell in [2, 5, 10, 50, 100, 200]:
    ratio_ee = cl_obs_from_cl[1][ell] / cl_ee[ell] if cl_ee[ell] > 0 else 0
    ratio_bb = cl_obs_from_cl[2][ell] / cl_bb[ell] if cl_bb[ell] > 0 else 0
    print(f"ℓ={ell:3d}: EE obs/CAMB_C_ℓ={ratio_ee:.4f}, BB obs/CAMB_C_ℓ={ratio_bb:.4f}")
