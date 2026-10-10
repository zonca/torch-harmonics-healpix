#!/usr/bin/env python3
"""NaMaster (pseudo-C_ell) baseline for Test 2 partial skies.

The naive MCMC baseline (mcmc_baselines_test2_3.evaluate_mcmc_test2)
applies a simple C_obs/f_sky correction to anafast spectra of zero-masked
maps; mask-induced E/B leakage is NOT corrected and the estimator
degrades strongly at f_sky <= 0.5 (medians 13-25%). This script runs the
proper pseudo-C_ell treatment on the SAME maps:

  - apodized mask (C1, 5 deg) instead of the sharp zero-mask,
  - NaMaster coupling-matrix deconvolution,
  - B-mode purification (purify_b=True) to remove E->B leakage,
  NaMaster-deconvolved bandpowers can be negative (mode-loss noise), so
  the exact Wishart likelihood (which requires positive C_obs) is replaced
  by the standard Gaussian bandpower approximation (Efstathiou 2004):
      chi2 = sum_l (C_obs - C_model)^2 / Var_l,
      Var_l = 2 C_model^2 / ((2l+1) f_sky_eff)
  over the same fit range l in [2, 2*N_side] and the same Gaussian-peak
  model as the naive fit.

  The map-generation sequence (seed -> parameter draws -> mask -> maps) is
  identical to evaluate_mcmc_test2, so both baselines see the same 1000
  maps per f_sky. Run on CPU (Popeye gen partition); requires pymaster.

Usage:
    python3 scripts/run_mcmc_namaster_v4.py --n_maps 1000 \
        --output results_v4b/mcmc_namaster_test2_v4b.json
"""

import argparse
import json
import os
import time

import numpy as np
import healpy as hp
import pymaster as nmt

from torch_harmonics_healpix.data_generation import NSIDE, LMAX, SIGMA_P
from torch_harmonics_healpix.data_generation_test2 import (
    LEP_MIN, LEP_MAX, LBP_MIN, LBP_MAX,
    generate_polarization_map, create_sky_mask,
)
from scipy.optimize import minimize_scalar

APO_SIZE_DEG = 5.0  # apodization scale, ~1 pixel at NSIDE=16 (3.7 deg)


def _chi2_bandpower(cl_obs, cl_model, ell, f_sky_eff):
    """Efstathiou (2004) Gaussian bandpower chi^2.

    Valid for deconvolved bandpowers that may be negative (unlike the
    exact Wishart likelihood, which requires C_obs > 0).
    """
    var = 2 * cl_model ** 2 / ((2 * ell + 1) * f_sky_eff)
    return np.sum((cl_obs - cl_model) ** 2 / var)


def namaster_estimate_ell_ep_bp(
    q_map, u_map, mask_apo, workspace, f_sky_eff,
    nside=NSIDE, lmax=LMAX, sigma_p=SIGMA_P,
):
    """Deconvolved EE/BB pseudo-C_ell + exact-likelihood peak fit.

    workspace: precomputed NmtWorkspace for this mask (coupling matrix
    depends only on the mask and binning, so it is computed once per
    f_sky and reused across maps).
    """
    # Full-sky Q/U in, apodized mask applied by NmtField internally
    f2 = nmt.NmtField(mask_apo, [q_map.astype(np.float64),
                                 u_map.astype(np.float64)],
                      purify_b=True)
    cl = nmt.compute_full_master(f2, f2, workspace)  # [EE, EB, BE, BB]
    cl_ee, cl_bb = cl[0], cl[3]

    lmax_fit = min(2 * nside, lmax)
    ell = np.arange(2, lmax_fit + 1)
    cl_ee = cl_ee[2: lmax_fit + 1]  # may be negative: no floor, use chi^2
    cl_bb = cl_bb[2: lmax_fit + 1]

    def nll_ee(ell_ep):
        cl_model = np.exp(-((ell - ell_ep) ** 2) / (2 * sigma_p**2)) + 1e-5
        return _chi2_bandpower(cl_ee, cl_model, ell, f_sky_eff)

    def nll_bb(ell_bp):
        cl_model = np.exp(-((ell - ell_bp) ** 2) / (2 * sigma_p**2)) + 1e-5
        return _chi2_bandpower(cl_bb, cl_model, ell, f_sky_eff)

    res_ep = minimize_scalar(nll_ee, bounds=(LEP_MIN, LEP_MAX), method="bounded")
    res_bp = minimize_scalar(nll_bb, bounds=(LBP_MIN, LBP_MAX), method="bounded")
    return res_ep.x, res_bp.x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_maps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsky", type=float, nargs="*", default=None,
                   help="Sky fractions (default: partial skies 0.5 0.2 0.1 0.05)")
    p.add_argument("--output", type=str, default="results_v4b/mcmc_namaster_test2_v4b.json")
    args = p.parse_args()

    fsky_values = args.fsky if args.fsky else [0.5, 0.2, 0.1, 0.05]
    nside, lmax = NSIDE, LMAX

    # Unbinned bandpowers (nlb=1): one bandpower per multipole
    b = nmt.NmtBin.from_nside_linear(nside, 1)

    out = {"version": "v4b_namaster", "method": "pseudo-C_l (NaMaster) with "
           "purify_b=True, C1 apodization 5deg, exact Gaussian likelihood",
           "n_maps": args.n_maps, "seed": args.seed,
           "apo_size_deg": APO_SIZE_DEG, "fsky": {}}

    for f_sky in fsky_values:
        print(f"\n=== NaMaster Test 2: f_sky = {f_sky} ===", flush=True)
        # Same rng sequence as evaluate_mcmc_test2 -> identical maps
        rng = np.random.default_rng(args.seed)
        ell_ep_true = rng.uniform(LEP_MIN, LEP_MAX, size=args.n_maps).astype(np.float32)
        ell_bp_true = rng.uniform(LBP_MIN, LBP_MAX, size=args.n_maps).astype(np.float32)
        mask = create_sky_mask(f_sky, nside, rng).astype(np.float32)

        # Apodize + workspace once per f_sky
        mask_apo = nmt.mask_apodization(mask, aposize=APO_SIZE_DEG, apotype="C1")
        f_sky_eff = float(np.mean(mask_apo ** 2))
        f_tmp = nmt.NmtField(mask_apo, [np.zeros(hp.nside2npix(nside))] * 2,
                             purify_b=True)
        workspace = nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(f_tmp, f_tmp, b)
        print(f"  f_sky_eff (apodized) = {f_sky_eff:.4f}", flush=True)

        ep_errors, bp_errors = [], []
        t0 = time.time()
        for i in range(args.n_maps):
            q, u = generate_polarization_map(
                ell_ep_true[i], ell_bp_true[i], nside, lmax, SIGMA_P, 0.0, rng
            )
            ep_pred, bp_pred = namaster_estimate_ell_ep_bp(
                q, u, mask_apo, workspace, f_sky_eff, nside, lmax
            )
            ep_errors.append(abs(ep_pred - ell_ep_true[i]) / ell_ep_true[i] * 100)
            bp_errors.append(abs(bp_pred - ell_bp_true[i]) / ell_bp_true[i] * 100)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{args.n_maps} maps "
                      f"({(time.time()-t0)/(i+1):.2f}s/map)", flush=True)

        res = {
            "ep_pct_error": float(np.mean(ep_errors)),
            "bp_pct_error": float(np.mean(bp_errors)),
            "ep_median_pct_error": float(np.median(ep_errors)),
            "bp_median_pct_error": float(np.median(bp_errors)),
            "time_per_map": (time.time() - t0) / args.n_maps,
            "f_sky_eff": f_sky_eff,
        }
        out["fsky"][str(f_sky)] = res
        print(f"  ep {res['ep_pct_error']:.2f}%  bp {res['bp_pct_error']:.2f}% "
              f"(medians {res['ep_median_pct_error']:.2f}/{res['bp_median_pct_error']:.2f}%)",
              flush=True)

        # Incremental write (keeps partial results on preemption)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
