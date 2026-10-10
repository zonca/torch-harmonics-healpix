#!/usr/bin/env python3
"""NaMaster (pseudo-C_ell) baseline for Test 2 partial skies.

The naive MCMC baseline (mcmc_baselines_test2_3.evaluate_mcmc_test2)
applies a simple C_obs/f_sky correction to anafast spectra of zero-masked
maps; mask-induced E/B leakage is NOT corrected and the estimator
degrades strongly at f_sky <= 0.5. This script runs the proper
pseudo-C_ell treatment on the SAME maps (identical rng sequence):

  - apodized mask (C1, 5 deg) instead of the sharp zero-mask,
  - NaMaster coupling-matrix deconvolution,
  - B-mode purification (purify_b=True) to remove E->B leakage,
  - bandpowers binned with NLB=2: the unbinned (nlb=1) coupling matrix
    at NSIDE=16 is ill-conditioned and deconvolution produces wild
    sign-alternating bandpowers (verified empirically),
  - the Gaussian-peak model is band-averaged and compared with the
    Efstathiou (2004) Gaussian bandpower chi^2:
        chi2 = sum_b (C_obs_b - C_model_b)^2 / Var_b,
        Var_b = 2 C_model_b^2 / (f_sky_eff * N_modes_b),
        N_modes_b = sum_{l in b} (2l+1)
    (valid for deconvolved bandpowers, which may be negative — unlike
    the exact Wishart likelihood, which requires C_obs > 0).

Fit range: bands whose minimum l is in [2, 2*N_side). The map-generation
sequence (seed -> parameter draws -> mask -> maps) is identical to
evaluate_mcmc_test2, so both baselines see the same maps per f_sky.
Run on CPU (Popeye gen partition); requires pymaster.

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
from scipy.optimize import minimize_scalar

from torch_harmonics_healpix.data_generation import NSIDE, LMAX, SIGMA_P
from torch_harmonics_healpix.data_generation_test2 import (
    LEP_MIN, LEP_MAX, LBP_MIN, LBP_MAX,
    generate_polarization_map, create_sky_mask,
)

APO_SIZE_DEG = 5.0  # apodization scale, ~1 pixel at NSIDE=16 (3.7 deg)
NLB = 2             # bandpower bin width; nlb=1 deconvolution is unstable


def build_bands(nside):
    """Bands (indices into the decoupled cell array, their ells) with
    min l in [2, 2*nside): skips l=0,1 (E/B identically zero) and the
    l >= 2*nside aliasing-dominated tail.
    """
    band_idx, ells = [], []
    n_bands = (3 * nside + NLB - 1) // NLB
    for b in range(n_bands):
        band_ells = [NLB * b + i for i in range(NLB) if NLB * b + i <= 3 * nside - 1]
        if 2 <= band_ells[0] < 2 * nside:
            band_idx.append(b)
            ells.append(band_ells)
    return band_idx, ells


def _chi2_binned(cl_obs_b, cl_model_b, n_modes_b, f_sky_eff):
    """Efstathiou (2004) Gaussian bandpower chi^2 on binned spectra."""
    var = 2 * cl_model_b ** 2 / (f_sky_eff * n_modes_b)
    return np.sum((cl_obs_b - cl_model_b) ** 2 / var)


def namaster_estimate_ell_ep_bp(
    q_map, u_map, mask_apo, workspace, f_sky_eff,
    band_idx, ells, n_modes,
    nside=NSIDE, sigma_p=SIGMA_P,
):
    """Deconvolved, binned EE/BB pseudo-C_l + band-averaged peak fit.

    workspace: precomputed NmtWorkspace for this mask and binning
    (coupling matrix depends only on the mask and binning, so it is
    computed once per f_sky and reused across maps).
    """
    f2 = nmt.NmtField(mask_apo, [q_map.astype(np.float64),
                                 u_map.astype(np.float64)],
                      purify_b=True)
    # pymaster 3.x: compute_full_master's 3rd positional arg is `bins`,
    # not workspace — use the explicit coupled->decoupled path instead.
    cl_coupled = nmt.compute_coupled_cell(f2, f2)   # [EE, EB, BE, BB]
    cl = workspace.decouple_cell(cl_coupled)        # [4, n_bands_tot]
    cl_ee_b = cl[0][band_idx]
    cl_bb_b = cl[3][band_idx]

    def model_bands(ell_peak):
        # Gaussian-peak model averaged within each band
        return np.array([
            np.mean(np.exp(-((np.asarray(band_ells) - ell_peak) ** 2)
                           / (2 * sigma_p ** 2)) + 1e-5)
            for band_ells in ells
        ])

    def nll_ee(ell_ep):
        return _chi2_binned(cl_ee_b, model_bands(ell_ep), n_modes, f_sky_eff)

    def nll_bb(ell_bp):
        return _chi2_binned(cl_bb_b, model_bands(ell_bp), n_modes, f_sky_eff)

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

    b = nmt.NmtBin.from_nside_linear(nside, NLB)
    band_idx, ells = build_bands(nside)
    n_modes = np.array([sum(2 * le + 1 for le in band_ells) for band_ells in ells],
                       dtype=float)
    print(f"Bands: {len(band_idx)} covering l={ells[0][0]}..{ells[-1][-1]}", flush=True)

    out = {"version": "v4b_namaster", "method": "pseudo-C_l (NaMaster): "
           "purify_b=True, C1 apodization 5deg, nlb=2 binning, "
           "Efstathiou bandpower chi^2 on band-averaged model",
           "n_maps": args.n_maps, "seed": args.seed,
           "apo_size_deg": APO_SIZE_DEG, "nlb": NLB, "fsky": {}}

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
                q, u, mask_apo, workspace, f_sky_eff, band_idx, ells, n_modes, nside
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
              f"(medians {res['ep_median_pct_error']:.2f}/"
              f"{res['bp_median_pct_error']:.2f}%)", flush=True)

        # Incremental write (keeps partial results on preemption)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
