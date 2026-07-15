#!/usr/bin/env python3
"""Precompute the Test 3 CAMB spectra cache (parallel, CPU).

Produces the same FITS cache that train_test3_v2.py --camb_cache would
build in-job (TAU_VALUES / CL_EE / CL_BB HDUs), but parallelized over
CPU cores so it takes minutes instead of ~2 hours. Run on Popeye, then
ship the small FITS file to the GPU cluster.

The τ values are drawn with the SAME seed as train_test3_v2.py
(seed=142), so the cache is bit-compatible with an in-job computation.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from torch_harmonics_healpix.data_generation_test3 import (
    generate_camb_spectra, N_CAMB_SPECTRA, LMAX, TAU_MIN, TAU_MAX,
)


def _one(args):
    tau, lmax = args
    return generate_camb_spectra(tau, lmax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_spectra", type=int, default=N_CAMB_SPECTRA)
    parser.add_argument("--lmax", type=int, default=LMAX)
    parser.add_argument("--seed", type=int, default=142,
                        help="Seed for τ values (142 = train_test3_v2.py)")
    parser.add_argument("--nproc", type=int, default=os.cpu_count())
    parser.add_argument("--output", type=str, required=True,
                        help="Output FITS cache path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tau_values = rng.uniform(TAU_MIN, TAU_MAX, size=args.n_spectra)

    print(f"Computing {args.n_spectra} CAMB spectra (lmax={args.lmax}) "
          f"on {args.nproc} processes...")
    with Pool(args.nproc) as pool:
        results = pool.map(_one, [(tau, args.lmax) for tau in tau_values],
                           chunksize=16)

    cl_ee_array = np.array([r[0] for r in results], dtype=np.float64)
    cl_bb_array = np.array([r[1] for r in results], dtype=np.float64)

    from astropy.io import fits as pf
    hdu_tau = pf.ImageHDU(tau_values.astype(np.float32)); hdu_tau.name = "TAU_VALUES"
    hdu_ee = pf.ImageHDU(cl_ee_array); hdu_ee.name = "CL_EE"
    hdu_bb = pf.ImageHDU(cl_bb_array); hdu_bb.name = "CL_BB"
    pf.HDUList([pf.PrimaryHDU(), hdu_tau, hdu_ee, hdu_bb]).writeto(
        args.output, overwrite=True)
    print(f"Saved {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
