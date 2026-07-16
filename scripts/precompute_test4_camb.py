#!/usr/bin/env python3
"""Precompute a Test 4 (r, tau) CAMB spectra cache in parallel (CPU).

Produces the FITS cache train_test4.py --camb_cache expects
(R_VALUES / TAU_VALUES / CL_EE / CL_BB HDUs), parallelized over CPU
cores. The (r, tau) sampling reproduces
data_generation_test4.precompute_camb_spectra_r_tau exactly for a given
seed (r uniform in log(r+eps), tau uniform).

Used for the training-set diversity experiment: the v3 baseline uses
5000 spectra (seed 142); the diversity arm uses 20000 (seed 777).
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau, N_CAMB_SPECTRA,
    R_MAX, R_LOG_EPSILON, TAU_MIN, TAU_MAX,
)


def _one(args):
    r, tau, lmax = args
    return generate_camb_spectra_r_tau(r, tau, lmax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_spectra", type=int, default=N_CAMB_SPECTRA)
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--seed", type=int, default=142,
                        help="142 reproduces the v3 in-job sampling")
    parser.add_argument("--nproc", type=int, default=os.cpu_count())
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    log_r_min = np.log(R_LOG_EPSILON)
    log_r_max = np.log(R_MAX + R_LOG_EPSILON)
    r_values = np.exp(rng.uniform(log_r_min, log_r_max,
                                  size=args.n_spectra)) - R_LOG_EPSILON
    tau_values = rng.uniform(TAU_MIN, TAU_MAX, size=args.n_spectra)

    print(f"Computing {args.n_spectra} CAMB (r,tau) spectra (lmax={args.lmax}) "
          f"on {args.nproc} processes...")
    with Pool(args.nproc) as pool:
        results = pool.map(_one, [(r, t, args.lmax)
                                  for r, t in zip(r_values, tau_values)],
                           chunksize=16)

    cl_ee_array = np.array([x[0] for x in results], dtype=np.float64)
    cl_bb_array = np.array([x[1] for x in results], dtype=np.float64)

    from astropy.io import fits as pf
    hdus = [pf.PrimaryHDU()]
    for name, data in (("R_VALUES", r_values.astype(np.float32)),
                       ("TAU_VALUES", tau_values.astype(np.float32)),
                       ("CL_EE", cl_ee_array), ("CL_BB", cl_bb_array)):
        h = pf.ImageHDU(data)
        h.name = name
        hdus.append(h)
    pf.HDUList(hdus).writeto(args.output, overwrite=True)
    print(f"Saved {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
