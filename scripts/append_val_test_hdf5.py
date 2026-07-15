"""
Append val/test splits to an existing HDF5 dataset that only has train.

Usage:
    python scripts/append_val_test_hdf5.py \
        --f_sky 1.0 --noise_std 6 --n_val 10000 --n_test 1000 \
        --camb_cache ... --output /path/to/existing.h5
"""

import argparse
import time
import numpy as np
import healpy as hp
import h5py
from pathlib import Path

from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
from torch_harmonics_healpix.data_generation_test4 import (
    generate_r_tau_map, N_CAMB_SPECTRA, precompute_camb_spectra_r_tau,
)

R_LOG_EPSILON = 1e-4


def append_split(h5file, split_name, n_maps, r_values, tau_values,
                 cl_ee_array, cl_bb_array, mask, nside, lmax,
                 noise_std, f_sky, seed):
    npix = hp.nside2npix(nside)
    grp = h5file.create_group(split_name)
    maps_ds = grp.create_dataset('maps', shape=(n_maps, 3, npix), dtype=np.float32,
                                  chunks=(1, 3, npix))
    targets_ds = grp.create_dataset('targets', shape=(n_maps, 2), dtype=np.float32)
    grp.attrs['n_maps'] = n_maps

    rng = np.random.default_rng(seed)
    spec_indices = rng.integers(0, len(r_values), size=n_maps)

    print(f"  Generating {split_name}: {n_maps} maps...")
    t0 = time.time()
    for i in range(n_maps):
        idx = spec_indices[i]
        q, u, _ = generate_r_tau_map(
            float(r_values[idx]), float(tau_values[idx]),
            nside, lmax, noise_std, f_sky, rng,
            cl_ee=cl_ee_array[idx], cl_bb=cl_bb_array[idx],
            mask=mask,
        )
        maps_ds[i, 0] = q
        maps_ds[i, 1] = u
        maps_ds[i, 2] = mask
        targets_ds[i, 0] = np.log(float(r_values[idx]) + R_LOG_EPSILON)
        targets_ds[i, 1] = float(tau_values[idx])

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_maps - i - 1) / rate
            print(f"    {i+1}/{n_maps} ({rate:.1f} maps/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"    Done: {n_maps} maps in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--lmax", type=int, default=383)
    parser.add_argument("--f_sky", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--camb_cache", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from astropy.io import fits
    print(f"Loading CAMB cache from {args.camb_cache}")
    with fits.open(args.camb_cache) as hdul:
        r_values = np.array(hdul['R_VALUES'].data)
        tau_values = np.array(hdul['TAU_VALUES'].data)
        cl_ee_array = np.array(hdul['CL_EE'].data)
        cl_bb_array = np.array(hdul['CL_BB'].data)

    rng = np.random.default_rng(args.seed)
    mask = create_sky_mask(args.f_sky, args.nside, rng).astype(np.float32)

    print(f"Appending val/test to {args.output}...")
    with h5py.File(args.output, 'a') as f:
        if 'val' in f:
            del f['val']
        if 'test' in f:
            del f['test']
        if 'mask' not in f:
            f.create_dataset('mask', data=mask)

        append_split(f, 'val', args.n_val, r_values, tau_values,
                     cl_ee_array, cl_bb_array, mask, args.nside, args.lmax,
                     args.noise_std, args.f_sky, seed=args.seed + 1000)
        append_split(f, 'test', args.n_test, r_values, tau_values,
                     cl_ee_array, cl_bb_array, mask, args.nside, args.lmax,
                     args.noise_std, args.f_sky, seed=args.seed + 2000)

    file_size_gb = Path(args.output).stat().st_size / 1e9
    print(f"Done! File size: {file_size_gb:.1f} GB")


if __name__ == '__main__':
    main()
