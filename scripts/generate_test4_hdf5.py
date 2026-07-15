"""
Pre-generate NSIDE=128 Test 4 datasets as HDF5 files for fast GPU training.

Writes maps incrementally to HDF5 (no RAM accumulation).
Each config gets its own file.

Usage:
    python scripts/generate_test4_hdf5.py --nside 128 --f_sky 1.0 --noise_std 0 \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --output results/test4_nside128_fsky1.0_noise0.h5
"""

import argparse
import time
import numpy as np
import healpy as hp
import h5py
from pathlib import Path

from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
from torch_harmonics_healpix.data_generation_test4 import (
    precompute_camb_spectra_r_tau,
    generate_r_tau_map,
    N_CAMB_SPECTRA,
)

R_LOG_EPSILON = 1e-4


def generate_split_incremental(h5file, split_name, n_maps, r_values, tau_values,
                                cl_ee_array, cl_bb_array, mask, nside, lmax,
                                noise_std, f_sky, seed, chunk_size=1000):
    """Generate maps and write incrementally to HDF5 (constant RAM)."""
    npix = hp.nside2npix(nside)

    if n_maps == 0:
        grp = h5file.create_group(split_name)
        grp.create_dataset('maps', shape=(0, 3, npix), dtype=np.float32)
        grp.create_dataset('targets', shape=(0, 2), dtype=np.float32)
        grp.attrs['n_maps'] = 0
        return

    # Create datasets with full shape but no data yet
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
            cl_ee=cl_ee_array[idx],
            cl_bb=cl_bb_array[idx],
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
    print(f"    Done: {n_maps} maps in {elapsed:.0f}s ({n_maps/elapsed:.1f} maps/s)")


def main():
    parser = argparse.ArgumentParser(description="Generate Test 4 HDF5 dataset")
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--lmax", type=int, default=None)
    parser.add_argument("--f_sky", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--camb_cache", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.lmax is None:
        args.lmax = 3 * args.nside - 1

    nside = args.nside
    lmax = args.lmax
    npix = hp.nside2npix(nside)

    print(f"NSIDE={nside}, lmax={lmax}, npix={npix}")
    print(f"f_sky={args.f_sky}, noise_std={args.noise_std}")
    print(f"n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}")

    # Pre-compute CAMB spectra
    if args.camb_cache and Path(args.camb_cache).exists():
        from astropy.io import fits
        print(f"Loading CAMB cache from {args.camb_cache}")
        with fits.open(args.camb_cache) as hdul:
            r_values = np.array(hdul['R_VALUES'].data)
            tau_values = np.array(hdul['TAU_VALUES'].data)
            cl_ee_array = np.array(hdul['CL_EE'].data)
            cl_bb_array = np.array(hdul['CL_BB'].data)
        print(f"  Loaded {len(r_values)} CAMB spectra")
    else:
        print(f"Pre-computing {N_CAMB_SPECTRA} CAMB spectra...")
        r_values, tau_values, cl_ee_array, cl_bb_array = \
            precompute_camb_spectra_r_tau(N_CAMB_SPECTRA, lmax, seed=args.seed + 100)

    # Create mask (shared across all splits)
    rng = np.random.default_rng(args.seed)
    mask = create_sky_mask(args.f_sky, nside, rng).astype(np.float32)

    # Create HDF5 file and write incrementally
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.attrs['nside'] = nside
        f.attrs['lmax'] = lmax
        f.attrs['f_sky'] = args.f_sky
        f.attrs['noise_std'] = args.noise_std

        f.create_dataset('mask', data=mask)

        generate_split_incremental(f, 'train', args.n_train,
                                    r_values, tau_values, cl_ee_array, cl_bb_array,
                                    mask, nside, lmax, args.noise_std, args.f_sky,
                                    seed=args.seed)
        generate_split_incremental(f, 'val', args.n_val,
                                    r_values, tau_values, cl_ee_array, cl_bb_array,
                                    mask, nside, lmax, args.noise_std, args.f_sky,
                                    seed=args.seed + 1000)
        generate_split_incremental(f, 'test', args.n_test,
                                    r_values, tau_values, cl_ee_array, cl_bb_array,
                                    mask, nside, lmax, args.noise_std, args.f_sky,
                                    seed=args.seed + 2000)

    file_size_gb = output_path.stat().st_size / 1e9
    print(f"Done! File size: {file_size_gb:.1f} GB")


if __name__ == '__main__':
    main()
