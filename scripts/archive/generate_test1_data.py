#!/usr/bin/env python3
"""Generate Test 1 datasets (HDF5) for ℓ_p estimation benchmark.

Generates HEALPix maps with Gaussian-peaked power spectra at various
noise levels, matching the benchmark from Krachmalnicoff & Tomasi (2019).

Usage:
    python generate_test1_data.py [--output-dir data] [--nside 16]
"""

import argparse
import os
import numpy as np
import h5py

from torch_harmonics_healpix.data_generation import generate_dataset, NSIDE

N_TRAIN = 100_000
N_VAL = 10_000
N_TEST = 1_000


def main():
    parser = argparse.ArgumentParser(description="Generate Test 1 datasets")
    parser.add_argument(
        "--output-dir", default="data", help="Directory for output HDF5 files"
    )
    parser.add_argument("--nside", type=int, default=NSIDE, help="HEALPix Nside")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for noise_std in [0, 5, 10, 15]:
        suffix = f"_noise{noise_std}" if noise_std > 0 else ""
        filepath = os.path.join(args.output_dir, f"test1{suffix}.h5")

        print(f"Generating noise_std={noise_std}...")

        print(f"  Training set ({N_TRAIN} maps)...")
        train_maps, train_ell_p = generate_dataset(
            N_TRAIN, nside=args.nside, noise_std=noise_std, seed=42
        )

        print(f"  Validation set ({N_VAL} maps)...")
        val_maps, val_ell_p = generate_dataset(
            N_VAL, nside=args.nside, noise_std=noise_std, seed=142
        )

        print(f"  Test set ({N_TEST} maps)...")
        test_maps, test_ell_p = generate_dataset(
            N_TEST, nside=args.nside, noise_std=noise_std, seed=242
        )

        with h5py.File(filepath, "w") as f:
            f.create_dataset("train_maps", data=train_maps)
            f.create_dataset("train_ell_p", data=train_ell_p)
            f.create_dataset("val_maps", data=val_maps)
            f.create_dataset("val_ell_p", data=val_ell_p)
            f.create_dataset("test_maps", data=test_maps)
            f.create_dataset("test_ell_p", data=test_ell_p)

        print(f"  Saved to {filepath}")

    print("Done!")


if __name__ == "__main__":
    main()
