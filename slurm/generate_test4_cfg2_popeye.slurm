#!/bin/bash
#SBATCH --job-name=thh-gen-cfg2
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%j.out

set -e
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

source ~/torch-hh-venv/bin/activate
cd ~/torch-harmonics-healpix

echo "=== Generating f_sky=1.0, noise=6 ==="
python3 -u scripts/generate_test4_hdf5.py \
    --nside 128 --lmax 383 \
    --f_sky 1.0 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --camb_cache ~/torch-harmonics-healpix/results/test4_camb_spectra_nside128.fits \
    --output /mnt/sdceph/users/zonca/torch-harmonics-healpix/test4_nside128_fsky1.0_noise6.h5

echo "=== Done: f_sky=1.0, noise=6 ==="
