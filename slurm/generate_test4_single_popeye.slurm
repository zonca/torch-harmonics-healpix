#!/bin/bash
#SBATCH --job-name=thh-gen-cfg
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%j.out

set -e
export PYTHONUNBUFFERED=1

VENV=~/torch-hh-venv
REPO=~/torch-harmonics-healpix
RESULTS=/mnt/sdceph/users/zonca/torch-harmonics-healpix
CAMB_CACHE=~/torch-harmonics-healpix/results/test4_camb_spectra_nside128.fits
NSIDE=128
LMAX=383

# Config comes from environment variable
FSKY=${FSKY:-1.0}
NOISE=${NOISE:-0}

source $VENV/bin/activate
cd $REPO

echo "=== Generating f_sky=${FSKY}, noise=${NOISE} ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --lmax $LMAX \
    --f_sky ${FSKY} --noise_std ${NOISE} \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside128_fsky${FSKY}_noise${NOISE}.h5

echo "=== Done: f_sky=${FSKY}, noise=${NOISE} ==="
