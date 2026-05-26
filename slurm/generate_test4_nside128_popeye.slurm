#!/bin/bash
#SBATCH --job-name=thh-gen-n128
#SBATCH --partition=gen
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --mail-type=END

set -e
export PYTHONUNBUFFERED=1

# Setup
VENV=~/torch-hh-venv
REPO=~/torch-harmonics-healpix
RESULTS=~/torch-harmonics-healpix/results
CAMB_CACHE=$RESULTS/test4_camb_spectra_nside128.fits
NSIDE=128
LMAX=383

source $VENV/bin/activate
cd $REPO
pip install h5py --quiet 2>/dev/null || true

# Config 1: f_sky=1.0, noise=0
echo "=== Config 1: f_sky=1.0, noise=0 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --lmax $LMAX \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside128_fsky1.0_noise0.h5

# Config 2: f_sky=1.0, noise=6
echo "=== Config 2: f_sky=1.0, noise=6 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --lmax $LMAX \
    --f_sky 1.0 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside128_fsky1.0_noise6.h5

# Config 3: f_sky=0.1, noise=0
echo "=== Config 3: f_sky=0.1, noise=0 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --lmax $LMAX \
    --f_sky 0.1 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside128_fsky0.1_noise0.h5

# Config 4: f_sky=0.1, noise=6
echo "=== Config 4: f_sky=0.1, noise=6 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --lmax $LMAX \
    --f_sky 0.1 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside128_fsky0.1_noise6.h5

echo "=== All datasets generated ==="
ls -lh $RESULTS/test4_nside128_fsky*.h5
