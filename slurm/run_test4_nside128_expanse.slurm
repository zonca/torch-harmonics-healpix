#!/bin/bash
#SBATCH --job-name=thh-t4-n128
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --account=sds166
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

# Test 4 at NSIDE=128: joint r/τ estimation with higher resolution
# NSIDE=128 → lmax=383, much more BB signal available for r constraint.
# Batch size reduced to 16 (maps are 64x larger than NSIDE=16).

set -e
export PYTHONUNBUFFERED=1

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
RESULTS=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
NSIDE=128
LMAX=383
CAMB_CACHE=$RESULTS/test4_camb_spectra_nside${NSIDE}.fits

cd $REPO

# Activate existing venv (set up by previous Expanse jobs)
source $REPO/.venv/bin/activate

echo "=== Test 4 NSIDE=128: Joint r/τ estimation ==="
echo "NSIDE=$NSIDE, LMAX=$LMAX"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Pre-compute CAMB spectra (all 4 configs share the same spectra)
# Config 1: f_sky=1.0, noise=0
echo ""
echo "=== Config 1: f_sky=1.0, noise=0 ==="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky1.0_noise0.json

# Config 2: f_sky=1.0, noise=6 μK-arcmin
echo ""
echo "=== Config 2: f_sky=1.0, noise=6 μK-arcmin ==="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky1.0_noise6.json

# Config 3: f_sky=0.1, noise=0
echo ""
echo "=== Config 3: f_sky=0.1, noise=0 ==="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky0.1_noise0.json

# Config 4: f_sky=0.1, noise=6 μK-arcmin
echo ""
echo "=== Config 4: f_sky=0.1, noise=6 μK-arcmin ==="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 6 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky0.1_noise6.json

echo ""
echo "=== All NSIDE=128 training complete ==="
date
