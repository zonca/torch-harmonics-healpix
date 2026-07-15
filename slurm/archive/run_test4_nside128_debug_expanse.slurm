#!/bin/bash
#SBATCH --job-name=thh-t4-n128d
#SBATCH --partition=debug
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_n128_debug_%j.out

# Test 4 NSIDE=128 on debug partition (fast scheduling, 1h limit)
# Run as many configs as possible within the time limit.

set -e
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
RESULTS=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
NSIDE=128
LMAX=383
CAMB_CACHE=$RESULTS/test4_camb_spectra_nside${NSIDE}.fits

cd $REPO
source $REPO/.venv/bin/activate

echo "=== Test 4 NSIDE=128 (debug partition) ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Config 1
echo "=== Config 1: f_sky=1.0, noise=0 ==="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky1.0_noise0.json || echo "CONFIG 1 FAILED"

# Config 2
echo "=== Config 2: f_sky=1.0, noise=6 ==="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky1.0_noise6.json || echo "CONFIG 2 FAILED"

# Config 3
echo "=== Config 3: f_sky=0.1, noise=0 ==="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky0.1_noise0.json || echo "CONFIG 3 FAILED"

# Config 4
echo "=== Config 4: f_sky=0.1, noise=6 ==="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 6 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky0.1_noise6.json || echo "CONFIG 4 FAILED"

echo "=== Done ==="
date
