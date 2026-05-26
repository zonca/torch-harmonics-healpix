#!/bin/bash
#SBATCH --job-name=thh-t4-n128
#SBATCH --partition=gpu
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_n128_gpu_%j.out

# Test 4 at NSIDE=128 on full GPU node (shorter queue than gpu-shared)
# Only uses 1 GPU for training, but full node has faster scheduling

set -e
export PYTHONUNBUFFERED=1

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
RESULTS=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
NSIDE=128
LMAX=383
CAMB_CACHE=$RESULTS/test4_camb_spectra_nside${NSIDE}.fits

cd $REPO

# Activate existing venv
source $REPO/.venv/bin/activate

echo "=== Test 4 NSIDE=128: Joint r/τ estimation ==="
echo "NSIDE=$NSIDE, LMAX=$LMAX"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

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
