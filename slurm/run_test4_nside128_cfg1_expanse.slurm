#!/bin/bash
#SBATCH --job-name=thh-t4-n128
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_n128_gpu_%j.out

# Test 4 NSIDE=128 — Config 1 only (short walltime for faster scheduling)
# Will submit remaining configs separately.

set -e
export PYTHONUNBUFFERED=1

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
RESULTS=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
NSIDE=128
LMAX=383
CAMB_CACHE=$RESULTS/test4_camb_spectra_nside${NSIDE}.fits

cd $REPO
source $REPO/.venv/bin/activate

echo "=== Test 4 NSIDE=128 Config 1 ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --batch_size 16 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS/test4_nside${NSIDE}_fsky1.0_noise0.json

echo "=== Config 1 done ==="
date
