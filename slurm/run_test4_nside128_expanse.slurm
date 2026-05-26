#!/bin/bash
#SBATCH --job-name=thh-t4-n128
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_n128_gpu_%j.out

# Test 4 at NSIDE=128: joint r/τ estimation with higher resolution
# NSIDE=128 → lmax=383, much more BB signal for r constraint.
# Batch size 16 (maps are 64× larger than NSIDE=16).

set -e
export PYTHONUNBUFFERED=1

echo "=== Test 4 NSIDE=128: Joint r/τ estimation ==="
hostname
date
nvidia-smi

# Create venv in /tmp (fast local storage)
WORKDIR=/tmp/zonca_torch_hh_t4_n128_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO_SRC
git pull origin main 2>&1 | tail -3
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
NSIDE=128
LMAX=383
CAMB_CACHE=$RESULTS_DIR/test4_camb_spectra_nside${NSIDE}.fits
mkdir -p $RESULTS_DIR

# Config 1: f_sky=1.0, noise=0
echo ""
echo "=========================================="
echo "=== Config 1: f_sky=1.0, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 16 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_nside${NSIDE}_fsky1.0_noise0.json

# Config 2: f_sky=1.0, noise=6 μK-arcmin
echo ""
echo "=========================================="
echo "=== Config 2: f_sky=1.0, noise=6 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 16 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_nside${NSIDE}_fsky1.0_noise6.json

# Config 3: f_sky=0.1, noise=0
echo ""
echo "=========================================="
echo "=== Config 3: f_sky=0.1, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 0 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 16 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_nside${NSIDE}_fsky0.1_noise0.json

# Config 4: f_sky=0.1, noise=6 μK-arcmin
echo ""
echo "=========================================="
echo "=== Config 4: f_sky=0.1, noise=6 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 6 \
    --nside $NSIDE --lmax $LMAX \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 16 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_nside${NSIDE}_fsky0.1_noise6.json

echo ""
echo "=== All NSIDE=128 training complete ==="
ls -lh $RESULTS_DIR/test4_nside${NSIDE}_*.json $RESULTS_DIR/test4_nside${NSIDE}_*.pt $CAMB_CACHE 2>/dev/null
date
