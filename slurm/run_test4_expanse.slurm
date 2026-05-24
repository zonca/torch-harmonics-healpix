#!/bin/bash
#SBATCH --job-name=thh-test4
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_gpu_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Test 4: SpectralCNN training (all 4 configs) ==="
hostname
date
nvidia-smi

WORKDIR=/tmp/zonca_torch_hh_t4_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb astropy huggingface_hub 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO_SRC
git pull origin main 2>&1 | tail -3
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
CAMB_CACHE=$RESULTS_DIR/test4_camb_spectra.fits
mkdir -p $RESULTS_DIR

# Config 1: f_sky=1.0, noise=0 (first run computes CAMB, saves to cache)
echo ""
echo "=========================================="
echo "=== Config 1: f_sky=1.0, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_fsky1.0_noise0.json

# Configs 2-4 reuse the FITS cache (no re-computation)
echo ""
echo "=========================================="
echo "=== Config 2: f_sky=1.0, noise=6 μK-arcmin ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_fsky1.0_noise6.json

echo ""
echo "=========================================="
echo "=== Config 3: f_sky=0.1, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_fsky0.1_noise0.json

echo ""
echo "=========================================="
echo "=== Config 4: f_sky=0.1, noise=6 μK-arcmin ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --camb_cache $CAMB_CACHE \
    --output $RESULTS_DIR/test4_fsky0.1_noise6.json

echo ""
echo "=========================================="
echo "=== Upload models to Hugging Face ==="
echo "=========================================="
python3 -c "
from huggingface_hub import HfApi
import os
token = open('$REPO_SRC/.hf_token').read().strip()
api = HfApi(token=token)
models = [
    'test4_fsky1.0_noise0.pt',
    'test4_fsky1.0_noise6.pt',
    'test4_fsky0.1_noise0.pt',
    'test4_fsky0.1_noise6.pt',
]
for fname in models:
    fpath = os.path.join('$RESULTS_DIR', fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f'models/{fname}',
            repo_id='zonca/torch-harmonics-healpix',
            repo_type='model',
        )
        print(f'Uploaded: {fname}')
    else:
        print(f'MISSING: {fname}')
"

echo ""
echo "=== All Test 4 configs complete ==="
ls -lh $RESULTS_DIR/test4_*.json $RESULTS_DIR/test4_*.pt $CAMB_CACHE 2>/dev/null
date
