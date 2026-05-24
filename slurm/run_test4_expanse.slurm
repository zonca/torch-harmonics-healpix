#!/bin/bash
#SBATCH --job-name=thh-test4
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_gpu_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Test 4: Joint r/τ estimation (all 4 configs) ==="
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
uv pip install healpy h5py scipy matplotlib camb huggingface_hub 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cp -r $REPO_SRC $WORKDIR/repo
cd $WORKDIR/repo
git pull origin main 2>&1 | tail -3
uv pip install -e "$WORKDIR/repo[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

echo ""
echo "=========================================="
echo "=== Fisher Forecast (CPU, fast) ==="
echo "=========================================="
for FSKY in 1.0 0.1; do
    for NOISE in 0 6; do
        echo "--- Fisher: f_sky=$FSKY, noise=$NOISE μK-arcmin ---"
        python3 scripts/fisher_forecast.py \
            --r 0.003 --tau 0.054 \
            --noise_std $NOISE --f_sky $FSKY \
            --nside 16 --lmax 47 2>&1 | tail -10
    done
done

echo ""
echo "=========================================="
echo "=== Config 1: f_sky=1.0, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test4_fsky1.0_noise0.json

echo ""
echo "=========================================="
echo "=== Config 2: f_sky=1.0, noise=6 μK-arcmin ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test4_fsky1.0_noise6.json

echo ""
echo "=========================================="
echo "=== Config 3: f_sky=0.1, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test4_fsky0.1_noise0.json

echo ""
echo "=========================================="
echo "=== Config 4: f_sky=0.1, noise=6 μK-arcmin ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test4_fsky0.1_noise6.json

echo ""
echo "=========================================="
echo "=== Upload models to Hugging Face ==="
echo "=========================================="
HF_TOKEN=$(cat $WORKDIR/repo/.hf_token 2>/dev/null)
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
from huggingface_hub import HfApi
import os
token = open('$WORKDIR/repo/.hf_token').read().strip()
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
else
    echo "WARNING: No HF token found, skipping upload"
fi

echo ""
echo "=== All Test 4 configs complete ==="
ls -lh $RESULTS_DIR/test4_*.json $RESULTS_DIR/test4_*.pt 2>/dev/null
date
