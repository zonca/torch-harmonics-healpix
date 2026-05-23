#!/bin/bash
#SBATCH --job-name=thh-save-models
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/save_models_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Train 3 representative models and save .pt weights ==="
echo "=== Node info ==="
hostname
date
nvidia-smi

echo "=== Setting up venv on /tmp ==="
WORKDIR=/tmp/zonca_torch_hh_models_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb huggingface_hub 2>&1 | tail -1

echo "=== Copying repo ==="
REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cp -r $REPO_SRC $WORKDIR/repo

echo "=== Pulling latest code ==="
cd $WORKDIR/repo
git pull origin main 2>&1 | tail -3

echo "=== Installing torch-harmonics-healpix in dev mode ==="
uv pip install -e "$WORKDIR/repo[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

echo ""
echo "============================================"
echo "=== Test 1: σ=0 (best SpectralCNN result) ==="
echo "============================================"
python3 scripts/train_test1_v2.py \
    --noise_std 0 --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test1_v2_fix_noise0.json

echo ""
echo "============================================"
echo "=== Test 2: f_sky=1.0 (full sky polarization) ==="
echo "============================================"
python3 scripts/train_test2_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test2_v2_fix_fsky1.0.json

echo ""
echo "============================================"
echo "=== Test 3: τ estimation, full sky ==="
echo "============================================"
python3 scripts/train_test3_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test3_v2_fix.json

echo ""
echo "============================================"
echo "=== All 3 models trained ==="
echo "============================================"
ls -lh $RESULTS_DIR/*.pt 2>/dev/null
date

echo ""
echo "=== Copying .pt files to repo for HF upload ==="
cp $RESULTS_DIR/test1_v2_fix_noise0.pt $WORKDIR/repo/
cp $RESULTS_DIR/test2_v2_fix_fsky1.0.pt $WORKDIR/repo/
cp $RESULTS_DIR/test3_v2_fix.pt $WORKDIR/repo/

echo "=== Uploading to Hugging Face Hub ==="
HF_TOKEN=$(cat $WORKDIR/repo/.hf_token 2>/dev/null)
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
from huggingface_hub import HfApi, create_repo
import os

token = open('$WORKDIR/repo/.hf_token').read().strip()
api = HfApi(token=token)

repo_id = 'zonca/torch-harmonics-healpix'
try:
    create_repo(repo_id, exist_ok=True, token=token)
    print(f'Repo ready: {repo_id}')
except Exception as e:
    print(f'Repo creation: {e}')

models = [
    ('test1_v2_fix_noise0.pt', 'SpectralCNN Test 1 (σ=0): ℓ_peak estimation from T maps'),
    ('test2_v2_fix_fsky1.0.pt', 'SpectralCNN Test 2 (f_sky=1.0): ℓ_Ep/ℓ_Bp estimation from Q/U maps'),
    ('test3_v2_fix.pt', 'SpectralCNN Test 3 (full sky): τ estimation from Q/U maps'),
]

for fname, desc in models:
    fpath = os.path.join('$WORKDIR/repo', fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f'models/{fname}',
            repo_id=repo_id,
            repo_type='model',
        )
        print(f'Uploaded: {fname} — {desc}')
    else:
        print(f'MISSING: {fname}')

print('Done!')
"
else
    echo "WARNING: No HF token found, skipping upload"
fi

echo ""
echo "=== All done ==="
date
