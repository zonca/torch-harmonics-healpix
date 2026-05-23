#!/bin/bash
#SBATCH --job-name=thh-t3-model
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/train_t3_model_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Test 3 only: τ estimation with shared CAMB fix ==="
hostname
date
nvidia-smi

WORKDIR=/tmp/zonca_torch_hh_t3_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

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
echo "=== Test 3: τ estimation, full sky ==="
python3 scripts/train_test3_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 \
    --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test3_v2_fix.json

echo ""
echo "=== Upload to Hugging Face ==="
HF_TOKEN=$(cat $WORKDIR/repo/.hf_token 2>/dev/null)
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
from huggingface_hub import HfApi
token = open('$WORKDIR/repo/.hf_token').read().strip()
api = HfApi(token=token)
fname = 'test3_v2_fix.pt'
fpath = '$RESULTS_DIR/' + fname
import os
if os.path.exists(fpath):
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=f'models/{fname}',
        repo_id='zonca/torch-harmonics-healpix',
        repo_type='model',
    )
    print(f'Uploaded: {fname}')
else:
    print(f'MISSING: {fpath}')
"
else
    echo "WARNING: No HF token found"
fi

echo ""
echo "=== Done ==="
date
