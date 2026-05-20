#!/bin/bash
#SBATCH --job-name=thh-v3
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/train_v3_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== V3 Multi-Resolution Training ==="
echo "=== Node info ==="
hostname
date
nvidia-smi

echo "=== Setting up venv on /tmp ==="
WORKDIR=/tmp/zonca_torch_hh_v3_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib 2>&1 | tail -1

echo "=== Copying repo ==="
REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cp -r $REPO_SRC $WORKDIR/repo

echo "=== Installing torch-harmonics-healpix in dev mode ==="
uv pip install -e "$WORKDIR/repo[dev]" --no-deps 2>&1 | tail -1

cd $WORKDIR/repo

echo "=== Package versions ==="
python3 -c "
import sys, torch, healpy, numpy, scipy
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'healpy: {healpy.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scipy: {scipy.__version__}')
"

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

# Train for each noise level with multi-resolution model
for NOISE in 0 5 10 15; do
    echo ""
    echo "============================================"
    echo "=== V3 Training with noise_std = $NOISE ==="
    echo "============================================"
    python3 scripts/train_test1_v3.py \
        --noise_std $NOISE \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --max_epochs 150 --patience 20 \
        --batch_size 32 --lr 1e-3 \
        --output $RESULTS_DIR/test1_v3_noise${NOISE}.json
done

echo ""
echo "=== V3 Training complete ==="
date

# Print summary
echo ""
echo "=== Results Summary ==="
for NOISE in 0 5 10 15; do
    echo "--- Noise σ_n = $NOISE ---"
    cat $RESULTS_DIR/test1_v3_noise${NOISE}.json 2>/dev/null || echo "  (not found)"
    echo ""
done
