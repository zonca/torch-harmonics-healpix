#!/bin/bash
#SBATCH --job-name=thh-train
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/train_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Node info ==="
hostname
date
nvidia-smi

echo "=== Setting up venv on /tmp ==="
WORKDIR=/tmp/zonca_torch_hh_train_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Use uv consistently for venv + pip
UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib pytest pytest-cov 2>&1 | tail -1

echo "=== Copying repo ==="
REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo
cp -r $REPO_SRC $WORKDIR/repo

echo "=== Installing torch-harmonics-healpix in dev mode ==="
uv pip install -e "$WORKDIR/repo[dev]" 2>&1 | tail -1

cd $WORKDIR/repo

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

# Train for each noise level
for NOISE in 0 5 10 15; do
    echo ""
    echo "============================================"
    echo "=== Training with noise_std = $NOISE ==="
    echo "============================================"
    python3 scripts/train_test1.py \
        --noise_std $NOISE \
        --n_train 50000 \
        --n_val 1000 \
        --n_test 500 \
        --epochs 50 \
        --batch_size 64 \
        --lr 1e-3 \
        --output $RESULTS_DIR/test1_noise${NOISE}.json
done

echo ""
echo "=== All training complete ==="
date

# Print summary
echo ""
echo "=== Results Summary ==="
for NOISE in 0 5 10 15; do
    echo "--- Noise σ_n = $NOISE ---"
    cat $RESULTS_DIR/test1_noise${NOISE}.json
    echo ""
done
