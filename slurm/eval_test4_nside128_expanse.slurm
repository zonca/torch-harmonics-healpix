#!/bin/bash
#SBATCH --job-name="thh-t4-eval"
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/eval_test4_nside128_%j.out

# Evaluate NSIDE=128 best models at Fisher fiducial point (r=0.003, τ=0.054)
# 1000 noise realizations per config, ~5 min per config, ~20 min total

set -e
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== NSIDE=128 Fiducial Evaluation ==="
hostname
date
nvidia-smi

WORKDIR=/tmp/zonca_torch_hh_t4_eval_$SLURM_JOB_ID
trap "rm -rf $WORKDIR" EXIT
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO_SRC
git pull origin main 2>&1 | tail -3
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

echo "=== Evaluating NSIDE=128 models at fiducial point ==="
python3 scripts/eval_test4_at_fiducial.py \
    --results_dir $RESULTS_DIR \
    --nside 128 \
    --lmax 383 \
    --n_realizations 1000

echo "=== Done ==="
ls -lh $RESULTS_DIR/test4_cnn_fiducial_*nside128*.json 2>/dev/null
date
