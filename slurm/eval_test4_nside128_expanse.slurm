#!/bin/bash
#SBATCH --job-name="thh-t4-eval"
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/eval_test4_nside128_%j.out

set -e
PYTHONUNBUFFERED=1

module load cpu/0.17.3b gpu/0.17.3b
nvidia-smi

WORKDIR=/tmp/zonca_torch_hh_t4_eval_$SLURM_JOB_ID
# Cleanup on exit
trap "rm -rf $WORKDIR" EXIT
mkdir -p $WORKDIR
cd $WORKDIR

# Create venv
uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results

echo "=== Evaluating NSIDE=128 models at fiducial point ==="
python3 $REPO_SRC/scripts/eval_test4_at_fiducial.py \
    --results_dir $RESULTS_DIR \
    --nside 128 \
    --lmax 383 \
    --n_realizations 1000

echo "=== Done ==="
