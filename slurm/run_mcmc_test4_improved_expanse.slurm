#!/bin/bash
#SBATCH --job-name=thh-mcmc-t4
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/mcmc_test4_improved_%j.out

# Improved MCMC baseline for Test 4 NSIDE=128
# Two-stage grid search with BB ℓ_max cutoff
# CPU-only on Expanse (no GPU needed)

set -e
export PYTHONUNBUFFERED=1

echo "=== Improved MCMC Test 4 NSIDE=128 ==="
hostname
date

WORKDIR=/tmp/zonca_torch_hh_mcmc_$SLURM_JOB_ID
trap "rm -rf $WORKDIR" EXIT
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO_SRC
git pull origin main 2>&1 | tail -3
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

python3 scripts/run_mcmc_test4_improved.py \
    --nside 128 \
    --lmax 383 \
    --lmax_bb 10 \
    --results_dir $RESULTS_DIR \
    --n_test 1000 \
    --coarse_grid 100 \
    --fine_grid 50 \
    --fiducial_only

echo "=== Done ==="
ls -lh $RESULTS_DIR/test4_mcmc_improved_*.json 2>/dev/null
date
