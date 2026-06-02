#!/bin/bash
#SBATCH --job-name=thh-mcmc-t4
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/mcmc_test4_improved_%j.out

# Improved MCMC baseline for Test 4 NSIDE=128
# Two-stage grid search with BB ℓ_max cutoff
# Uses Popeye CPU (no GPU needed)

# No set -e — CAMB segfaults would kill the job; Python handles errors internally
export PYTHONUNBUFFERED=1

echo "=== Improved MCMC Test 4 NSIDE=128 ==="
hostname
date

source ~/torch-hh-venv/bin/activate

REPO_DIR=~/torch-harmonics-healpix
RESULTS_DIR=~/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR/logs

cd $REPO_DIR
git pull origin main 2>&1 | tail -3

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
date
