#!/bin/bash
#SBATCH --job-name=thh-fisher-v3
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/fisher_v3_%j.out

# Fisher matrix baseline v3 — all configs, both NSIDEs
# CPU only — runs on Popeye
# ~minutes total (Fisher is fast, no MCMC)

set -e
export PYTHONUNBUFFERED=1

echo "=== Fisher Matrix Baseline v3 ==="
date
hostname

echo "=== Loading Python ==="
module load python/3.11.11
source ~/torch-hh-venv/bin/activate

echo "=== Pulling latest code ==="
cd ~/torch-harmonics-healpix
git pull

RESULTS_DIR=/mnt/home/azonca/torch-harmonics-healpix/results_v3
mkdir -p $RESULTS_DIR

echo "=== Running Fisher matrix: NSIDE=16 + 128, all 4 configs ==="
python3 scripts/run_fisher_mcmc_test4.py \
    --nside 16 128 \
    --skip_mcmc \
    --r_fiducial 0.003 \
    --tau_fiducial 0.054 \
    --results_dir $RESULTS_DIR

echo ""
echo "=== Fisher v3 complete ==="
date
