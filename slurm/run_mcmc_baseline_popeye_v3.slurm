#!/bin/bash
#SBATCH --job-name=thh-mcmc-v3
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/mcmc_baseline_v3_%j.out

# Full MCMC baseline v3 — Metropolis-Hastings with A_lens nuisance param
# Both NSIDEs, all 4 configs, 100 test maps each
# CPU only — runs on Popeye
# NSIDE=16: ~30-60 min per config
# NSIDE=128: ~2-4 hours per config
# Total: ~10-20 hours

set -e
export PYTHONUNBUFFERED=1

echo "=== MCMC Baseline v3 ==="
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

echo ""
echo "=== Phase 1: MCMC NSIDE=16 (all 4 configs, 100 maps each) ==="
date
python3 scripts/run_fisher_mcmc_test4.py \
    --nside 16 \
    --n_test 100 \
    --mcmc_steps 5000 \
    --mcmc_burn 2000 \
    --r_fiducial 0.003 \
    --tau_fiducial 0.054 \
    --results_dir $RESULTS_DIR
echo "=== Phase 1 complete ==="
date

echo ""
echo "=== Phase 2: MCMC NSIDE=128 (all 4 configs, 100 maps each) ==="
date
python3 scripts/run_fisher_mcmc_test4.py \
    --nside 128 \
    --n_test 100 \
    --mcmc_steps 5000 \
    --mcmc_burn 2000 \
    --r_fiducial 0.003 \
    --tau_fiducial 0.054 \
    --results_dir $RESULTS_DIR
echo "=== Phase 2 complete ==="
date

echo ""
echo "=== MCMC Baseline v3 complete ==="
date
