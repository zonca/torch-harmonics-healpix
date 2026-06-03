#!/bin/bash
#SBATCH --job-name=thh-fisher-mcmc
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/fisher_mcmc_%j.out

# Fisher matrix + MCMC baseline for Test 4
# CPU only — runs on Popeye
# Fisher is fast; MCMC with 100 maps × 5000 steps × 4 configs × 2 NSIDEs

export PYTHONUNBUFFERED=1

echo "=== Fisher + MCMC Test 4 ==="
date
hostname

source ~/torch-hh-venv/bin/activate
cd ~/torch-harmonics-healpix

# Phase 1: Fisher matrix (fast, ~2 min)
echo ""
echo "=== Phase 1: Fisher Matrix ==="
python3 scripts/run_fisher_mcmc_test4.py --skip_mcmc --nside 16 128

echo ""
echo "=== Phase 1 complete ==="
date

# Phase 2: MCMC at NSIDE=16 (fast, ~30 min)
echo ""
echo "=== Phase 2: MCMC NSIDE=16 ==="
python3 scripts/run_fisher_mcmc_test4.py --nside 16 --n_test 100 --mcmc_steps 5000 --mcmc_burn 2000

echo ""
echo "=== Phase 2 complete ==="
date

# Phase 3: MCMC at NSIDE=128 (slow, ~4-8h)
echo ""
echo "=== Phase 3: MCMC NSIDE=128 ==="
python3 scripts/run_fisher_mcmc_test4.py --nside 128 --n_test 100 --mcmc_steps 5000 --mcmc_burn 2000

echo ""
echo "=== Phase 3 complete ==="
date

echo "=== All done ==="
