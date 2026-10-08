#!/bin/bash
#SBATCH --job-name=thh-mcmc-v4
#SBATCH --partition=gen
#SBATCH --account=soap
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/results_v4/mcmc_cpu_%j.out
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# v4 MCMC baselines (CPU only): Test 1 all noise levels + Tests 2/3.
# Exact Gaussian likelihood, fit range l<=2*Nside (see mcmc_baseline.py).
# CAMB spectra cached to /mnt/sdceph (shared with the GPU jobs).

set -e
export PYTHONUNBUFFERED=1
export THH_CAMB_CACHE_DIR=/mnt/sdceph/users/zonca/torch-harmonics-healpix/camb_cache

cd /mnt/home/azonca/torch-harmonics-healpix
mkdir -p results_v4 $THH_CAMB_CACHE_DIR

echo "=== Node info ==="
hostname
date

echo "=== Activating venv ==="
source ~/torch-hh-v4-venv/bin/activate

echo "=== Package versions ==="
python3 -c "
import torch, healpy, numpy, scipy, sys
print(f'Python: {sys.version}')
print(f'torch: {torch.__version__}')
print(f'healpy: {healpy.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scipy: {scipy.__version__}')
import camb; print(f'camb: {camb.__version__}')
"

echo "=== Test 1: all noise levels, 1000 maps each ==="
python3 scripts/run_mcmc_benchmark.py 1000 \
    --output results_v4/mcmc_test1_v4.json

echo "=== Tests 2 and 3: 1000 maps each ==="
python3 scripts/run_mcmc_test23_v4.py --n_maps 1000 --output_dir results_v4

echo "=== DONE ==="
date
