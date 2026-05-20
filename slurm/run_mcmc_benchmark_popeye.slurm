#!/bin/bash
#SBATCH --job-name=thh-mcmc
#SBATCH --partition=gen
#SBATCH --output=~/torch-harmonics-healpix/results/mcmc_benchmark_%j.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -e
export PYTHONUNBUFFERED=1

echo "=== Popeye MCMC Benchmark ==="
echo "=== Node info ==="
hostname
date
lscpu | head -5
cat /proc/meminfo | grep MemTotal

echo "=== Loading Python ==="
source ~/torch-hh-venv/bin/activate

echo "=== Package versions ==="
python3 -c "
import healpy, numpy, scipy, sys
print(f'Python: {sys.version}')
print(f'healpy: {healpy.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scipy: {scipy.__version__}')
"

echo "=== Pulling latest code ==="
cd ~/torch-harmonics-healpix
git pull origin main

echo "=== Reinstalling package ==="
pip install -e . --no-deps 2>&1 | tail -1

RESULTS_DIR=~/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

echo "=== Running MCMC benchmark with 1000 test maps ==="
python3 -c "
import sys, json, time
sys.path.insert(0, 'src')
import numpy as np
from torch_harmonics_healpix.mcmc_baseline import evaluate_mcmc_baseline
from torch_harmonics_healpix.data_generation import generate_dataset

results = {}
for noise_std in [0, 5, 10, 15]:
    print(f'\\n=== noise_std = {noise_std} ===')
    np.random.seed(42 + int(noise_std))
    maps, ell_p_true = generate_dataset(n_maps=1000, noise_std=noise_std, seed_start=0)
    t0 = time.time()
    err = evaluate_mcmc_baseline(maps, ell_p_true, noise_std=noise_std)
    t1 = time.time()
    time_per_map = (t1 - t0) / 1000
    print(f'MCMC % error: {err:.2f}%')
    print(f'Time per map: {time_per_map:.4f}s')
    results[str(noise_std)] = {
        'pct_error': float(err),
        'time_per_map': float(time_per_map),
        'n_test': 1000,
    }

with open('$RESULTS_DIR/mcmc_1000maps_popeye.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\\nResults saved.')
"

echo "=== Done ==="
date
