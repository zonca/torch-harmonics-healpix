#!/bin/bash
#SBATCH --job-name=thh-mcmc
#SBATCH --partition=gen
#SBATCH --output=$HOME/torch-harmonics-healpix/results/mcmc_all_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -e
export PYTHONUNBUFFERED=1

echo "=== Popeye MCMC Baselines (All 3 Tests) ==="
echo "=== Node info ==="
hostname
date
lscpu | head -5
cat /proc/meminfo | grep MemTotal

echo "=== Loading Python ==="
source $HOME/torch-hh-venv/bin/activate

echo "=== Pulling latest code ==="
cd $HOME/torch-harmonics-healpix
git pull origin main 2>&1 | tail -3

echo "=== Reinstalling package ==="
pip install -e . --no-deps 2>&1 | tail -1

echo "=== Package versions ==="
python3 -c "
import healpy, numpy, scipy, sys
try:
    import camb
    print(f'camb: {camb.__version__}')
except:
    print('camb: not available')
print(f'Python: {sys.version}')
print(f'healpy: {healpy.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scipy: {scipy.__version__}')
"

RESULTS_DIR=$HOME/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

# =============================================
# Test 1: MCMC baseline for ℓ_p estimation
# =============================================
echo ""
echo "============================================"
echo "=== Test 1 MCMC: ℓ_p estimation ==="
echo "============================================"

python3 -c "
import sys, json, time
sys.path.insert(0, 'src')
import numpy as np
from torch_harmonics_healpix.mcmc_baseline import evaluate_mcmc_baseline
from torch_harmonics_healpix.data_generation import generate_dataset

results = {}
for noise_std in [0, 5, 10, 15]:
    print(f'\n=== noise_std = {noise_std} ===')
    np.random.seed(42 + int(noise_std))
    maps, ell_p_true = generate_dataset(n_maps=1000, noise_std=noise_std, seed=42+int(noise_std))
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

with open('$RESULTS_DIR/mcmc_test1_popeye.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nTest 1 MCMC results saved.')
"

# =============================================
# Test 2: MCMC baseline for ℓ_Ep/ℓ_Bp estimation
# =============================================
echo ""
echo "============================================"
echo "=== Test 2 MCMC: ℓ_Ep/ℓ_Bp estimation ==="
echo "============================================"

python3 -c "
import sys, json
sys.path.insert(0, 'src')
from torch_harmonics_healpix.mcmc_baselines_test2_3 import evaluate_mcmc_test2

results = {}
for fsky in [1.0, 0.5, 0.2, 0.1, 0.05]:
    print(f'\n=== f_sky = {fsky} ===')
    res = evaluate_mcmc_test2(n_maps=1000, f_sky=fsky, seed=42)
    print(f'MCMC ℓ_Ep % error: {res[\"ep_pct_error\"]:.2f}%')
    print(f'MCMC ℓ_Bp % error: {res[\"bp_pct_error\"]:.2f}%')
    print(f'Time per map: {res[\"time_per_map\"]:.4f}s')
    results[str(fsky)] = res

with open('$RESULTS_DIR/mcmc_test2_popeye.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nTest 2 MCMC results saved.')
"

# =============================================
# Test 3: MCMC baseline for τ estimation
# =============================================
echo ""
echo "============================================"
echo "=== Test 3 MCMC: τ estimation ==="
echo "============================================"

python3 -c "
import sys, json
sys.path.insert(0, 'src')
from torch_harmonics_healpix.mcmc_baselines_test2_3 import evaluate_mcmc_test3

print('\n=== Test 3: τ estimation (full sky) ===')
res = evaluate_mcmc_test3(n_maps=1000, seed=42)
print(f'MCMC τ % error: {res[\"tau_pct_error\"]:.2f}%')
print(f'Time per map: {res[\"time_per_map\"]:.4f}s')

with open('$RESULTS_DIR/mcmc_test3_popeye.json', 'w') as f:
    json.dump(res, f, indent=2)
print('\nTest 3 MCMC results saved.')
"

echo ""
echo "=== All MCMC baselines complete ==="
date
