#!/bin/bash
#SBATCH --job-name=thh-t4-mcmc
#SBATCH --partition=gen
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zonca/torch-harmonics-healpix/test4_mcmc_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Test 4: MCMC baselines for joint r/τ estimation ==="
hostname
date

cd $HOME/torch-harmonics-healpix
source .venv/bin/activate

RESULTS_DIR=$HOME/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

echo "=== Running MCMC baselines for all 4 configs ==="
python3 -c "
import numpy as np
import healpy as hp
import json
import time
import psutil
import os

from torch_harmonics_healpix.data_generation_test4 import (
    precompute_camb_spectra_r_tau, generate_r_tau_map, NSIDE, LMAX, R_MAX, R_LOG_EPSILON
)
from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
from torch_harmonics_healpix.mcmc_baselines_test2_3 import mcmc_baseline_r_tau

# Noise conversion: μK-arcmin -> μK per pixel
def noise_arcmin_to_uK(noise_arcmin, nside):
    npix = hp.nside2npix(nside)
    pixel_area_rad2 = 4 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_rad2 * (180 * 60 / np.pi)**2
    return noise_arcmin / np.sqrt(pixel_area_arcmin2)

configs = [
    {'f_sky': 1.0, 'noise_arcmin': 0, 'label': 'fsky1.0_noise0'},
    {'f_sky': 1.0, 'noise_arcmin': 6, 'label': 'fsky1.0_noise6'},
    {'f_sky': 0.1, 'noise_arcmin': 0, 'label': 'fsky0.1_noise0'},
    {'f_sky': 0.1, 'noise_arcmin': 6, 'label': 'fsky0.1_noise6'},
]

n_test = 1000
rng = np.random.default_rng(42)

# Pre-compute CAMB spectra
print('Pre-computing 5000 CAMB spectra...')
t0 = time.time()
r_values, tau_values, cl_ee_array, cl_bb_array = precompute_camb_spectra_r_tau(5000, LMAX, seed=12345)
print(f'CAMB pre-computation: {time.time()-t0:.1f}s')

# Create shared masks
shared_masks = {}
for f_sky in [1.0, 0.1]:
    mask_rng = np.random.default_rng(0)
    shared_masks[f_sky] = create_sky_mask(f_sky, NSIDE, mask_rng).astype(np.float32)

# Grid for MCMC search
r_grid = np.linspace(0, R_MAX, 50)
tau_grid = np.linspace(0.03, 0.08, 50)

# Memory tracking
process = psutil.Process(os.getpid())
mem_peak = 0

for cfg in configs:
    f_sky = cfg['f_sky']
    noise_uK = noise_arcmin_to_uK(cfg['noise_arcmin'], NSIDE) if cfg['noise_arcmin'] > 0 else 0.0
    mask = shared_masks[f_sky]

    print(f'\n=== Config: {cfg[\"label\"]} ===')
    print(f'  f_sky={f_sky}, noise={cfg[\"noise_arcmin\"]} μK-arcmin = {noise_uK:.2f} μK/pixel')

    r_errors = []
    tau_errors = []
    t_start = time.time()

    for i in range(n_test):
        # Pick random spectrum
        idx = rng.integers(0, len(r_values))
        r_true = r_values[idx]
        tau_true = tau_values[idx]

        # Generate map
        q, u, _ = generate_r_tau_map(
            r_true, tau_true, NSIDE, LMAX, noise_uK, f_sky, rng,
            cl_ee=cl_ee_array[idx], cl_bb=cl_bb_array[idx]
        )

        # Run MCMC baseline
        result = mcmc_baseline_r_tau(
            q, u, mask, r_grid, tau_grid,
            cl_ee_array, cl_bb_array,
            noise_std=noise_uK, nside=NSIDE, lmax=LMAX
        )

        r_pct = abs(result['r_best'] - r_true) / max(r_true, 0.001) * 100
        tau_pct = abs(result['tau_best'] - tau_true) / tau_true * 100
        r_errors.append(r_pct)
        tau_errors.append(tau_pct)

        if (i+1) % 100 == 0:
            mem = process.memory_info().rss / 1e9
            mem_peak = max(mem_peak, mem)
            print(f'  {i+1}/{n_test}: r={np.mean(r_errors):.1f}%, τ={np.mean(tau_errors):.1f}%, mem={mem:.1f}GB')

    elapsed = time.time() - t_start
    mem = process.memory_info().rss / 1e9
    mem_peak = max(mem_peak, mem)

    result_data = {
        'test': 'test4_mcmc',
        'config': cfg['label'],
        'f_sky': f_sky,
        'noise_arcmin': cfg['noise_arcmin'],
        'noise_uK_per_pixel': float(noise_uK),
        'n_test': n_test,
        'r_pct_error': float(np.mean(r_errors)),
        'tau_pct_error': float(np.mean(tau_errors)),
        'r_pct_error_std': float(np.std(r_errors)),
        'tau_pct_error_std': float(np.std(tau_errors)),
        'cpu_time_seconds': elapsed,
        'peak_memory_GB': mem_peak,
        'method': 'chi2_grid_search',
        'r_grid_size': len(r_grid),
        'tau_grid_size': len(tau_grid),
    }

    outpath = f'{RESULTS_DIR}/test4_mcmc_{cfg[\"label\"]}.json'
    with open(outpath, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Results saved: {outpath}')
    print(f'  r: {np.mean(r_errors):.1f}% ± {np.std(r_errors):.1f}%')
    print(f'  τ: {np.mean(tau_errors):.1f}% ± {np.std(tau_errors):.1f}%')
    print(f'  CPU time: {elapsed:.1f}s, Peak memory: {mem_peak:.1f}GB')

print('\n=== All MCMC baselines complete ===')
date
"
