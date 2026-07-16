#!/bin/bash
#SBATCH --job-name=thh-t4-camb
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/precompute_test4_caches_%j.out

# Test 4 CAMB caches for the publishability experiments:
# 1. 5000 spectra, lmax=47, seed=142  — reproduces the v3 N16 in-job
#    sampling (for seed-replica runs on NRP)
# 2. 20000 spectra, lmax=95, seed=777 — the training-diversity arm
#    (4x the spectral library of the v3 N32 baseline)

set -e
export PYTHONUNBUFFERED=1

date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

mkdir -p results_v3

python3 scripts/precompute_test4_camb.py \
    --n_spectra 5000 --lmax 47 --seed 142 --nproc 48 \
    --output results_v3/camb_cache_test4_n16_5k.fits

python3 scripts/precompute_test4_camb.py \
    --n_spectra 20000 --lmax 95 --seed 777 --nproc 48 \
    --output results_v3/camb_cache_test4_n32_20k.fits

echo "=== test4 caches complete ==="
date
