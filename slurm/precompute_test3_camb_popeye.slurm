#!/bin/bash
#SBATCH --job-name=thh-t3-camb
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/precompute_test3_camb_%j.out

# Precompute the Test 3 CAMB spectra cache with the C_l fix (raw_cl=True).
# The old cache / in-job spectra used D_l amplitudes (unit bug).

set -e
export PYTHONUNBUFFERED=1

date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

mkdir -p results_v3
python3 scripts/precompute_test3_camb.py \
    --nproc 48 \
    --output results_v3/camb_cache_test3_v3.fits

echo "=== precompute_test3_camb complete ==="
date
