#!/bin/bash
#SBATCH --job-name=thh-nmt
#SBATCH --partition=gen
#SBATCH --account=soap
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/results_v4b/mcmc_namaster_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# NaMaster pseudo-C_l baseline for Test 2 partial skies (CPU only).
# Same 1000 maps per f_sky as the naive MCMC baseline (identical rng
# sequence); replaces the C/f_sky hack with apodization + coupling-matrix
# deconvolution + B-mode purification. Complements the CNN story: if the
# CNN still wins, it wins against a real pseudo-C_l pipeline.

set -e
export PYTHONUNBUFFERED=1

cd /mnt/home/azonca/torch-harmonics-healpix
mkdir -p results_v4b

echo "=== Node info ==="
hostname; date

echo "=== Activating venv ==="
source ~/torch-hh-v4-venv/bin/activate

echo "=== pymaster check ==="
python3 -c "import pymaster; print('pymaster', pymaster.__version__)"

python3 scripts/run_mcmc_namaster_v4.py \
    --n_maps 1000 \
    --output results_v4b/mcmc_namaster_test2_v4b.json

echo "=== DONE ==="
date
