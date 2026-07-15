#!/bin/bash
#SBATCH --job-name=thh-eval-n128
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/eval_n128_fiducial_%j.out

# NSIDE=128 hc=32 fiducial-point evaluation on CPU (queue-independence
# hedge for Expanse job 52109433). The shared mask is reconstructed with
# seed 42, verified byte-identical statistics to the HDF5 /mask dataset
# (sum=19656, first=119770, last=196364 for fsky=0.1).
# Checkpoints staged into results_v3/ from Expanse via local relay.

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=48

date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

RESULTS=/mnt/home/azonca/torch-harmonics-healpix/results_v3

python3 scripts/eval_test4_at_fiducial.py \
    --results_dir $RESULTS --models_dir $RESULTS \
    --nside 128 --hidden_channels 32 \
    --mask_seed 42 \
    --n_realizations 1000

echo "=== eval_n128_fiducial (CPU) complete ==="
date
