#!/bin/bash
#SBATCH --job-name=thh-eval-dr
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/eval_directr_%j.out

# Multi-fiducial response measurement of the direct-r ablation
# checkpoint (N32, linear r head): does removing the log(r+eps)
# objective restore an input-dependent r response?

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=24

date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

RESULTS=/mnt/home/azonca/torch-harmonics-healpix/results_v3

for R in 0.003 0.001 0.006; do
  echo "=== directr (NSIDE=32, linear head) fiducial r=$R ==="
  python3 scripts/eval_test4_at_fiducial.py \
    --results_dir $RESULTS --models_dir $RESULTS/experiments/directr \
    --nside 32 --hidden_channels 32 \
    --r_param linear \
    --r_fid $R --tau_fid 0.054 --n_realizations 1000 \
    --out_prefix test4_cnn_fiducial_directr_r${R}
done

echo "=== eval_directr complete ==="
date
