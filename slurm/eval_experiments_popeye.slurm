#!/bin/bash
#SBATCH --job-name=thh-eval-exp
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/eval_experiments_%j.out

# Fiducial + multi-fiducial evaluation of the publishability-experiment
# checkpoints (NRP-trained):
# - diversity arms div5k/div20k (NSIDE=32): response measurement at
#   r = 0.001 / 0.003 / 0.006 — does 4x spectral diversity restore the
#   r response, or is the collapse unchanged?
# - N16 seed replicas s2/s3 (eviction-shortened at ~epoch 53, near
#   converged): central fiducial for seed scatter on the Fisher ratio.
# Checkpoints staged under results_v3/experiments/<tag>/ with standard
# names so eval_test4_at_fiducial.py finds them via --models_dir.

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

for TAG in div5k div20k; do
  for R in 0.003 0.001 0.006; do
    echo "=== $TAG (NSIDE=32) fiducial r=$R ==="
    python3 scripts/eval_test4_at_fiducial.py \
      --results_dir $RESULTS --models_dir $RESULTS/experiments/$TAG \
      --nside 32 --hidden_channels 32 \
      --r_fid $R --tau_fid 0.054 --n_realizations 1000 \
      --out_prefix test4_cnn_fiducial_${TAG}_r${R}
  done
done

for TAG in s2 s3; do
  echo "=== N16 replica $TAG central fiducial ==="
  python3 scripts/eval_test4_at_fiducial.py \
    --results_dir $RESULTS --models_dir $RESULTS/experiments/$TAG \
    --nside 16 --hidden_channels 32 \
    --n_realizations 1000 \
    --out_prefix test4_cnn_fiducial_n16_${TAG}
done

echo "=== eval_experiments complete ==="
date
