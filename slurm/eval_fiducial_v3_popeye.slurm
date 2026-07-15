#!/bin/bash
#SBATCH --job-name=thh-eval-cpu
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/eval_fiducial_v3_cpu_%j.out

# CPU mirror of the NSIDE=16/32 part of eval_fiducial_v3_expanse.slurm.
# Runs the fiducial-point evaluation on Popeye so the paper numbers do not
# depend on the Expanse GPU queue. Checkpoints must be staged into
# results_v3/ (scp from Expanse via the local relay).

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

FIDS="0.003:0.054 0.001:0.054 0.006:0.054 0.003:0.040 0.003:0.070"

for NS in 16 32; do
  for FID in $FIDS; do
    R=${FID%%:*}; TAU=${FID##*:}
    echo "=== NSIDE=$NS hc=32 fiducial r=$R tau=$TAU ==="
    python3 scripts/eval_test4_at_fiducial.py \
      --results_dir $RESULTS --models_dir $RESULTS \
      --nside $NS --hidden_channels 32 \
      --r_fid $R --tau_fid $TAU --n_realizations 1000
  done
done

echo "=== eval_fiducial_v3 (CPU) complete ==="
date
