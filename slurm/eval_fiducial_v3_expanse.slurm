#!/bin/bash
#SBATCH --job-name=thh-eval-fid
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/eval_fiducial_v3_%j.out

# Fiducial-point evaluation of ALL v3 checkpoints (corrected pipeline).
# Produces the CNN RMSE = sqrt(bias^2 + sigma^2) numbers that are directly
# comparable to the fixed-lmax Fisher bounds (fisher_forecast.py, lmax=500).
#
# - NSIDE=16 hc=32 and NSIDE=32 hc=32: central fiducial + 4 off-center
#   points (tests whether the CNN generalizes away from the prior mean)
# - NSIDE=32 hc=64: central fiducial (capacity scaling)
# - NSIDE=128 hc=32: central fiducial, shared mask loaded from HDF5
#   (HDF5 masks were generated with seed 42, NOT the on-the-fly seed 0)

set -e
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

BASE=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix
REPO=$BASE/repo_latest
VENV=$BASE/venv
RESULTS=$BASE/results_v3
HDF5=$BASE/hdf5_striped

source $VENV/bin/activate
cd $REPO

date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# (r, tau) fiducial points: central + off-center
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

echo "=== NSIDE=32 hc=64 central fiducial ==="
python3 scripts/eval_test4_at_fiducial.py \
  --results_dir $RESULTS --models_dir $RESULTS \
  --nside 32 --hidden_channels 64 --n_realizations 1000

echo "=== NSIDE=128 hc=32 central fiducial (mask from HDF5) ==="
python3 scripts/eval_test4_at_fiducial.py \
  --results_dir $RESULTS --models_dir $RESULTS \
  --nside 128 --hidden_channels 32 --n_realizations 1000 \
  --hdf5_mask_pattern "$HDF5/test4_nside128_{label}.h5"

echo "=== eval_fiducial_v3 complete ==="
date
