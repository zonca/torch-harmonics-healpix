#!/bin/bash
#SBATCH --job-name=thh-test3-v3
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_test3_v3_%j.out

# Test 3 (tau estimation) retrained with the C_l fix (raw_cl=True).
# All previous Test 3 results used D_l amplitudes (unit bug) and are
# not comparable to Krachmalnicoff & Tomasi (2019).
# Requires the corrected CAMB cache (precompute_test3_camb_popeye.slurm,
# shipped to $BASE/camb_cache_test3_v3.fits).

set -e
export PYTHONUNBUFFERED=1

BASE=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix
REPO=$BASE/repo_latest
VENV=$BASE/venv
RESULTS=$BASE/results_v3
CACHE=$BASE/camb_cache_test3_v3.fits

date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source $VENV/bin/activate
cd $REPO

if [ ! -f "$CACHE" ]; then
  echo "WARNING: corrected CAMB cache not found at $CACHE"
  echo "Spectra will be computed in-job (~2h of CPU time)."
  CACHE_ARG=""
else
  CACHE_ARG="--camb_cache $CACHE"
fi

mkdir -p $RESULTS

python3 scripts/train_test3_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 \
    --batch_size 32 --lr 1e-3 \
    $CACHE_ARG \
    --output $RESULTS/test3_v3.json

echo "=== Test 3 v3 complete ==="
date
cat $RESULTS/test3_v3.json
