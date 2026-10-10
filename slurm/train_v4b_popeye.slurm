#!/bin/bash
#SBATCH --job-name=thh-v4b
#SBATCH --partition=gpupreempt
#SBATCH --account=soap
#SBATCH --qos=gpupreempt
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/results_v4b/train_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --gres=gpu:a100-pcie-40gb:1

# v4b: budget-matched final campaign — 100k training maps (KT19 budget),
# 3 seeds per config for error bars. Every training run checkpoints after
# each epoch and resumes with --resume, so gpupreempt preemption and the
# 24h walltime just cost a resubmit, not a restart.
#
# Campaign (3 seeds each):
#   Test 2 (headline): f_sky = 1.0, 0.5, 0.2, 0.1, 0.05
#   Test 3 (headline): full sky, noiseless
#   Test 1: sigma_n = 0, 5, 10, 15
# Test 2 first (headline result), then Test 3, then Test 1.
#
# Outputs: results_v4b/test2_v4b_fsky{F}_seed{S}.json (+ .pt best weights,
# + _ckpt.pt resumable checkpoints in checkpoints_v4b/)

set -e
export PYTHONUNBUFFERED=1
export THH_CAMB_CACHE_DIR=/mnt/sdceph/users/zonca/torch-harmonics-healpix/camb_cache

cd /mnt/home/azonca/torch-harmonics-healpix
mkdir -p results_v4b checkpoints_v4b

echo "=== Node info ==="
hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true; date

echo "=== Activating venv ==="
source ~/torch-hh-v4-venv/bin/activate

echo "=== CUDA check ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA FAILED'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# Shared CAMB cache for Test 3 (all seeds use the same spectral grid)
CAMB_CACHE=/mnt/sdceph/users/zonca/torch-harmonics-healpix/camb_cache/test3_tau_grid.fits

RESUME="--resume"

# --- Test 2: 5 f_sky x 3 seeds (headline) ---
for FSKY in 1.0 0.5 0.2 0.1 0.05; do
  for SEED in 42 123 777; do
    OUT=results_v4b/test2_v4b_fsky${FSKY}_seed${SEED}.json
    if [ -f "$OUT" ]; then echo "=== SKIP $OUT (done) ==="; continue; fi
    echo "=== Test 2: f_sky=$FSKY seed=$SEED ==="
    python3 scripts/train_test2_v2.py \
        --f_sky $FSKY \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --max_epochs 150 --patience 20 --batch_size 32 \
        --seed $SEED $RESUME \
        --checkpoint checkpoints_v4b/test2_fsky${FSKY}_seed${SEED}_ckpt.pt \
        --output $OUT
  done
done

# --- Test 3: 3 seeds (headline) ---
for SEED in 42 123 777; do
  OUT=results_v4b/test3_v4b_seed${SEED}.json
  if [ -f "$OUT" ]; then echo "=== SKIP $OUT (done) ==="; continue; fi
  echo "=== Test 3: seed=$SEED ==="
  python3 scripts/train_test3_v2.py \
      --n_train 100000 --n_val 10000 --n_test 1000 \
      --max_epochs 150 --patience 20 --batch_size 32 \
      --seed $SEED $RESUME \
      --camb_cache $CAMB_CACHE \
      --checkpoint checkpoints_v4b/test3_seed${SEED}_ckpt.pt \
      --output $OUT
done

# --- Test 1: 4 noise levels x 3 seeds ---
for NOISE in 0 5 10 15; do
  for SEED in 42 123 777; do
    OUT=results_v4b/test1_v4b_noise${NOISE}_seed${SEED}.json
    if [ -f "$OUT" ]; then echo "=== SKIP $OUT (done) ==="; continue; fi
    echo "=== Test 1: noise_std=$NOISE seed=$SEED ==="
    python3 scripts/train_test1_v3.py \
        --noise_std $NOISE \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --max_epochs 150 --patience 20 --batch_size 32 \
        --seed $SEED $RESUME --skip_mcmc \
        --checkpoint checkpoints_v4b/test1_noise${NOISE}_seed${SEED}_ckpt.pt \
        --output $OUT
  done
done

echo "=== DONE ==="
date
