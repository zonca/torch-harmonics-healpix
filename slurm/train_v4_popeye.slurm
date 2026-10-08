#!/bin/bash
#SBATCH --job-name=thh-train-v4
#SBATCH --partition=gpupreempt
#SBATCH --account=soap
#SBATCH --qos=gpupreempt
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/results_v4/train_gpu_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

# v4 GPU retrains: Tests 1-3 SpectralCNN on the FIXED polarization data
# (healpy synfast new=True) with the new stack (torch 2.11 + cu130,
# torch-harmonics 0.9.2, healpy 1.20.1). One A100-40GB (pcn-16-06).
# gpupreempt is preemptible: checkpoints after every epoch + --resume support.
# All CPU-heavy setup (CAMB grid) is cached on /mnt/sdceph by the CPU job.

set -e
export PYTHONUNBUFFERED=1
export THH_CAMB_CACHE_DIR=/mnt/sdceph/users/zonca/torch-harmonics-healpix/camb_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/home/azonca/torch-harmonics-healpix
mkdir -p results_v4 checkpoints_v4

echo "=== Node info ==="
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
date

echo "=== Activating venv ==="
source ~/torch-hh-v4-venv/bin/activate

echo "=== CUDA check ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA FAILED'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "=== Package versions ==="
python3 -c "
import torch, torch_harmonics, healpy, numpy, scipy, sys
print(f'Python: {sys.version}')
print(f'torch: {torch.__version__}')
print(f'torch-harmonics: {torch_harmonics.__version__}')
print(f'healpy: {healpy.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scipy: {scipy.__version__}')
"

# --- Test 1: one model per noise level (v3 architecture, fixed data) ---
for NOISE in 0 5 10 15; do
    echo "=== Test 1: noise_std=$NOISE ==="
    python3 scripts/train_test1_v3.py \
        --noise_std $NOISE \
        --n_train 20000 --n_val 2000 --n_test 1000 \
        --max_epochs 150 --patience 20 --batch_size 32 \
        --output results_v4/test1_v4_noise$NOISE.json
done

# --- Test 2: one model per f_sky (full sky + 4 KT19 caps) ---
for FSKY in 1.0 0.5 0.2 0.1 0.05; do
    echo "=== Test 2: f_sky=$FSKY ==="
    python3 scripts/train_test2_v2.py \
        --f_sky $FSKY \
        --n_train 20000 --n_val 2000 --n_test 1000 \
        --max_epochs 150 --patience 20 --batch_size 32 \
        --output results_v4/test2_v4_fsky$FSKY.json
done

# --- Test 3: full sky, noiseless (paper setup) ---
echo "=== Test 3 ==="
python3 scripts/train_test3_v2.py \
    --n_train 20000 --n_val 2000 --n_test 1000 \
    --max_epochs 150 --patience 20 --batch_size 32 \
    --output results_v4/test3_v4.json

echo "=== DONE ==="
date
