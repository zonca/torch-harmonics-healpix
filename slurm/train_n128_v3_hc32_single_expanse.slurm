#!/bin/bash
#SBATCH --job-name=thh-n128-hc32
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n128_v3_hc32_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_striped

FSKY=${1:-1.0}
NOISE=${2:-0}
LABEL="fsky${FSKY}_noise${NOISE}"

mkdir -p $RESULTS_DIR

echo "=== N128 training (v3, hidden_channels=32, 422M params) ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source $VENV/bin/activate
cd $REPO

echo "=== Training $LABEL (N128, hc=32) ==="
echo "Config: --nside 128 --f_sky $FSKY --noise_std $NOISE"
echo "Model: hidden_channels=32, num_blocks=3 (~422M params)"
echo "Loss: MSE(r) + Huber(τ, delta=0.01)"
echo "HDF5: $HDF5_DIR/test4_nside128_${LABEL}.h5"
echo "Output: $RESULTS_DIR/test4_nside128_hc32_${LABEL}.json"

python3 scripts/train_test4.py \
    --nside 128 \
    --f_sky $FSKY \
    --noise_std $NOISE \
    --hidden_channels 32 \
    --num_blocks 3 \
    --batch_size 2 \
    --lr 1e-3 \
    --max_epochs 25 \
    --patience 10 \
    --scheduler cosine \
    --cosine_T_max 25 \
    --cosine_eta_min 1e-7 \
    --grad_clip 1.0 \
    --weight_decay 1e-4 \
    --hdf5_path $HDF5_DIR/test4_nside128_${LABEL}.h5 \
    --num_workers 0 \
    --cache_chunk_size 5000 \
    --output $RESULTS_DIR/test4_nside128_hc32_${LABEL}.json

echo "=== $LABEL complete ==="
date
