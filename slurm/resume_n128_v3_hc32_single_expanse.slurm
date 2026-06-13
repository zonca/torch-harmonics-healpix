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
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/resume_n128_v3_hc32_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2

FSKY=${1:-1.0}
NOISE=${2:-0}
LABEL="fsky${FSKY}_noise${NOISE}"
CKPT=$RESULTS_DIR/test4_nside128_hc32_${LABEL}_nside128_best_resume.pt

mkdir -p $RESULTS_DIR

echo "=== N128 RESUME training (v3, hc=32) ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source $VENV/bin/activate
cd $REPO

if [ ! -f "$CKPT" ]; then
  echo "ERROR: Resume checkpoint not found: $CKPT"
  exit 1
fi

echo "Resuming $LABEL from $CKPT"

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
    --resume $CKPT \
    --output $RESULTS_DIR/test4_nside128_hc32_${LABEL}.json

echo "=== $LABEL resume complete ==="
date
