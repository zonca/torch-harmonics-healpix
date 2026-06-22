#!/bin/bash
#SBATCH --job-name=thh-n32-cap
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n32_cap_%j.out

# N32 CNN capacity scaling study
# Tests whether larger hidden_channels (64, 128) can break through
# the ~56-59% r error ceiling observed with hc=32.
#
# Usage:
#   sbatch slurm/train_n32_capacity_scaling.slurm          # all configs
#   sbatch slurm/train_n32_capacity_scaling.slurm 64       # only hc=64
#   sbatch slurm/train_n32_capacity_scaling.slurm 128 1.0  # hc=128, fsky=1.0 only

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_n32

# Args: HC_LIST, FSKY, NOISE
HC_LIST=${1:-"64 128"}
FSKY=${2:-all}
NOISE=${3:-all}

mkdir -p $RESULTS_DIR

echo "=== N32 CNN capacity scaling study ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source $VENV/bin/activate
cd $REPO

NSIDE=32
NUM_BLOCKS=3
MAX_EPOCHS=50
PATIENCE=15

# Build config list
if [ "$FSKY" = "all" ]; then
    CONFIGS="1.0:0 1.0:6 0.1:0 0.1:6"
else
    CONFIGS="${FSKY}:${NOISE}"
fi

for HC in $HC_LIST; do
    # Scale batch_size with capacity to fit V100 32GB
    # hc=32: bs=16, hc=64: bs=8, hc=128: bs=4
    if [ "$HC" -ge 128 ]; then
        BATCH_SIZE=4
    elif [ "$HC" -ge 64 ]; then
        BATCH_SIZE=8
    else
        BATCH_SIZE=16
    fi

    echo ""
    echo "================================================================"
    echo "=== hidden_channels=$HC, batch_size=$BATCH_SIZE ==="
    echo "================================================================"

    for PAIR in $CONFIGS; do
        FSKY_VAL=$(echo $PAIR | cut -d: -f1)
        NOISE_VAL=$(echo $PAIR | cut -d: -f2)
        LABEL="fsky${FSKY_VAL}_noise${NOISE_VAL}"
        HDF5_FILE=$HDF5_DIR/test4_nside32_${LABEL}.h5
        CNN_JSON=$RESULTS_DIR/test4_nside32_hc${HC}_${LABEL}.json

        echo ""
        echo "--- hc=$HC, $LABEL (NSIDE=$NSIDE) ---"

        if [ -f "$CNN_JSON" ]; then
            echo "Results already exist: $CNN_JSON — skipping"
            continue
        fi

        if [ ! -f "$HDF5_FILE" ]; then
            echo "ERROR: HDF5 not found: $HDF5_FILE — skipping"
            continue
        fi

        date
        python3 scripts/train_test4.py \
            --nside $NSIDE \
            --f_sky $FSKY_VAL \
            --noise_std $NOISE_VAL \
            --hidden_channels $HC \
            --num_blocks $NUM_BLOCKS \
            --batch_size $BATCH_SIZE \
            --lr 1e-3 \
            --max_epochs $MAX_EPOCHS \
            --patience $PATIENCE \
            --scheduler cosine \
            --cosine_T_max $MAX_EPOCHS \
            --cosine_eta_min 1e-7 \
            --grad_clip 1.0 \
            --weight_decay 1e-4 \
            --hdf5_path $HDF5_FILE \
            --num_workers 0 \
            --output $CNN_JSON

        echo "=== $LABEL (hc=$HC) complete ==="
        date
    done
done

echo ""
echo "================================================================"
echo "=== N32 capacity scaling study complete ==="
echo "================================================================"
date
