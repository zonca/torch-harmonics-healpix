#!/bin/bash
#SBATCH --job-name=thh-n32-hc32
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n32_v3_%j.out

# N32 CNN training — GPU only (HDF5 must already exist from Popeye generation)
# Fisher + HDF5 gen done on Popeye (CPU), this script only does CNN training

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_n32

# Optional: run single config via sbatch args (default: all 4)
FSKY=${1:-all}
NOISE=${2:-all}

mkdir -p $RESULTS_DIR

echo "=== N32 CNN training (v3, hc=32) ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source $VENV/bin/activate
cd $REPO

NSIDE=32
HIDDEN_CHANNELS=32
NUM_BLOCKS=3
BATCH_SIZE=16
MAX_EPOCHS=50
PATIENCE=15

if [ "$FSKY" = "all" ]; then
    CONFIGS="1.0:0 1.0:6 0.1:0 0.1:6"
else
    CONFIGS="${FSKY}:${NOISE}"
fi

for PAIR in $CONFIGS; do
    FSKY_VAL=$(echo $PAIR | cut -d: -f1)
    NOISE_VAL=$(echo $PAIR | cut -d: -f2)
    LABEL="fsky${FSKY_VAL}_noise${NOISE_VAL}"
    HDF5_FILE=$HDF5_DIR/test4_nside32_${LABEL}.h5
    CNN_JSON=$RESULTS_DIR/test4_nside32_hc32_${LABEL}.json

    echo ""
    echo "================================================================"
    echo "=== CNN training: $LABEL (NSIDE=$NSIDE) ==="
    echo "================================================================"

    if [ -f "$CNN_JSON" ]; then
        echo "Results already exist: $CNN_JSON — skipping"
        continue
    fi

    if [ ! -f "$HDF5_FILE" ]; then
        echo "ERROR: HDF5 not found: $HDF5_FILE — skipping (run Popeye gen first)"
        continue
    fi

    date
    python3 scripts/train_test4.py \
        --nside $NSIDE \
        --f_sky $FSKY_VAL \
        --noise_std $NOISE_VAL \
        --hidden_channels $HIDDEN_CHANNELS \
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

    echo "=== $LABEL complete ==="
    date
done

echo ""
echo "================================================================"
echo "=== All N32 CNN training complete ==="
echo "================================================================"
date
