#!/bin/bash
#SBATCH --job-name=thh-t4-n128-fs${FSKY}_n${NOISE}
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n128_fsky${FSKY}_noise${NOISE}_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FSKY=${FSKY:-1.0}
NOISE=${NOISE:-0}
NSIDE=128

echo "=== N128 training: fsky=${FSKY}, noise=${NOISE} ==="
date
hostname

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO
git pull origin main 2>&1 | tail -2

DATA_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2
mkdir -p $RESULTS_DIR

LABEL="fsky${FSKY}_noise${NOISE}"
HDF5_FILE=$DATA_DIR/test4_nside128_${LABEL}.h5

if [ ! -f $HDF5_FILE ]; then
    echo "HDF5 missing: $HDF5_FILE"
    exit 1
fi

python3 scripts/train_test4.py \
    --nside $NSIDE \
    --f_sky $FSKY \
    --noise_std $NOISE \
    --batch_size 16 \
    --max_epochs 25 \
    --hidden_channels 32 \
    --num_blocks 3 \
    --cosine_T_max 25 \
    --output $RESULTS_DIR/test4_nside128_${LABEL} \
    --hdf5_path $HDF5_FILE || echo "FAILED"

echo "=== Done ==="
ls -lh $RESULTS_DIR/test4_nside128_${LABEL}_*.json 2>/dev/null
date
