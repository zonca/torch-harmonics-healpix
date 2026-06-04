#!/bin/bash
#SBATCH --job-name=thh-n128-train
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n128_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FSKY=${FSKY:-1.0}
NOISE=${NOISE:-0}
NSIDE=128
LABEL="fsky${FSKY}_noise${NOISE}"

echo "=== N128 training: ${LABEL} ==="
date
hostname

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
DATA_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2
mkdir -p $RESULTS_DIR

if [ ! -f ${DATA_DIR}/test4_nside128_${LABEL}.h5 ]; then
    echo "HDF5 missing: ${DATA_DIR}/test4_nside128_${LABEL}.h5"
    exit 1
fi

# Use shared venv if it exists
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
if [ -f $VENV/bin/python3 ]; then
    source $VENV/bin/activate
else
    echo "Creating venv..."
    uv venv $VENV
    source $VENV/bin/activate
    uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install torch-harmonics==0.8.0 --no-deps
    uv pip install healpy h5py scipy matplotlib camb astropy
    cd $REPO && uv pip install -e .
fi

cd $REPO
git pull origin main 2>&1 | tail -2

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
    --hdf5_path ${DATA_DIR}/test4_nside128_${LABEL}.h5 || echo "FAILED"

echo "=== Done ==="
ls -lh $RESULTS_DIR/test4_nside128_${LABEL}_*.json 2>/dev/null
date
