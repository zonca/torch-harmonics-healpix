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

# Per-job venv on compute node /tmp (fast local storage)
WORKDIR=/tmp/zonca_n128train_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

echo "Creating venv..."
uv venv .venv --python 3.11
source .venv/bin/activate

echo "Installing torch..."
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1

echo "Installing torch-harmonics..."
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1

echo "Installing other deps..."
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

cd $REPO
uv pip install -e . 2>&1 | tail -1

echo "CUDA check..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

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
