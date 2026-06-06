#!/bin/bash
#SBATCH --job-name=thh-n128-resume
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/resume_n128_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FSKY=${FSKY:-1.0}
NOISE=${NOISE:-0}
NSIDE=128
LABEL="fsky${FSKY}_noise${NOISE}"

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2

echo "=== Resuming N128 training: ${LABEL} ==="
date
hostname

# Per-job venv on compute node /tmp
WORKDIR=/tmp/zonca_n128resume_$SLURM_JOB_ID
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

git pull origin main 2>&1 | tail -2

echo "CUDA check..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Find the best checkpoint from previous run
CKPT="$RESULTS_DIR/test4_nside128_${LABEL}"
RESUME_CKPT="$RESULTS_DIR/test4_nside128_${LABEL}_best_resume.pt"
echo "Best checkpoint: $CKPT"
echo "Resume checkpoint: $RESUME_CKPT"
ls -lh $CKPT $RESUME_CKPT 2>/dev/null

# Use resume checkpoint if it exists, otherwise fall back to old best checkpoint
if [ -f "$RESUME_CKPT" ]; then
    USE_CKPT="$RESUME_CKPT"
    echo "Using full resume checkpoint (includes optimizer/scheduler state)"
else
    USE_CKPT="$CKPT"
    echo "Using old-format best checkpoint (model weights only, starting from epoch 19)"
    START_EPOCH="--start_epoch 18"
fi

# HDF5 path
HDF5="/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2/test4_nside128_${LABEL}.h5"

# Output path (append _resume to distinguish from original)
OUTPUT="$RESULTS_DIR/test4_nside128_${LABEL}_resume.json"

# Train with resume
echo ""
echo "=== Starting resumed training ==="
python3 scripts/train_test4.py \
    --f_sky $FSKY \
    --noise_std $NOISE \
    --nside $NSIDE \
    --max_epochs 150 \
    --patience 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --scheduler cosine \
    --cosine_T_max 25 \
    --cosine_eta_min 0 \
    --hidden_channels 32 \
    --num_blocks 3 \
    --hdf5_path "$HDF5" \
    --num_workers 0 \
    --cache_chunk_size 2000 \
    --resume "$USE_CKPT" \
    $START_EPOCH \
    --output "$OUTPUT" || echo "FAILED: ${LABEL}"

echo ""
echo "=== Done ==="
ls -lh $OUTPUT 2>/dev/null
date
