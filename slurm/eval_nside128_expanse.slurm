#!/bin/bash
#SBATCH --job-name=thh-n128-eval
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/eval_n128_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== N128 test evaluation ==="
date
hostname

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2

# Per-job venv on compute node /tmp
WORKDIR=/tmp/zonca_n128eval_$SLURM_JOB_ID
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

# Evaluate all 4 configs
for FSKY_NOISE in "1.0_noise0" "1.0_noise6" "0.1_noise0" "0.1_noise6"; do
    LABEL="fsky${FSKY_NOISE}"
    CKPT="$RESULTS_DIR/test4_nside128_${LABEL}"
    HDF5="/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2/test4_nside128_${LABEL}.h5"
    OUT="$RESULTS_DIR/test4_nside128_${LABEL}_eval.json"

    echo ""
    echo "=== Evaluating ${LABEL} ==="
    python3 scripts/eval_test4.py \
        --checkpoint "$CKPT" \
        --hdf5_path "$HDF5" \
        --output "$OUT" \
        --nside 128 \
        --batch_size 32 \
        --hidden_channels 32 \
        --num_blocks 3 || echo "FAILED: ${LABEL}"
done

echo ""
echo "=== Done ==="
ls -lh $RESULTS_DIR/test4_nside128_*_eval.json 2>/dev/null
date
