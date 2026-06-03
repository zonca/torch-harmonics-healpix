#!/bin/bash
#SBATCH --job-name=thh-t4-retrain
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/retrain_n128_%j.out

# Retrain Test 4 NSIDE=128 with corrected C_ℓ data (v2)
# Uses HDF5 v2 datasets (generated with raw_cl=True)
# 422M params, hidden_channels=32, batch_size=16
# Huber loss for τ (delta=0.01), CosineAnnealingLR T_max=25

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Retrain Test 4 NSIDE=128 (C_ℓ v2 data) ==="
date
hostname
nvidia-smi

# Create venv in /tmp
WORKDIR=/tmp/zonca_torch_hh_retrain_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO_SRC
git pull origin main 2>&1 | tail -3
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
NSIDE=128
mkdir -p $RESULTS_DIR

# Train all 4 configs sequentially
for CFG in "1.0 0" "1.0 6" "0.1 0" "0.1 6"; do
    FSKY=$(echo $CFG | cut -d' ' -f1)
    NOISE=$(echo $CFG | cut -d' ' -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    echo ""
    echo "=========================================="
    echo "=== Config: f_sky=${FSKY}, noise=${NOISE} ==="
    echo "=========================================="
    python3 scripts/train_test4.py \
        --f_sky ${FSKY} --noise_std ${NOISE} \
        --nside $NSIDE --lmax 383 \
        --hdf5_path $HDF5_DIR/test4_nside128_fsky${FSKY}_noise${NOISE}.h5 \
        --num_workers 0 \
        --max_epochs 150 --patience 20 --batch_size 16 --lr 1e-3 \
        --hidden_channels 32 --num_blocks 3 \
        --cosine_T_max 25 \
        --output $RESULTS_DIR/test4_nside128_${LABEL}.json || echo "CONFIG ${LABEL} FAILED"
done

echo ""
echo "=== All NSIDE=128 retraining complete ==="
ls -lh $RESULTS_DIR/test4_nside128_*.json $RESULTS_DIR/test4_nside128_*.pt 2>/dev/null
date
