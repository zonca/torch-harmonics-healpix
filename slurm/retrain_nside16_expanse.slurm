#!/bin/bash
#SBATCH --job-name=thh-t4-n16
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/retrain_n16_%j.out

# Retrain Test 4 NSIDE=16 with corrected C_ℓ (raw_cl=True)
# On-the-fly generation (no HDF5 needed for N16 — fast hp.synfast)
# Huber loss for τ, CosineAnnealingLR T_max=25

export PYTHONUNBUFFERED=1

echo "=== Retrain Test 4 NSIDE=16 (C_ℓ fix) ==="
date
hostname
nvidia-smi

WORKDIR=/tmp/zonca_torch_hh_retrain_n16_$SLURM_JOB_ID
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
mkdir -p $RESULTS_DIR

# Train all 4 configs
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
        --nside 16 --lmax 47 \
        --max_epochs 150 --patience 20 --batch_size 64 --lr 1e-3 \
        --hidden_channels 32 --num_blocks 3 \
        --huber_tau_delta 0.01 --cosine_T_max 25 \
        --output $RESULTS_DIR/test4_nside16_${LABEL}.json || echo "CONFIG ${LABEL} FAILED"
done

echo ""
echo "=== All NSIDE=16 retraining complete ==="
ls -lh $RESULTS_DIR/test4_nside16_*.json $RESULTS_DIR/test4_nside16_*.pt 2>/dev/null
date
