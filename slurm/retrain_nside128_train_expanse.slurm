#!/bin/bash
#SBATCH --job-name=thh-t4-n128t
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/retrain_n128_train_%j.out

# Train CNN on NSIDE=128 using pre-generated HDF5 v2 data
# Run AFTER HDF5 generation is complete

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== NSIDE=128 CNN training (v2 C_ℓ fix) ==="
date
hostname

WORKDIR=/tmp/zonca_torch_hh_retrain_n128t_$SLURM_JOB_ID
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

DATA_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2
mkdir -p $RESULTS_DIR

NSIDE=128

for FSKY_NOISE in "1.0_0" "1.0_6" "0.1_0" "0.1_6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d_ -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d_ -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    HDF5_FILE=$DATA_DIR/test4_nside128_${LABEL}.h5
    OUTPUT=$RESULTS_DIR/test4_nside128_${LABEL}

    if [ ! -f $HDF5_FILE ]; then
        echo "  CONFIG $LABEL: HDF5 missing, skipping"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "=== Training $LABEL ==="
    echo "=========================================="
    python3 scripts/train_test4.py \
        --nside $NSIDE \
        --f_sky $FSKY \
        --noise_std $NOISE \
        --batch_size 16 \
        --epochs 25 \
        --hidden_channels 32 \
        --num_blocks 3 \
        --cosine_T_max 25 \
        --output $OUTPUT \
        --hdf5_path $HDF5_FILE || echo "CONFIG $LABEL FAILED"
done

echo ""
echo "=== All NSIDE=128 training complete ==="
ls -lh $RESULTS_DIR/test4_nside128_*.json 2>/dev/null
date
