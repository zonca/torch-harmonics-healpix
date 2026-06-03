#!/bin/bash
#SBATCH --job-name=thh-t4-n128
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/retrain_n128_%j.out

# Phase 1: Generate HDF5 v2 data on Expanse Lustre
# Phase 2: Train CNN (GPU)
# All on one GPU node to avoid Popeye→Expanse data transfer issues

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== NSIDE=128 retrain with v2 C_ℓ fix ==="
date
hostname

# Setup venv
REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
DATA_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
mkdir -p $DATA_DIR $RESULTS_DIR

# Create venv if needed
if [ ! -d $VENV ]; then
    echo "Creating venv..."
    uv venv $VENV
fi
source $VENV/bin/activate

# Install packages
echo "Installing packages..."
uv pip install healpy camb torch-harmonics==0.8.0 --no-deps 2>/dev/null
uv pip install h5py scipy numpy matplotlib 2>/dev/null

cd $REPO

NSIDE=128

# =============================================
# Phase 1: Generate HDF5 v2 data
# =============================================
echo ""
echo "=== Phase 1: Generate HDF5 v2 data ==="
date

for FSKY_NOISE in "1.0_0" "1.0_6" "0.1_0" "0.1_6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d_ -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d_ -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    FILENAME="test4_nside128_${LABEL}.h5"

    if [ -f $DATA_DIR/$FILENAME ]; then
        echo "  $FILENAME already exists, skipping"
        continue
    fi

    echo ""
    echo "  Generating $LABEL..."
    python3 scripts/generate_test4_hdf5.py \
        --nside $NSIDE --f_sky $FSKY --noise_std $NOISE \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --output $DATA_DIR/$FILENAME
    if [ $? -ne 0 ]; then
        echo "  FAILED: $LABEL"
    else
        echo "  Done: $FILENAME"
    fi
done

echo "=== Phase 1 complete ==="
date

# =============================================
# Phase 2: Train CNN on each config
# =============================================
echo ""
echo "=== Phase 2: Train CNN ==="
date

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
    echo "  Training $LABEL..."
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
        --hdf5_path $HDF5_FILE

    echo "  Done: $LABEL"
done

echo "=== All NSIDE=128 retraining complete ==="
date
