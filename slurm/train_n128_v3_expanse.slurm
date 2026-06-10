#!/bin/bash
#SBATCH --job-name=thh-n128-v3
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-cosmic2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n128_v3_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2

# Model config: hidden_channels=8, num_blocks=3 → ~30M params (vs 422M before)
# This is much smaller, should generalize better
# We add weight_decay and early stopping on val accuracy

mkdir -p $RESULTS_DIR

echo "=== Fresh N128 training (v3, small model) ==="
date
hostname

source $VENV/bin/activate
cd $REPO

# Train all 4 configs
for FSKY_NOISE in "1.0_0" "1.0_6" "0.1_0" "0.1_6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d_ -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d_ -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    HDF5_FILE=$HDF5_DIR/test4_nside128_${LABEL}.h5
    OUTPUT=$RESULTS_DIR/test4_nside128_${LABEL}.json

    echo ""
    echo "=== Training $LABEL (N128, small model) ==="
    echo "Config: --nside 128 --f_sky $FSKY --noise_std $NOISE"
    echo "Model: hidden_channels=8, num_blocks=3 (~30M params)"
    echo "Regularization: weight_decay=1e-4, grad_clip=1.0, patience=10"
    echo "HDF5: $HDF5_FILE"
    echo "Output: $OUTPUT"

    python3 scripts/train_test4.py \
        --nside 128 \
        --f_sky $FSKY \
        --noise_std $NOISE \
        --hidden_channels 8 \
        --num_blocks 3 \
        --batch_size 16 \
        --lr 1e-3 \
        --max_epochs 150 \
        --patience 10 \
        --scheduler cosine \
        --cosine_T_max 50 \
        --cosine_eta_min 1e-7 \
        --grad_clip 1.0 \
        --weight_decay 1e-4 \
        --hdf5_path $HDF5_FILE \
        --num_workers 0 \
        --cache_chunk_size 5000 \
        --output $OUTPUT

    echo "=== $LABEL complete ==="
done

echo ""
echo "=== All N128 v3 training complete ==="
date
