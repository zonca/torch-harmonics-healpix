#!/bin/bash
#SBATCH --job-name=thh-n16-v3
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-cosmic2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n16_v3_%j.out

export PYTHONUNBUFFERED=1

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
CAMB_CACHE=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/camb_cache_n16.fits

mkdir -p $RESULTS_DIR

echo "=== Fresh N16 training (v3) ==="
date
hostname

source $VENV/bin/activate
cd $REPO

# Train all 4 configs
for FSKY_NOISE in "1.0_0" "1.0_6" "0.1_0" "0.1_6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d_ -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d_ -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    OUTPUT=$RESULTS_DIR/test4_nside16_${LABEL}.json

    echo ""
    echo "=== Training $LABEL ==="
    echo "Config: --nside 16 --f_sky $FSKY --noise_std $NOISE"
    echo "Model: hidden_channels=32, num_blocks=3"
    echo "Output: $OUTPUT"
    echo "CAMB cache: $CAMB_CACHE"

    python3 scripts/train_test4.py \
        --nside 16 \
        --f_sky $FSKY \
        --noise_std $NOISE \
        --hidden_channels 32 \
        --num_blocks 3 \
        --batch_size 64 \
        --lr 1e-3 \
        --max_epochs 150 \
        --patience 20 \
        --scheduler cosine \
        --cosine_T_max 100 \
        --cosine_eta_min 1e-7 \
        --grad_clip 1.0 \
        --n_train 100000 \
        --n_val 10000 \
        --n_test 1000 \
        --camb_cache $CAMB_CACHE \
        --output $OUTPUT

    echo "=== $LABEL complete ==="
done

echo ""
echo "=== All N16 v3 training complete ==="
date
