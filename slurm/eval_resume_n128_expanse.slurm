#!/bin/bash
#SBATCH --job-name=thh-eval-n128
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/eval_resume_n128_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
DATA_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv

source $VENV/bin/activate
cd $REPO

echo "=== Evaluating resumed N128 best checkpoints ==="
date

for FSKY_NOISE in "1.0_0" "1.0_6" "0.1_0" "0.1_6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d_ -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d_ -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    HDF5_FILE=$DATA_DIR/test4_nside128_${LABEL}.h5
    CHECKPOINT=$RESULTS_DIR/test4_nside128_${LABEL}_resume_nside128_best.pt
    OUTPUT=$RESULTS_DIR/test4_nside128_${LABEL}_resume_eval.json

    echo ""
    echo "--- Evaluating $LABEL ---"
    python3 scripts/eval_test4.py \
        --checkpoint $CHECKPOINT \
        --hdf5_path $HDF5_FILE \
        --output $OUTPUT \
        --nside 128 \
        --hidden_channels 32 \
        --num_blocks 3 \
        --batch_size 32
    if [ $? -ne 0 ]; then
        echo "  FAILED: $LABEL"
    fi
done

echo ""
echo "=== Printing results ==="
for F in $RESULTS_DIR/test4_nside128_fsky*_resume_eval.json; do
    echo "--- $(basename $F) ---"
    python3 -c "import json; d=json.load(open('$F')); print(f\"  r_error: {d['r_pct_error']:.1f}% | tau_error: {d['tau_pct_error']:.1f}% | r_p50: {d['r_p50']:.1f}% | r_p68: {d['r_p68']:.1f}% | tau_p50: {d['tau_p50']:.1f}% | tau_p68: {d['tau_p68']:.1f}%\")"
done

echo ""
echo "=== All evaluations complete ==="
date
