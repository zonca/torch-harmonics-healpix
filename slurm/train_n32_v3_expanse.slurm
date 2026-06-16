#!/bin/bash
#SBATCH --job-name=thh-n32
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n32_v3_%j.out

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_n32

# Optional: run single config via sbatch args (default: all 4)
FSKY=${1:-all}
NOISE=${2:-all}

mkdir -p $RESULTS_DIR $HDF5_DIR

echo "=== N32 pipeline: Fisher + HDF5 gen + training (v3, hc=32) ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source $VENV/bin/activate
cd $REPO

NSIDE=32
HIDDEN_CHANNELS=32
NUM_BLOCKS=3
BATCH_SIZE=16
MAX_EPOCHS=50
PATIENCE=15

if [ "$FSKY" = "all" ]; then
    CONFIGS="1.0:0 1.0:6 0.1:0 0.1:6"
else
    CONFIGS="${FSKY}:${NOISE}"
fi

for PAIR in $CONFIGS; do
    FSKY=$(echo $PAIR | cut -d: -f1)
    NOISE=$(echo $PAIR | cut -d: -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    HDF5_FILE=$HDF5_DIR/test4_nside32_${LABEL}.h5
    FISHER_JSON=$RESULTS_DIR/test4_fisher_nside32_${LABEL}.json
    CNN_JSON=$RESULTS_DIR/test4_nside32_hc32_${LABEL}.json

    echo ""
    echo "================================================================"
    echo "=== Config: $LABEL (NSIDE=$NSIDE) ==="
    echo "================================================================"

    # Step 1: Fisher forecast (fast, ~30s)
    if [ ! -f "$FISHER_JSON" ]; then
        echo "--- Step 1: Fisher forecast ---"
        date
        python3 scripts/fisher_forecast.py \
            --nside $NSIDE --f_sky $FSKY --noise_std $NOISE \
            --output $FISHER_JSON
        echo "Fisher saved: $FISHER_JSON"
    else
        echo "--- Step 1: Fisher already exists: $FISHER_JSON ---"
    fi

    # Step 2: Generate HDF5 data (~20 min)
    if [ ! -f "$HDF5_FILE" ]; then
        echo "--- Step 2: Generating HDF5 data ---"
        date
        python3 scripts/generate_test4_hdf5.py \
            --nside $NSIDE --f_sky $FSKY --noise_std $NOISE \
            --n_train 100000 --n_val 10000 --n_test 1000 \
            --output $HDF5_FILE
        echo "HDF5 saved: $HDF5_FILE"
    else
        echo "--- Step 2: HDF5 already exists: $HDF5_FILE ---"
    fi

    # Step 3: Train CNN (~2-3h)
    if [ ! -f "$CNN_JSON" ]; then
        echo "--- Step 3: Training CNN (hc=32, 422M params) ---"
        date
        python3 scripts/train_test4.py \
            --nside $NSIDE \
            --f_sky $FSKY \
            --noise_std $NOISE \
            --hidden_channels $HIDDEN_CHANNELS \
            --num_blocks $NUM_BLOCKS \
            --batch_size $BATCH_SIZE \
            --lr 1e-3 \
            --max_epochs $MAX_EPOCHS \
            --patience $PATIENCE \
            --scheduler cosine \
            --cosine_T_max $MAX_EPOCHS \
            --cosine_eta_min 1e-7 \
            --grad_clip 1.0 \
            --weight_decay 1e-4 \
            --hdf5_path $HDF5_FILE \
            --num_workers 0 \
            --output $CNN_JSON
        echo "CNN results saved: $CNN_JSON"
    else
        echo "--- Step 3: CNN results already exist: $CNN_JSON ---"
    fi

    echo "=== Config $LABEL complete ==="
    date
done

echo ""
echo "================================================================"
echo "=== All N32 configs complete ==="
echo "================================================================"
date

# Print summary
echo ""
echo "=== SUMMARY ==="
python3 -c "
import json, glob, os
print(f\"{'Config':<25} {'CNN r%':>8} {'Fisher r%':>10} {'CNN/Fisher':>10} {'CNN τ%':>8} {'Fisher τ%':>10}\")
print('-' * 75)
for f in sorted(glob.glob('$RESULTS_DIR/test4_nside32_hc32_fsky*.json')):
    d = json.load(open(f))
    label = os.path.basename(f).replace('test4_nside32_hc32_','').replace('.json','')
    fisher_f = f.replace('_nside32_hc32_', '_nside32_').replace('test4_nside32_', 'test4_fisher_nside32_')
    try:
        fd = json.load(open(fisher_f))
        fisher_r = fd.get('r_fisher_pct', fd.get('r_pct_error', 0))
        fisher_tau = fd.get('tau_fisher_pct', fd.get('tau_pct_error', 0))
    except:
        fisher_r = 0
        fisher_tau = 0
    ratio = d['r_pct_error'] / fisher_r if fisher_r > 0 else float('inf')
    print(f'{label:<25} {d[\"r_pct_error\"]:>7.1f}% {fisher_r:>9.1f}% {ratio:>9.2f}x {d[\"tau_pct_error\"]:>7.1f}% {fisher_tau:>9.1f}%')
"
