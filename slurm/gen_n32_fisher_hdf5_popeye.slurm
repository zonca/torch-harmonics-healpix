#!/bin/bash
#SBATCH --job-name=thh-n32-prep
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/gen_n32_%j.out

# N32 preparation: Fisher forecasts + HDF5 generation
# CPU only — runs on Popeye
# After this completes, transfer HDF5 to Expanse and run CNN training there

set -e
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== N32 prep: Fisher + HDF5 generation ==="
date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

RESULTS_DIR=~/torch-harmonics-healpix/results_v3
HDF5_DIR=~/torch-harmonics-healpix/hdf5_n32
mkdir -p $RESULTS_DIR $HDF5_DIR

NSIDE=32
LMAX=95
N_TRAIN=100000
N_VAL=10000
N_TEST=1000

# --- Step 1: Fisher forecasts (fast, ~1 min each) ---
echo ""
echo "================================================================"
echo "=== Step 1: Fisher forecasts ==="
echo "================================================================"
date

for FSKY_NOISE in "1.0:0" "1.0:6" "0.1:0" "0.1:6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d: -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d: -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    FISHER_JSON=$RESULTS_DIR/test4_fisher_nside32_${LABEL}.json

    if [ ! -f "$FISHER_JSON" ]; then
        echo "--- Fisher: $LABEL ---"
        python3 scripts/fisher_forecast.py \
            --nside $NSIDE --f_sky $FSKY --noise_std $NOISE \
            --output $FISHER_JSON
    else
        echo "--- Fisher already exists: $LABEL ---"
    fi
done

# --- Step 2: HDF5 generation (~20 min per config) ---
echo ""
echo "================================================================"
echo "=== Step 2: HDF5 generation ==="
echo "================================================================"
date

for FSKY_NOISE in "1.0:0" "1.0:6" "0.1:0" "0.1:6"; do
    FSKY=$(echo $FSKY_NOISE | cut -d: -f1)
    NOISE=$(echo $FSKY_NOISE | cut -d: -f2)
    LABEL="fsky${FSKY}_noise${NOISE}"
    HDF5_FILE=$HDF5_DIR/test4_nside32_${LABEL}.h5

    if [ ! -f "$HDF5_FILE" ]; then
        echo "--- Generating: $LABEL ---"
        date
        python3 scripts/generate_test4_hdf5.py \
            --nside $NSIDE --lmax $LMAX \
            --f_sky $FSKY --noise_std $NOISE \
            --n_train $N_TRAIN --n_val $N_VAL --n_test $N_TEST \
            --output $HDF5_FILE
    else
        echo "--- HDF5 already exists: $LABEL ---"
    fi
done

echo ""
echo "================================================================"
echo "=== All N32 prep complete ==="
echo "================================================================"
date
ls -lh $HDF5_DIR/test4_nside32_*.h5

echo ""
echo "=== NEXT: Transfer HDF5 to Expanse and run CNN training ==="
echo "  scp $HDF5_DIR/test4_nside32_*.h5 expanse:/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_n32/"
