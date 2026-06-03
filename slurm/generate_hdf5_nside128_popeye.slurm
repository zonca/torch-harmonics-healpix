#!/bin/bash
#SBATCH --job-name=thh-hdf5-xfer
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/hdf5_xfer_%j.out

# Regenerate HDF5 v2 with C_ℓ fix and transfer to Expanse via rsync
# Uses /tmp for local generation, rsync --partial for resumable transfers
# Deletes each file from /tmp after successful transfer to save space
# CPU only — Popeye

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== HDF5 v2 generation + rsync transfer ==="
date
hostname

source ~/torch-hh-venv/bin/activate
cd ~/torch-harmonics-healpix

NSIDE=128
LOCAL_DIR=/tmp/hdf5_nside128_$SLURM_JOB_ID
mkdir -p $LOCAL_DIR

REMOTE=zonca@login.expanse.sdsc.edu:/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2/

# Function: generate one config and rsync it
generate_and_transfer() {
    local FSKY=$1
    local NOISE=$2
    local LABEL="fsky${FSKY}_noise${NOISE}"
    local FILENAME="test4_nside128_${LABEL}.h5"

    echo ""
    echo "=== Config: f_sky=${FSKY}, noise=${NOISE} ==="
    python3 scripts/generate_test4_hdf5.py \
        --nside $NSIDE --f_sky ${FSKY} --noise_std ${NOISE} \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --output $LOCAL_DIR/${FILENAME}

    echo "Transferring ${FILENAME}..."
    rsync -av --partial --progress \
        -e 'ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10' \
        $LOCAL_DIR/${FILENAME} $REMOTE
    if [ $? -eq 0 ]; then
        echo "Transfer successful, cleaning up local file"
        rm -f $LOCAL_DIR/${FILENAME}
    else
        echo "Transfer FAILED for ${FILENAME} — local file kept in $LOCAL_DIR"
    fi
}

generate_and_transfer 1.0 0
generate_and_transfer 1.0 6
generate_and_transfer 0.1 0
generate_and_transfer 0.1 6

# Cleanup
rm -rf $LOCAL_DIR

echo "=== All HDF5 v2 generation + transfer complete ==="
date
