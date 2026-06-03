#!/bin/bash
#SBATCH --job-name=thh-gen-hdf5
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/gen_hdf5_n128_%j.out

# Regenerate NSIDE=128 HDF5 datasets with corrected C_ℓ (raw_cl=True)
# Generate to /tmp (3.4TB local disk), then scp to Expanse Lustre
# CPU only — Popeye

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== Regenerating HDF5 datasets (C_ℓ fix) ==="
date
hostname

source ~/torch-hh-venv/bin/activate
cd ~/torch-harmonics-healpix

NSIDE=128
LOCAL_DIR=/tmp/hdf5_nside128_$SLURM_JOB_ID
mkdir -p $LOCAL_DIR

EXSPANSE_LUSTRE=zonca@login.expanse.sdsc.edu:/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2

# Config 1: fsky=1.0, noise=0
echo ""
echo "=== Config 1: fsky=1.0, noise=0 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --output $LOCAL_DIR/test4_nside128_fsky1.0_noise0.h5
scp $LOCAL_DIR/test4_nside128_fsky1.0_noise0.h5 $EXSPANSE_LUSTRE/ || echo "SCP failed for cfg1"

# Config 2: fsky=1.0, noise=6
echo ""
echo "=== Config 2: fsky=1.0, noise=6 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --f_sky 1.0 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --output $LOCAL_DIR/test4_nside128_fsky1.0_noise6.h5
scp $LOCAL_DIR/test4_nside128_fsky1.0_noise6.h5 $EXSPANSE_LUSTRE/ || echo "SCP failed for cfg2"

# Config 3: fsky=0.1, noise=0
echo ""
echo "=== Config 3: fsky=0.1, noise=0 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --f_sky 0.1 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --output $LOCAL_DIR/test4_nside128_fsky0.1_noise0.h5
scp $LOCAL_DIR/test4_nside128_fsky0.1_noise0.h5 $EXSPANSE_LUSTRE/ || echo "SCP failed for cfg3"

# Config 4: fsky=0.1, noise=6
echo ""
echo "=== Config 4: fsky=0.1, noise=6 ==="
python3 scripts/generate_test4_hdf5.py \
    --nside $NSIDE --f_sky 0.1 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --output $LOCAL_DIR/test4_nside128_fsky0.1_noise6.h5
scp $LOCAL_DIR/test4_nside128_fsky0.1_noise6.h5 $EXSPANSE_LUSTRE/ || echo "SCP failed for cfg4"

# Cleanup
rm -rf $LOCAL_DIR

echo "=== HDF5 generation + transfer complete ==="
date
