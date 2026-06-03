#!/bin/bash
#SBATCH --job-name=thh-n128-cfg4
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/gen_cfg4_%j.out

# Generate only cfg4 (fsky=0.1, noise=6) HDF5
# Use GPU node to avoid Popeye->Expanse transfer issues

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== N128 cfg4 HDF5 generation ==="
date
hostname

VENV=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
DATA_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
mkdir -p $DATA_DIR

if [ -f $VENV/bin/python3 ]; then
    source $VENV/bin/activate
else
    echo "No venv found — creating..."
    uv venv $VENV
    source $VENV/bin/activate
    uv pip install astropy healpy camb h5py scipy numpy matplotlib 2>/dev/null
    uv pip install torch-harmonics==0.8.0 --no-deps 2>/dev/null
    cd $REPO && uv pip install -e . 2>/dev/null
fi

cd $REPO
git pull origin main 2>&1 | tail -2

python3 scripts/generate_test4_hdf5.py \
    --nside 128 --f_sky 0.1 --noise_std 6 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --output $DATA_DIR/test4_nside128_fsky0.1_noise6.h5

echo "=== cfg4 generation complete ==="
date
ls -lh $DATA_DIR/test4_nside128_fsky0.1_noise6.h5
