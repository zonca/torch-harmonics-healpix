#!/bin/bash
#SBATCH --job-name=thh-t4-cfg2
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_n128_hdf5_cfg2_%j.out

# Test 4 NSIDE=128 Config 2: f_sky=1.0, noise=6
# HDF5 mode, 100K maps, batch_size=16, 422M params

set -e
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== Test 4 Config 2: f_sky=1.0, noise=6 (HDF5) ==="
hostname
date
nvidia-smi

WORKDIR=/tmp/zonca_torch_hh_t4_hdf5_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb astropy 2>&1 | tail -1

REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cd $REPO_SRC
git pull origin main 2>&1 | tail -3
uv pip install -e "$REPO_SRC[dev]" --no-deps 2>&1 | tail -1

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_striped
mkdir -p $RESULTS_DIR

python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --nside 128 --lmax 383 \
    --hdf5_path $HDF5_DIR/test4_nside128_fsky1.0_noise6.h5 \
    --num_workers 0 \
    --cache_chunk_size 5000 \
    --max_epochs 150 --patience 20 --batch_size 16 --lr 1e-3 \
    --hidden_channels 32 --num_blocks 3 \
    --output $RESULTS_DIR/test4_nside128_hdf5_fsky1.0_noise6.json

echo "=== Config 2 complete ==="
ls -lh $RESULTS_DIR/test4_nside128_hdf5_fsky1.0_noise6.* 2>/dev/null
date
