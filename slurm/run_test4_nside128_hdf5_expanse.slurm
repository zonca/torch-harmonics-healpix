#!/bin/bash
#SBATCH --job-name=thh-t4-hdf5
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test4_n128_hdf5_%j.out

# Test 4 NSIDE=128 with HDF5 pre-generated datasets (100K train maps)
# Account: sds275
# HDF5 mode eliminates the hp.synfast bottleneck — much faster epochs
# 422M params, hidden_channels=32, batch_size=2 (fits on V100 32GB)
# num_workers=4 (HDF5 mode supports multi-process DataLoader)

set -e
export PYTHONUNBUFFERED=1

echo "=== Test 4 NSIDE=128: Joint r/τ estimation (HDF5 mode) ==="
echo "=== Account: sds275, 100K pre-generated maps per config ==="
hostname
date
nvidia-smi

# Create venv in /tmp (fast local storage)
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
HDF5_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data
NSIDE=128
mkdir -p $RESULTS_DIR

# Config 1: f_sky=1.0, noise=0
echo ""
echo "=========================================="
echo "=== Config 1: f_sky=1.0, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 0 \
    --nside $NSIDE --lmax 383 \
    --hdf5_path $HDF5_DIR/test4_nside128_fsky1.0_noise0.h5 \
    --num_workers 4 \
    --max_epochs 150 --patience 20 --batch_size 2 --lr 1e-3 \
    --hidden_channels 32 --num_blocks 3 \
    --output $RESULTS_DIR/test4_nside128_hdf5_fsky1.0_noise0.json || echo "CONFIG 1 FAILED"

# Config 2: f_sky=1.0, noise=6 μK-arcmin
echo ""
echo "=========================================="
echo "=== Config 2: f_sky=1.0, noise=6 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 1.0 --noise_std 6 \
    --nside $NSIDE --lmax 383 \
    --hdf5_path $HDF5_DIR/test4_nside128_fsky1.0_noise6.h5 \
    --num_workers 4 \
    --max_epochs 150 --patience 20 --batch_size 2 --lr 1e-3 \
    --hidden_channels 32 --num_blocks 3 \
    --output $RESULTS_DIR/test4_nside128_hdf5_fsky1.0_noise6.json || echo "CONFIG 2 FAILED"

# Config 3: f_sky=0.1, noise=0
echo ""
echo "=========================================="
echo "=== Config 3: f_sky=0.1, noise=0 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 0 \
    --nside $NSIDE --lmax 383 \
    --hdf5_path $HDF5_DIR/test4_nside128_fsky0.1_noise0.h5 \
    --num_workers 4 \
    --max_epochs 150 --patience 20 --batch_size 2 --lr 1e-3 \
    --hidden_channels 32 --num_blocks 3 \
    --output $RESULTS_DIR/test4_nside128_hdf5_fsky0.1_noise0.json || echo "CONFIG 3 FAILED"

# Config 4: f_sky=0.1, noise=6 μK-arcmin
echo ""
echo "=========================================="
echo "=== Config 4: f_sky=0.1, noise=6 ==="
echo "=========================================="
python3 scripts/train_test4.py \
    --f_sky 0.1 --noise_std 6 \
    --nside $NSIDE --lmax 383 \
    --hdf5_path $HDF5_DIR/test4_nside128_fsky0.1_noise6.h5 \
    --num_workers 4 \
    --max_epochs 150 --patience 20 --batch_size 2 --lr 1e-3 \
    --hidden_channels 32 --num_blocks 3 \
    --output $RESULTS_DIR/test4_nside128_hdf5_fsky0.1_noise6.json || echo "CONFIG 4 FAILED"

echo ""
echo "=== All NSIDE=128 HDF5 training complete ==="
ls -lh $RESULTS_DIR/test4_nside128_hdf5_*.json $RESULTS_DIR/test4_nside128_hdf5_*.pt 2>/dev/null
date
