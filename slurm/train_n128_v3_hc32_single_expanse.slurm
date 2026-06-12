#!/bin/bash
#SBATCH --job-name=thh-n128-hc32
#SBATCH --partition=gpu-shared
#SBATCH --account=sds275
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/logs/train_n128_v3_hc32_%j.out
#SBATCH --export=ALL

set -e
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=== N128 training (v3, hidden_channels=32, 422M params) ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Setup
source /etc/profile.d/modules.sh
module load gpu/0.17.4
module load python/3.11.5
uv venv /expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv
source /expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/venv/bin/activate

REPO=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
HDF5=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2
RESULTS=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3

# Parse config from job name or use argument
FSKY=${1:-1.0}
NOISE=${2:-0}
LABEL="fsky${FSKY}_noise${NOISE}"

echo "=== Training ${LABEL} (N128, hc=32) ==="
echo "Config: --nside 128 --f_sky ${FSKY} --noise_std ${NOISE}"
echo "Model: hidden_channels=32, num_blocks=3 (~422M params)"
echo "Loss: MSE(r) + Huber(τ, delta=0.01)"
echo "HDF5: ${HDF5}/test4_nside128_${LABEL}.h5"
echo "Output: ${RESULTS}/test4_nside128_hc32_${LABEL}.json"
echo "Device: cuda"

python3 ${REPO}/scripts/train_test4.py \
    --f_sky ${FSKY} --noise_std ${NOISE} \
    --nside 128 --lmax 383 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 25 --patience 10 --batch_size 2 --lr 1e-3 \
    --hidden_channels 32 --num_blocks 3 \
    --cosine_T_max 25 \
    --hdf5_path ${HDF5}/test4_nside128_${LABEL}.h5 \
    --camb_cache ${RESULTS}/test4_camb_spectra_nside128.fits \
    --output ${RESULTS}/test4_nside128_hc32_${LABEL}.json

echo "=== Training ${LABEL} complete ==="
date
