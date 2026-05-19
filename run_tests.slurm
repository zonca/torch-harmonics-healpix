#!/bin/bash
#SBATCH --job-name=torch-hh-test
#SBATCH --partition=gpu-shared
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH --account=sds166
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/run_tests_%j.out

set -e

# === Configuration ===
WORKDIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix
REPO_DIR=$WORKDIR/repo
VENV=/tmp/zonca_torch_hh_test/.venv

echo "=== Node info ==="
hostname
date
nvidia-smi | head -12

echo "=== Setting up venv on /tmp ==="
rm -rf /tmp/zonca_torch_hh_test
mkdir -p /tmp/zonca_torch_hh_test
cd /tmp/zonca_torch_hh_test
UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-harmonics==0.8.0 --no-deps
uv pip install healpy h5py scipy matplotlib pytest

echo "=== Copying repo ==="
# Copy repo to /tmp for fast I/O during tests
cp -r $REPO_DIR /tmp/zonca_torch_hh_test/repo
cd /tmp/zonca_torch_hh_test/repo

echo "=== Installing torch-harmonics-healpix in dev mode ==="
uv pip install -e ".[dev]"

echo "=== Running unit tests (CPU) ==="
python -m pytest tests/test_resample.py -v 2>&1 || echo "WARN: resample tests had issues"
python -m pytest tests/test_data_generation.py -v 2>&1 || echo "WARN: data generation tests had issues"
python -m pytest tests/test_mcmc_baseline.py -v 2>&1 || echo "WARN: MCMC baseline tests had issues"

echo "=== Running model test (GPU) ==="
python -m pytest tests/test_model_gpu.py -v 2>&1 || echo "WARN: model GPU test had issues"

echo "=== Running MCMC baseline benchmark ==="
python scripts/run_mcmc_benchmark.py 2>&1 || echo "WARN: MCMC benchmark had issues"

echo "=== All tests complete ==="
date
