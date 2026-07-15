#!/bin/bash
#SBATCH --job-name=thh-tests-cpu
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/tests_cpu_%j.out

# Full unit test suite on CPU (GPU tests auto-skip without CUDA).

set -e
export PYTHONUNBUFFERED=1

date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

python3 -m pytest tests/ -v --tb=short

echo "=== CPU test suite complete ==="
date
