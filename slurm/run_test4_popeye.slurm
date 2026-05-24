#!/bin/bash
#SBATCH --job-name=thh-t4-baselines
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

# Fisher forecast + MCMC baselines for Test 4 (joint r/τ estimation).
# CPU-only (Popeye gen partition). Uses same FITS CAMB cache as Expanse training script.

set -e
export PYTHONUNBUFFERED=1

WORKDIR=~/torch-harmonics-healpix
VENV=$WORKDIR/.venv
RESULTS_DIR=$WORKDIR/results
REPO_SRC=$WORKDIR

cd $WORKDIR

# Always recreate venv to ensure all deps
echo "=== Creating venv ==="
date
rm -rf $VENV
uv venv $VENV --python 3.11
source $VENV/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
uv pip install numpy healpy astropy scipy matplotlib camb psutil 2>&1 | tail -3
uv pip install -e "$REPO_SRC" --no-deps 2>&1 | tail -1

echo ""
echo "=== Fisher Forecast (4 configs) ==="
date
python3 scripts/fisher_forecast.py

echo ""
echo "=== MCMC Baselines (4 configs, 1000 maps each) ==="
date
export RESULTS_DIR
python3 scripts/run_mcmc_test4.py

echo ""
echo "=== All done ==="
date
