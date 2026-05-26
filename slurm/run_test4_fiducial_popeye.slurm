#!/bin/bash
#SBATCH --job-name=thh-t4-fiducial
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Evaluate SpectralCNN at Fisher fiducial point (r=0.003, τ=0.054)
# for direct apples-to-apples comparison with Fisher forecast.
# CPU-only (Popeye gen partition).

set -e
export PYTHONUNBUFFERED=1

WORKDIR=~/torch-harmonics-healpix
VENV=$WORKDIR/.venv
REPO_SRC=$WORKDIR

cd $WORKDIR

# Recreate venv
echo "=== Creating venv ==="
date
rm -rf $VENV
uv venv $VENV --python 3.11
source $VENV/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
uv pip install numpy healpy astropy scipy matplotlib camb "torch-harmonics==0.8.0" 2>&1 | tail -3
uv pip install -e "$REPO_SRC" --no-deps 2>&1 | tail -1

# Copy trained models from results dir (if not already there)
# Models were already downloaded to results/ by previous jobs

echo ""
echo "=== Evaluating CNN at fiducial point ==="
date
python3 scripts/eval_test4_at_fiducial.py --results_dir results --n_realizations 1000

echo ""
echo "=== Done ==="
date
