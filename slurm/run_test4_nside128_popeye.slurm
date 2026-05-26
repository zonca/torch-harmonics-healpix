#!/bin/bash
#SBATCH --job-name=thh-t4-n128-cpu
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

# Test 4 baselines at NSIDE=128:
# 1. Fisher forecast (fast, CPU)
# 2. MCMC baselines (slow, CPU — larger maps at NSIDE=128)
# 3. Fiducial-point CNN evaluation (after models arrive from Expanse)

set -e
export PYTHONUNBUFFERED=1

WORKDIR=~/torch-harmonics-healpix
VENV=$WORKDIR/.venv-n128
REPO_SRC=$WORKDIR
NSIDE=128
LMAX=383
RESULTS=$WORKDIR/results

cd $WORKDIR

# Create fresh venv for NSIDE=128
echo "=== Creating venv ==="
date
rm -rf $VENV
uv venv $VENV --python 3.11
source $VENV/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
uv pip install numpy healpy astropy scipy matplotlib camb "torch-harmonics==0.8.0" psutil 2>&1 | tail -3
uv pip install -e "$REPO_SRC" --no-deps 2>&1 | tail -1

echo ""
echo "=== Fisher Forecast NSIDE=128 (4 configs) ==="
date
python3 scripts/fisher_forecast.py --f_sky 1.0 --noise_std 0 --nside $NSIDE --lmax $LMAX --output $RESULTS/test4_fisher_nside128_fsky1.0_noise0.json
python3 scripts/fisher_forecast.py --f_sky 1.0 --noise_std 6 --nside $NSIDE --lmax $LMAX --output $RESULTS/test4_fisher_nside128_fsky1.0_noise6.json
python3 scripts/fisher_forecast.py --f_sky 0.1 --noise_std 0 --nside $NSIDE --lmax $LMAX --output $RESULTS/test4_fisher_nside128_fsky0.1_noise0.json
python3 scripts/fisher_forecast.py --f_sky 0.1 --noise_std 6 --nside $NSIDE --lmax $LMAX --output $RESULTS/test4_fisher_nside128_fsky0.1_noise6.json

echo ""
echo "=== MCMC Baselines NSIDE=128 (4 configs, 1000 test maps each) ==="
date
python3 scripts/run_mcmc_test4.py --nside $NSIDE --lmax $LMAX --results_dir $RESULTS

echo ""
echo "=== Fisher and MCMC baselines complete ==="
date

# Wait for CNN models from Expanse (check every 10 min, up to 8 hours)
echo ""
echo "=== Waiting for CNN models from Expanse ==="
MODELS="test4_nside128_fsky1.0_noise0_nside128.pt test4_nside128_fsky1.0_noise6_nside128.pt test4_nside128_fsky0.1_noise0_nside128.pt test4_nside128_fsky0.1_noise6_nside128.pt"
WAITED=0
MAX_WAIT=28800  # 8 hours

while true; do
    ALL_FOUND=true
    for m in $MODELS; do
        if [ ! -f "$RESULTS/$m" ]; then
            ALL_FOUND=false
            break
        fi
    done

    if $ALL_FOUND; then
        echo "All CNN models found!"
        break
    fi

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Timed out waiting for CNN models after 8 hours. Exiting."
        exit 0
    fi

    sleep 600
    WAITED=$((WAITED + 600))
    echo "  Still waiting for CNN models... ($(($WAITED / 60)) min elapsed)"
done

echo ""
echo "=== Evaluating CNN at fiducial point (NSIDE=128) ==="
date
python3 scripts/eval_test4_at_fiducial.py --results_dir $RESULTS --nside $NSIDE --lmax $LMAX --n_realizations 1000

echo ""
echo "=== All NSIDE=128 baselines complete ==="
date
