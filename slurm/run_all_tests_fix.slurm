#!/bin/bash
#SBATCH --job-name=thh-all-fix
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/train_all_fix_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Full re-run with all bugfixes ==="
echo "=== Fixes: shared mask, inpainting, CAMB get_results, robust eval ==="
echo "=== Node info ==="
hostname
date
nvidia-smi

echo "=== Setting up venv on /tmp ==="
WORKDIR=/tmp/zonca_torch_hh_fix_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install torch-harmonics==0.8.0 --no-deps 2>&1 | tail -1
uv pip install healpy h5py scipy matplotlib camb 2>&1 | tail -1

echo "=== Copying repo ==="
REPO_SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/repo_latest
cp -r $REPO_SRC $WORKDIR/repo

echo "=== Pulling latest code (includes shared mask + CAMB + eval fixes) ==="
cd $WORKDIR/repo
git pull origin main 2>&1 | tail -5

echo "=== Installing torch-harmonics-healpix in dev mode ==="
uv pip install -e "$WORKDIR/repo[dev]" --no-deps 2>&1 | tail -1

echo "=== Package versions ==="
python3 -c "
import sys, torch, healpy, numpy, scipy
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'healpy: {healpy.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scipy: {scipy.__version__}')
try:
    import camb
    print(f'camb: {camb.__version__}')
except:
    print('camb: not available')
"

RESULTS_DIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results
mkdir -p $RESULTS_DIR

# =============================================
# Test 1: ℓ_peak from T-only maps (4 noise levels)
# =============================================
# Already completed with correct setup (no mask, no shared mask issue)
# Results from v2 (σ_p=5.0) are valid. Just re-confirm with current code.

for NOISE in 0 5 10 15; do
    echo ""
    echo "============================================"
    echo "=== Test 1: noise_std=$NOISE ==="
    echo "============================================"
    python3 scripts/train_test1_v2.py \
        --noise_std $NOISE \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --max_epochs 150 --patience 20 \
        --batch_size 32 --lr 1e-3 \
        --output $RESULTS_DIR/test1_v2_fix_noise${NOISE}.json
done

# =============================================
# Test 2: ℓ_Ep/ℓ_Bp from Q/U polarization maps
# =============================================
# Full sky (no mask, no inpainting needed)
echo ""
echo "============================================"
echo "=== Test 2: f_sky=1.0, noise_std=0 ==="
echo "============================================"
python3 scripts/train_test2_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 \
    --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test2_v2_fix_fsky1.0.json

# Partial sky with shared mask + inpainting
for FSKY in 0.5 0.2 0.1 0.05; do
    echo ""
    echo "============================================"
    echo "=== Test 2 (shared mask+inpaint): f_sky=$FSKY ==="
    echo "============================================"
    python3 scripts/train_test2_v2.py \
        --f_sky $FSKY --noise_std 0 \
        --n_train 100000 --n_val 10000 --n_test 1000 \
        --max_epochs 150 --patience 20 \
        --batch_size 32 --lr 1e-3 \
        --output $RESULTS_DIR/test2_v2_fix_fsky${FSKY}.json
done

# =============================================
# Test 3: τ estimation from Q/U maps
# =============================================
echo ""
echo "============================================"
echo "=== Test 3: τ estimation, full sky ==="
echo "============================================"
python3 scripts/train_test3_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 \
    --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test3_v2_fix.json

echo ""
echo "=== ALL TESTS COMPLETE ==="
date

# Print summary
echo ""
echo "=== Results Summary ==="
echo "--- Test 1 ---"
for NOISE in 0 5 10 15; do
    echo "noise = $NOISE:"
    cat $RESULTS_DIR/test1_v2_fix_noise${NOISE}.json 2>/dev/null || echo "  (not found)"
    echo ""
done
echo "--- Test 2 ---"
for FSKY in 1.0 0.5 0.2 0.1 0.05; do
    echo "f_sky = $FSKY:"
    cat $RESULTS_DIR/test2_v2_fix_fsky${FSKY}.json 2>/dev/null || echo "  (not found)"
    echo ""
done
echo "--- Test 3 ---"
cat $RESULTS_DIR/test3_v2_fix.json 2>/dev/null || echo "  (not found)"
