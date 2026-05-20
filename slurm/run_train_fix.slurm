#!/bin/bash
#SBATCH --job-name=thh-fix
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/train_fix_%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Fix: shared mask + robust evaluation ==="
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

echo "=== Pulling latest code (shared mask fix) ==="
cd $WORKDIR/repo
git pull origin main 2>&1 | tail -3

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
# Test 2 (inpaint + shared mask): partial sky
# =============================================
# f_sky=1.0 already good (1.5%/1.6%, no mask needed)
# f_sky=0.5 was 3.6%/3.0% with inpaint (but different masks)
# Re-run all partial-sky with shared mask fix

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
# Test 3: τ estimation (shared mask + CAMB fix)
# =============================================

echo ""
echo "============================================"
echo "=== Test 3 (shared mask): τ estimation, full sky ==="
echo "============================================"
python3 scripts/train_test3_v2.py \
    --f_sky 1.0 --noise_std 0 \
    --n_train 100000 --n_val 10000 --n_test 1000 \
    --max_epochs 150 --patience 20 \
    --batch_size 32 --lr 1e-3 \
    --output $RESULTS_DIR/test3_v2_fix.json

echo ""
echo "=== All fix runs complete ==="
date

# Print summary
echo ""
echo "=== Results Summary ==="
for FSKY in 0.5 0.2 0.1 0.05; do
    echo "f_sky = $FSKY:"
    cat $RESULTS_DIR/test2_v2_fix_fsky${FSKY}.json 2>/dev/null || echo "  (not found)"
    echo ""
done
echo "--- Test 3 ---"
cat $RESULTS_DIR/test3_v2_fix.json 2>/dev/null || echo "  (not found)"
