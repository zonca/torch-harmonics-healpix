#!/bin/bash
#SBATCH --job-name=debug-sht
#SBATCH --partition=gpu-shared
#SBATCH --account=sds166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/debug_sht.out

module load gpu 2>/dev/null

VENV=/tmp/zonca_debug_sht
python3 -m venv $VENV 2>/dev/null
source $VENV/bin/activate
uv pip install torch-harmonics 2>/dev/null

python3 << 'PYEOF'
import torch
from torch_harmonics import RealSHT, InverseRealSHT

nlat, nlon, lmax, mmax = 32, 64, 47, 32
sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")
print(f"Requested: lmax={lmax}, mmax={mmax}")
print(f"SHT lmax attr={sht.lmax}, mmax attr={sht.mmax}")
print(f"SHT weights shape: {sht.weights.shape}")

x = torch.randn(1, 1, nlat, nlon)
try:
    c = sht(x)
    print(f"Coeff shape: {c.shape}")
except Exception as e:
    print(f"ERROR: {e}")

# Try with lmax = nlat - 1 = 31
print("\n--- Trying lmax = nlat-1 = 31 ---")
lmax2 = nlat - 1
mmax2 = min(lmax2, nlon // 2)
sht2 = RealSHT(nlat, nlon, lmax=lmax2, mmax=mmax2, grid="equiangular")
print(f"Requested: lmax={lmax2}, mmax={mmax2}")
print(f"SHT lmax attr={sht2.lmax}, mmax attr={sht2.mmax}")
print(f"SHT weights shape: {sht2.weights.shape}")
x2 = torch.randn(1, 1, nlat, nlon)
try:
    c2 = sht2(x2)
    print(f"Coeff shape: {c2.shape}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")

# Also try the default lmax
print("\n--- Trying default lmax ---")
sht3 = RealSHT(nlat, nlon, grid="equiangular")
print(f"SHT lmax attr={sht3.lmax}, mmax attr={sht3.mmax}")
print(f"SHT weights shape: {sht3.weights.shape}")
x3 = torch.randn(1, 1, nlat, nlon)
try:
    c3 = sht3(x3)
    print(f"Coeff shape: {c3.shape}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
PYEOF
