#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --partition=gpu-shared
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH --account=sds166
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/test_gpu_%j.out

set -e

WORKDIR=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix
VENV=/tmp/zonca_torch_hh/.venv

echo "=== Node info ==="
hostname
nvidia-smi | head -12

echo "=== Setting up venv on /tmp ==="
mkdir -p /tmp/zonca_torch_hh
cd /tmp/zonca_torch_hh
UV_LINK_MODE=copy uv venv .venv --python 3.11
source .venv/bin/activate

echo "=== Installing packages ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-harmonics==0.8.0 --no-deps
uv pip install healpy h5py scipy matplotlib

echo "=== Testing PyTorch GPU ==="
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(1000, 1000).cuda()
    y = x @ x
    print(f"GPU tensor ops: OK")
else:
    print("ERROR: CUDA not available!")
PYEOF

echo "=== Testing torch-harmonics ==="
python3 << 'PYEOF'
import torch
import torch_harmonics as th
print(f"torch-harmonics version: {th.__version__}")
nlat, nlon = 32, 64
sht = th.RealSHT(nlat, nlon, lmax=31, grid="equiangular")
isht = th.InverseRealSHT(nlat, nlon, lmax=31, grid="equiangular")
x = torch.randn(2, nlat, nlon)
if torch.cuda.is_available():
    sht = sht.cuda()
    isht = isht.cuda()
    x = x.cuda()
alm = sht(x)
x_rec = isht(alm)
print(f"SHT forward: OK (input {x.shape} -> output {alm.shape})")
print(f"SHT inverse: OK (output {x_rec.shape})")
print(f"GPU SHT: OK")
PYEOF

echo "=== Testing healpy ==="
python3 << 'PYEOF'
import healpy as hp
import numpy as np
print(f"healpy version: {hp.__version__}")
cl = np.ones(48)
m = hp.synfast(cl, nside=16, lmax=47, verbose=False)
print(f"healpy synfast: OK (Nside=16, Npix={len(m)})")
cl_out = hp.anafast(m, lmax=47)
print(f"healpy anafast: OK (Cl shape: {cl_out.shape})")
PYEOF

echo "=== Testing HEALPix -> Equiangular resampling ==="
python3 << 'PYEOF'
import torch
import torch_harmonics as th
import healpy as hp
import numpy as np

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
nlat, nlon = 32, 64

# Create a simple HEALPix map
cl = np.zeros(48); cl[10] = 1.0
healpix_map = hp.synfast(cl, nside=NSIDE, lmax=47, verbose=False, random_seed=42)

# Convert to equiangular via ang2pix
lats = np.linspace(np.pi/2, -np.pi/2, nlat, endpoint=False)
lons = np.linspace(0, 2*np.pi, nlon, endpoint=False)
lon_grid, lat_grid = np.meshgrid(lons, lats)
pixel_indices = hp.ang2pix(NSIDE, lat_grid.ravel(), lon_grid.ravel(), lonlat=False)
equi_map = healpix_map[pixel_indices].reshape(nlat, nlon)

# SHT on equiangular
sht = th.RealSHT(nlat, nlon, lmax=47, grid="equiangular")
if torch.cuda.is_available():
    equi_tensor = torch.from_numpy(equi_map).float().cuda().unsqueeze(0)
    sht = sht.cuda()
else:
    equi_tensor = torch.from_numpy(equi_map).float().unsqueeze(0)

alm = sht(equi_tensor)
print(f"HEALPix -> Equiangular -> SHT: OK (alm shape: {alm.shape})")
print(f"Integration test: PASSED")
PYEOF

echo "=== All tests complete ==="
