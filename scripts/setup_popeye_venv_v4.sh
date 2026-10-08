#!/bin/bash
# Build the v4 venv on Popeye (run on login node: pip-only, no compute).
# Stack: torch 2.11.0+cu130, torch-harmonics 0.9.2, healpy 1.20.1,
#        numpy, scipy, camb, pytest — matches the locally-verified pairing.
set -e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.11.11 2>/dev/null || module load python/3.11

VENV=~/torch-hh-v4-venv
rm -rf $VENV
python3 -m venv $VENV
source $VENV/bin/activate
python3 -m pip install --upgrade pip

# torch cu130 (A100 sm_80 supported; V100 sm_70 NOT — use A100 nodes)
python3 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# torch-harmonics 0.9.2 requires torch>=2.11,<2.12 (the old ABI pin is obsolete)
python3 -m pip install torch-harmonics==0.9.2
python3 -m pip install healpy==1.20.1 numpy scipy camb pytest astropy

echo "=== installed ==="
python3 -c "
import torch, torch_harmonics, healpy, numpy, scipy, camb, sys
print('Python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('torch-harmonics:', torch_harmonics.__version__)
print('healpy:', healpy.__version__)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('camb:', camb.__version__)
import torch_harmonics as th
sht = th.RealSHT(16, 32)
print('SHT construct OK')
"
echo "VENV READY: $VENV"
