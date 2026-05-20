# Test 1: Reproduce ℓ_p Estimation from Scalar Maps

Reproduce the first benchmark from Krachmalnicoff & Tomasi (2019, arXiv:1902.04083) Section 6.1.1 using torch-harmonics instead of NNhealpix.

## Objective

Estimate the peak multipole ℓ_p of a Gaussian random field on the sphere, directly from HEALPix temperature maps, using a spectral CNN built on torch-harmonics. Compare against the NNhealpix and MCMC baselines.

## Problem Definition

**Power spectrum model:**
```
C_ℓ = exp(-(ℓ - ℓ_p)² / (2 σ_p²)) + 10⁻⁵
```
- σ_p = 5 (fixed)
- ℓ_p ∈ [5, 20] (uniformly sampled, this is the parameter to estimate)
- ℓ_max = 3 × N_side - 1 = 47

**Map simulation:**
- HEALPix, N_side = 16 → N_pix = 3072
- Each map is a Gaussian realization of the power spectrum above
- Optional: add white noise with σ_n ∈ {0, 5, 10, 15}

**Dataset sizes:**
- Training: 100,000 maps
- Validation: 10,000 maps
- Test: 1,000 maps

Each map uses a random ℓ_p from U[5, 20] and a random seed.

**Output:** single scalar ℓ_p estimate

## Baselines (from the paper)

| Noise level | NNhealpix (mean % error) | MCMC (mean % error) |
|---|---|---|
| No noise (σ_n=0) | 1.3% | 0.7% |
| S/N=1 (σ_n=5) | 2.9% | 2.5% |
| S/N=1/2 (σ_n=10) | 5.2% | 4.8% |
| S/N=1/3 (σ_n=15) | 8.4% | 7.8% |

Mean % error = average of |ℓ_p_pred - ℓ_p_true| / ℓ_p_true × 100 over the test set.

## Implementation Plan

### Step 1: Environment setup on Expanse

```bash
# SSH to Expanse
ssh expanse

# Request GPU node
salloc --nodes=1 --ntasks-per-node=4 --partition=gpu --gres=gpu:v100:1 --walltime=02:00:00

# Create conda environment
conda create -n torch-harmonics-healpix python=3.11 -y
conda activate torch-harmonics-healpix

# Install dependencies
pip install torch torchvision  # CUDA version matching Expanse GPU
pip install torch-harmonics
pip install healpy
pip install numpy scipy matplotlib
```

### Step 2: Data generation

File: `generate_test1_data.py`

```python
import healpy as hp
import numpy as np
import h5py

NSIDE = 16
LMAX = 3 * NSIDE - 1  # 47
SIGMA_P = 5
N_TRAIN = 100_000
N_VAL = 10_000
N_TEST = 1_000

def generate_power_spectrum(ell_p, sigma_p=SIGMA_P, lmax=LMAX):
    ell = np.arange(lmax + 1)
    cl = np.exp(-(ell - ell_p)**2 / (2 * sigma_p**2)) + 1e-5
    # healpy expects a 1D array of length lmax+1
    return cl

def generate_dataset(n_maps, nside=NSIDE, lmax=LMAX, noise_std=0.0, seed_start=0):
    maps = np.zeros((n_maps, hp.nside2npix(nside)), dtype=np.float32)
    ell_p_values = np.random.uniform(5, 20, size=n_maps).astype(np.float32)
    
    for i in range(n_maps):
        cl = generate_power_spectrum(ell_p_values[i])
        m = hp.synfast(cl, nside=nside, lmax=lmax, verbose=False,
                       random_seed=seed_start + i)
        if noise_std > 0:
            m += np.random.normal(0, noise_std, size=m.shape)
        maps[i] = m
    
    return maps, ell_p_values

# Generate all datasets
for noise_std in [0, 5, 10, 15]:
    suffix = f"_noise{noise_std}" if noise_std > 0 else ""
    
    np.random.seed(42)
    train_maps, train_ell_p = generate_dataset(N_TRAIN, noise_std=noise_std, seed_start=0)
    val_maps, val_ell_p = generate_dataset(N_VAL, noise_std=noise_std, seed_start=N_TRAIN)
    test_maps, test_ell_p = generate_dataset(N_TEST, noise_std=noise_std, seed_start=N_TRAIN + N_VAL)
    
    with h5py.File(f"data/test1{suffix}.h5", "w") as f:
        f.create_dataset("train_maps", data=train_maps)
        f.create_dataset("train_ell_p", data=train_ell_p)
        f.create_dataset("val_maps", data=val_maps)
        f.create_dataset("val_ell_p", data=val_ell_p)
        f.create_dataset("test_maps", data=test_maps)
        f.create_dataset("test_ell_p", data=test_ell_p)
```

**Notes:**
- Use `hp.synfast` with explicit seeds for reproducibility
- Store as float32 to save disk space
- HDF5 format for efficient loading during training
- Total data size: ~1.2 GB per noise level (100k maps × 3072 pixels × 4 bytes)

### Step 3: HEALPix ↔ Equiangular resampling layer

File: `healpix_resample.py`

This is the key bridge. We need to convert HEALPix maps to the [nlat, nlon] equiangular grid that torch-harmonics expects, and back.

```python
import torch
import torch.nn as nn
import healpy as hp
import numpy as np

class HealpixToEquiangular(nn.Module):
    """Convert HEALPix map to equiangular [nlat, nlon] grid for torch-harmonics."""
    
    def __init__(self, nside, nlat=None, nlon=None):
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        
        # Default: enough resolution for lmax = 3*nside - 1
        if nlat is None:
            nlat = 2 * nside  # matches common equiangular grids
        if nlon is None:
            nlon = 2 * nlat  # ~1-degree longitude resolution
        
        self.nlat = nlat
        self.nlon = nlon
        
        # Precompute the interpolation indices
        # For each (lat, lon) point in the equiangular grid,
        # find which HEALPix pixels to sample from
        lats = np.linspace(np.pi/2, -np.pi/2, nlat, endpoint=False)
        lons = np.linspace(0, 2*np.pi, nlon, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Convert to HEALPix pixel indices
        self.register_buffer('pixel_indices', 
            torch.from_numpy(
                hp.ang2pix(nside, lat_grid.ravel(), lon_grid.ravel(), lonlat=False)
            ).long()
        )
    
    def forward(self, healpix_map):
        """
        Args:
            healpix_map: [batch, npix] or [batch, channels, npix]
        Returns:
            equiangular: [batch, nlat, nlon] or [batch, channels, nlat, nlon]
        """
        if healpix_map.dim() == 2:
            # [batch, npix] -> [batch, nlat, nlon]
            equi = healpix_map[:, self.pixel_indices]
            return equi.view(-1, self.nlat, self.nlon)
        elif healpix_map.dim() == 3:
            # [batch, channels, npix] -> [batch, channels, nlat, nlon]
            equi = healpix_map[:, :, self.pixel_indices]
            return equi.view(-1, healpix_map.shape[1], self.nlat, self.nlon)


class EquiangularToHealpix(nn.Module):
    """Convert equiangular [nlat, nlon] grid back to HEALPix map."""
    
    def __init__(self, nside, nlat=None, nlon=None):
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        
        if nlat is None:
            nlat = 2 * nside
        if nlon is None:
            nlon = 2 * nlat
        
        self.nlat = nlat
        self.nlon = nlon
        
        # For each HEALPix pixel, find its (lat, lon) position
        # and compute bilinear interpolation weights on the equiangular grid
        theta, phi = hp.pix2ang(nside, np.arange(self.npix))
        # theta is colatitude [0, pi], phi is longitude [0, 2pi]
        
        # Convert to equiangular grid indices (fractional)
        lat_idx = (np.pi/2 - theta) / (np.pi / nlat)  # fractional row index
        lon_idx = phi / (2 * np.pi / nlon)  # fractional col index
        
        # Nearest neighbor for simplicity (can upgrade to bilinear later)
        self.register_buffer('lat_indices', torch.from_numpy(np.clip(
            np.round(lat_idx).astype(int), 0, nlat-1)).long())
        self.register_buffer('lon_indices', torch.from_numpy(np.clip(
            np.round(lon_idx).astype(int) % nlon, 0, nlon-1)).long())
    
    def forward(self, equi_map):
        """
        Args:
            equi_map: [batch, nlat, nlon] or [batch, channels, nlat, nlon]
        Returns:
            healpix: [batch, npix] or [batch, channels, npix]
        """
        if equi_map.dim() == 3:
            return equi_map[:, self.lat_indices, self.lon_indices]
        elif equi_map.dim() == 4:
            return equi_map[:, :, self.lat_indices, self.lon_indices]
```

**Note:** The initial implementation uses nearest-neighbor interpolation for simplicity. We'll upgrade to bilinear (or use `torch_harmonics.ResampleS2`) if accuracy requires it.

### Step 4: Model architecture — SpectralConvS2

File: `models/spectral_cnn.py`

Two variants to compare:

#### Variant A: SpectralConvS2 (SFNO-style)

```python
import torch
import torch.nn as nn
from torch_harmonics import RealSHT, InverseRealSHT, SpectralConvS2

class SpectralCNN(nn.Module):
    """Spectral convolution CNN for scalar HEALPix maps.
    
    Architecture inspired by NNhealpix paper but using spectral convolutions:
    - 4 spectral conv blocks, each halves resolution (effectively Nside 16→8→4→2→1)
    - Uses SpectralConvS2 for rotation-equivariant convolution
    - FC head for parameter regression
    """
    def __init__(self, nlat, nlon, lmax=47, hidden_channels=32):
        super().__init__()
        
        # SHT transforms at different resolutions
        # Block 1: full resolution
        self.sht1 = RealSHT(nlat, nlon, lmax=lmax, grid="equiangular")
        self.isht1 = InverseRealSHT(nlat, nlon, lmax=lmax, grid="equiangular")
        self.spec_conv1 = SpectralConvS2(
            (nlat, nlon), (nlat, nlon), 
            in_channels=1, out_channels=hidden_channels
        )
        
        # Block 2: halved resolution
        nlat2, nlon2 = nlat//2, nlon//2
        lmax2 = lmax // 2
        self.sht2 = RealSHT(nlat2, nlon2, lmax=lmax2, grid="equiangular")
        self.isht2 = InverseRealSHT(nlat2, nlon2, lmax=lmax2, grid="equiangular")
        self.spec_conv2 = SpectralConvS2(
            (nlat2, nlon2), (nlat2, nlon2),
            in_channels=hidden_channels, out_channels=hidden_channels
        )
        self.downsample1 = nn.AvgPool2d(2)
        
        # ... (similar blocks 3, 4)
        
        # FC head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * nlat_final * nlon_final, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1)  # single output: ell_p
        )
    
    def forward(self, x):
        # x: [batch, nlat, nlon] (equiangular)
        x = x.unsqueeze(1)  # [batch, 1, nlat, nlon]
        
        x = self.spec_conv1(x)
        x = self.downsample1(x)
        x = self.spec_conv2(x)
        # ... etc
        
        return self.fc(x).squeeze(-1)
```

#### Variant B: DISCO convolution (S2UNet-style)

```python
from torch_harmonics import DiscreteContinuousConvS2, ResampleS2

class DiscoCNN(nn.Module):
    """DISCO convolution CNN for scalar HEALPix maps.
    
    Uses local spherical convolutions (more like standard CNNs but on the sphere).
    """
    def __init__(self, nlat, nlon, hidden_channels=32):
        super().__init__()
        
        # Block 1
        self.conv1 = DiscreteContinuousConvS2(
            in_channels=1, out_channels=hidden_channels,
            in_shape=(nlat, nlon), out_shape=(nlat, nlon),
            kernel_shape=3, grid_in="equiangular", grid_out="equiangular"
        )
        self.down1 = ResampleS2(nlat, nlon, nlat//2, nlon//2)
        
        # ... similar blocks
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_size, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1)
        )
```

**Recommendation:** Start with Variant A (SpectralConvS2) since ℓ_p is literally a spectral parameter. The spectral convolution should have an inherent advantage here.

### Step 5: Training script

File: `train_test1.py`

```python
# Key hyperparameters (matching NNhealpix paper where possible)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = "Adam"
SCHEDULER = "ReduceLROnPlateau"  # reduce by 10x after 5 epochs no improvement
EARLY_STOPPING = 20  # epochs without improvement
MAX_EPOCHS = 100

# Metric
def mean_percentage_error(pred, target):
    return (torch.abs(pred - target) / target * 100).mean().item()
```

### Step 6: MCMC baseline

File: `mcmc_baseline.py`

Reproduce the MCMC baseline from the paper using healpy:

```python
import healpy as hp
from scipy.optimize import minimize

def mcmc_estimate_ell_p(map_data, sigma_p=5, lmax=47, noise_std=0):
    """Estimate ell_p from a single map using MCMC on the power spectrum."""
    # Compute power spectrum from map
    cl_est = hp.anafast(map_data, lmax=lmax)
    
    # Chi-squared likelihood (Eq. 5 in paper)
    def neg_log_likelihood(ell_p):
        cl_model = np.exp(-(np.arange(lmax+1) - ell_p)**2 / (2*sigma_p**2)) + 1e-5
        if noise_std > 0:
            N_ell = 4 * np.pi * noise_std**2 / len(map_data)
            cl_model += N_ell
        sigma_cl = cl_model * np.sqrt(2 / (2 * np.arange(lmax+1) + 1))
        chi2 = np.sum(((cl_est - cl_model) / sigma_cl)**2)
        return chi2
    
    # Minimize
    result = minimize(neg_log_likelihood, x0=12.5, bounds=[(5, 20)])
    return result.x[0]
```

### Step 7: Evaluation and comparison

For each noise level (σ_n = 0, 5, 10, 15):
1. Train the torch-harmonics model
2. Evaluate mean % error on test set
3. Run MCMC baseline on same test set
4. Compare against NNhealpix numbers from the paper
5. Produce scatter plots of ℓ_p_pred vs ℓ_p_true (like Fig. 11 in the paper)

## File structure

```
torch-harmonics-healpix/
├── PLAN.md                    # Overall project plan
├── TEST1.md                   # This file
├── ALTERNATIVES.md            # Alternatives considered
├── README.md                  # Package overview
├── data/
│   └── test1_noise0.h5        # Generated datasets (git-ignored)
├── generate_test1_data.py     # Data generation script
├── healpix_resample.py        # HEALPix ↔ equiangular conversion
├── models/
│   ├── spectral_cnn.py        # SpectralConvS2 model
│   └── disco_cnn.py           # DISCO convolution model
├── train_test1.py             # Training loop
├── evaluate_test1.py          # Evaluation script
└── mcmc_baseline.py           # MCMC baseline for comparison
```

## Success criteria

| Noise level | NNhealpix | Our target (stretch) |
|---|---|---|
| No noise | 1.3% | ≤ 0.7% (match MCMC) |
| S/N=1 | 2.9% | ≤ 2.5% (match MCMC) |
| S/N=1/2 | 5.2% | ≤ 4.8% (match MCMC) |
| S/N=1/3 | 8.4% | ≤ 7.8% (match MCMC) |

The **minimum viable success** is matching NNhealpix performance. The **stretch goal** is approaching MCMC accuracy, which should be achievable because:
1. SpectralConvS2 operates directly in ℓ-space where ℓ_p is defined
2. Rotation equivariance means more efficient learning
3. Modern GPU-optimized PyTorch vs 2019 TensorFlow on CPU

## Timeline

- **Day 1-2**: Environment setup on Expanse, data generation, resampling layer
- **Day 3-4**: Model implementation (SpectralConvS2 variant)
- **Day 5-6**: Training runs for all noise levels
- **Day 7**: MCMC baseline, evaluation, comparison plots
- **Day 8**: If time allows, try DISCO convolution variant

## Open questions

1. **Equiangular grid resolution**: what nlat/nlon to use? Start with nlat=32, nlon=64 (enough for lmax=47). May need to experiment.
2. **Resampling accuracy**: will nearest-neighbor HEALPix↔equiangular be accurate enough, or do we need bilinear? The paper uses N_side=16 which is very coarse (3072 pixels). Nearest-neighbor might suffice.
3. **Number of spectral conv layers**: the paper uses 4 Conv+Pool blocks. We should start with the same depth. But spectral convolutions might need fewer layers since they're global.
4. **Batch normalization**: the paper doesn't use it. torch-harmonics examples do. Try both.
