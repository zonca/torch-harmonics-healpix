# AGENTS — torch-harmonics-healpix

## Remote Compute

- **Expanse** (`ssh expanse`): GPU jobs only. Use Slurm with `sbatch`. Partition `gpu-shared`, account `sds275`. **NEVER submit CPU-only jobs on Expanse — that wastes GPU allocation.** CPU work goes on Popeye.
- **Popeye** (`ssh popeye`): CPU jobs only. Use Slurm with `sbatch`. Partition `gen`. SSH requires keyboard-interactive 2FA — **automated via Bitwarden** (Flatiron Popeye entry `d924502d`, same password + TOTP). Use expect script pattern: Verification code (Flatiron TOTP) → Password (Flatiron password). Home directory: `/mnt/home/azonca` (NOT `/home/azonca`). ControlMaster socket: `ssh -o ControlPath=~/.ssh/sockets/%r@%h-%p -o ControlMaster=no popeye`.
- **NRP Nautilus** (`https://nrp.ai`): GPU via Kubernetes. Namespace `sdsc-scicomp`. See `../nrp/AGENTS.md` for setup.
  - Training job YAML: `../nrp/examples/train-test4-nside128.yaml`
  - PVC `thh-data` (50Gi, `rook-cephfs`) for persistent results at `/data`
  - kubectl wrapper: `../nrp/kubectl.sh` (auto-refreshing OIDC token)
  - **Important on CephFS**: set `HDF5_USE_FILE_LOCKING=FALSE`
- **Code sync:** Always `git commit && git push` from local, then `git pull` on remote. Do NOT use `scp` or `rsync` to copy code around.
  - Exception: NRP uses `git clone` in initContainer (code is in the repo, PVC is for data only)

## Platform Specs & Software Versions

### Expanse (GPU — compute node)
- **CPU:** AMD EPYC 7742 64-Core
- **GPU:** NVIDIA Tesla V100-SXM2-32GB, Driver 525.85.12, CUDA 12.0
- **RAM:** ~128 GB (login node)
- **Python:** 3.11.5
- **PyTorch:** 2.6.0+cu124
- **torch-harmonics:** 0.8.0 (**must pin with `--no-deps`** — ≥0.9.0 has C++ ABI issue on V100)
- **healpy:** 1.19.0
- **numpy:** 2.4.4
- **scipy:** 1.17.1
- **Venv:** Use `uv venv` (not `python3 -m venv`) in Slurm scripts

### Popeye (CPU — compute node)
- **CPU:** Intel Xeon Platinum 8168 @ 2.70GHz, 48 cores
- **RAM:** ~758 GB
- **Python:** 3.11.11 (via `module load python/3.11.11`)
- **Venv:** `~/torch-hh-venv` (activate with `source ~/torch-hh-venv/bin/activate`)
- **healpy:** 1.19.0
- **numpy:** 2.4.6
- **scipy:** 1.17.1
- **Note:** System Python 3.6.8 is too old; always use the module + venv

## Slurm Scripts

All Slurm scripts are in `slurm/`:
- `run_tests.slurm` — Expanse GPU: run full test suite (34 tests)
- `run_train_test1.slurm` — Expanse GPU: train SpectralCNN for Test 1 (all noise levels)
- `run_test4_nside128_hdf5_expanse.slurm` — Expanse GPU: train Test 4 NSIDE=128 with HDF5 (account sds275)
- `run_mcmc_benchmark_popeye.slurm` — Popeye CPU: MCMC baseline benchmark (1000 maps)
- `generate_test4_nside128_popeye.slurm` — Popeye CPU: HDF5 pre-generation for Test 4

### Paper-Ready Pipeline (v2 — C_ℓ fix)

These scripts implement the corrected pipeline with `raw_cl=True` in CAMB:

- `slurm/run_fisher_mcmc_popeye.slurm` — Popeye CPU: Fisher matrix + MH-MCMC baseline (both NSIDEs)
- `slurm/generate_hdf5_nside128_popeye.slurm` — Popeye CPU: regenerate HDF5 with C_ℓ fix → scp to Expanse
- `slurm/retrain_nside16_expanse.slurm` — Expanse GPU: retrain CNN at NSIDE=16
- `slurm/retrain_nside128_expanse.slurm` — Expanse GPU: retrain CNN at NSIDE=128 (needs HDF5 v2)
- `scripts/run_fisher_mcmc_test4.py` — Fisher matrix + MH-MCMC with A_lens nuisance param

Key scripts:
- `scripts/run_fisher_mcmc_test4.py` — Computes Fisher Cramér-Rao bounds + Metropolis-Hastings MCMC
  with lensing BB as A_lens nuisance parameter. Uses pre-computed 50×50 CAMB spectral grid
  with bilinear interpolation for O(1) MCMC steps (avoiding ~10s CAMB calls per step).

### NRP Nautilus (GPU — Kubernetes)

Training is run as Kubernetes Jobs, not Slurm. See `../nrp/AGENTS.md` for full details.

- **GPU observed:** NVIDIA A40 (48GB), driver 595.71.05 (first run: GTX 1080 Ti 11GB — OOM with hidden_channels=32)
- **Image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` (packages installed at runtime)
- **Model config for NSIDE=128:** hidden_channels=32, num_blocks=3, batch_size=2 (422M params — full model)
- **OOM caveat:** Default hidden_channels=32 → 422M params, OOMs on ≤16GB GPUs. Use hidden_channels=8 or request A40.
- **Underfitting caveat:** hidden_channels=8 (30M params) underfits at NSIDE=128 — plateaued at ~63% r error vs 54% with hidden_channels=32 after epoch 1. Always use hidden_channels=32 on ≥48GB GPUs.
- **Python:** 3.11 (from image)
- **PyTorch:** 2.6.0+cu124
- **torch-harmonics:** 0.8.0 (pinned with `--no-deps`)
- **healpy:** 1.19.0
- **CAMB:** installed via pip
- **PVC:** `thh-data` (50Gi, rook-cephfs) at `/data` — stores results and CAMB cache
- **Note:** CephFS requires `HDF5_USE_FILE_LOCKING=FALSE`
- **Env:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Key Technical Notes

- `RealSHT`/`InverseRealSHT`: `lmax` and `mmax` are **dimension sizes**, not max indices. Output is complex.
- `hp.synfast()` does NOT accept `seed` kwarg in healpy 1.19.0 — use `np.random.seed()` save/restore.
- Slurm Python scripts: always set `PYTHONUNBUFFERED=1` for real-time output.
- In Slurm `--output` paths, use absolute paths (e.g. `/expanse/lustre/...`), not `$HOME` or `~` — they don't expand in SBATCH directives.
- **HDF5 on Lustre**: set `HDF5_USE_FILE_LOCKING=FALSE` in Slurm scripts. Random DataLoader shuffle + single-chunk RAM cache = 95% cache miss rate (eviction every call). Fix: use `ChunkShuffleSampler` (groups indices by chunk, shuffles chunk order + within-chunk indices).
- **τ divergence bug**: MSE loss on τ causes gradient explosion around epoch 11 (τ %err → 10^12%+) regardless of LR scheduler type. Root cause: MSE gradient = 2(τ_pred - τ_true) grows linearly with prediction error → positive feedback loop. Fix: use Huber loss for τ (delta=0.01) — linear gradient far from target caps magnitude, quadratic near target preserves accuracy. Hard clamping (torch.clamp) creates dead gradients and makes it worse. **Note:** Huber loss fixes cfg1/2/4 but cfg3 (fsky=0.1, noise=0) still diverges at ep11 — this config has the most extreme signal-to-noise ratio (small sky patch, zero noise) causing overconfident predictions. Best checkpoint is saved before divergence.
- **CosineAnnealingLR T_max**: With 24h walltime and ~58 min/epoch, only ~24 epochs fit. Set `--cosine_T_max 25` for full cosine decay cycle; default T_max=150 leaves LR essentially constant.
- **D_ℓ→C_ℓ unit bug (CRITICAL, fixed)**: `generate_camb_spectra_r_tau` was calling CAMB's default `get_total_cls(CMB_unit='muK')` which returns D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in μK², but `hp.synfast` expects C_ℓ in μK²·sr. This boosted map amplitudes at high ℓ by a factor of ℓ(ℓ+1)/(2π). **Fix**: use `raw_cl=True` to get C_ℓ directly. All HDF5 datasets and trained models before this fix have wrong amplitudes — must regenerate and retrain.
- **Fisher matrix (Cramér-Rao bounds)** at fiducial r=0.003, τ=0.054:
  - NSIDE=16: σ_r=0.00156 (52%), σ_τ=0.00223 (4.1%) for fsky=1.0/noise=0
  - NSIDE=16: σ_r=0.00495 (165%), σ_τ=0.00704 (13%) for fsky=0.1/noise=0
  - NSIDE=128: σ_r=0.000225 (7.5%), σ_τ=0.00110 (2.0%) for fsky=1.0/noise=0
  - NSIDE=128: σ_r=0.000555 (18.5%), σ_τ=0.00134 (2.5%) for fsky=1.0/noise=6
  - NSIDE=128: σ_r=0.000713 (23.8%), σ_τ=0.00347 (6.4%) for fsky=0.1/noise=0
  - NSIDE=128: σ_r=0.00176 (58.5%), σ_τ=0.00424 (7.8%) for fsky=0.1/noise=6
  - **Key insight**: fsky=0.1 configs have σ_r > r_fiducial at NSIDE=16 (cannot constrain r)
- **MCMC speed**: Never call CAMB per MCMC step (~10s each). Pre-compute a 50×50 spectral grid and use bilinear interpolation for O(1) evaluations. 100× speedup.
- **MCMC pseudo-C_ℓ limitation**: Simple pseudo-C_ℓ MCMC (even with A_lens nuisance param and proper cosmic variance likelihood) cannot constrain r — the chain always drifts to the boundary (r→0.01, τ→0.08). Root cause: single-realization C_ℓ estimates from `hp.anafast` are too noisy (cosmic variance) for a pseudo-C_ℓ comparison to identify the true parameters. **Use Fisher matrix as the Cramér-Rao baseline** — this is standard in CMB literature. The Fisher bound gives the optimal σ_r and σ_τ achievable by any unbiased estimator.
- **NSIDE=16 CNN results (v3)** — CNN beats Fisher on r by 1.6–3×:
  - fsky=1.0, noise=0: CNN r=21.9% vs Fisher 52.1% (0.42× Fisher)
  - fsky=1.0, noise=6: CNN r=32.7% vs Fisher 56.9% (0.57× Fisher)
  - fsky=0.1, noise=0: CNN r=57.6% vs Fisher 164.9% (0.35× Fisher)
  - fsky=0.1, noise=6: CNN r=56.3% vs Fisher 180.0% (0.31× Fisher)
  - τ error is higher than Fisher (15–28% vs 4–13%) — network optimizes r over τ
- **NSIDE=128 CNN results (v3, hc=8 underfitted)** — CNN ~58% r error (hc=8 underfits, hc=32 training pending):
  - fsky=1.0, noise=0: CNN r=59.1% vs Fisher 7.5% (7.87× Fisher)
  - fsky=1.0, noise=6: CNN r=58.6% vs Fisher 18.5% (3.17× Fisher)
  - fsky=0.1, noise=0: CNN r=58.4% vs Fisher 23.8% (2.46× Fisher)
  - fsky=0.1, noise=6: no result yet
  - **Note**: hidden_channels=8 (30M params) severely underfits at NSIDE=128. Need hc=32 (422M params).
- **v3 results directory**: `results_v3/` — Fisher JSONs, CNN JSONs, MCMC JSONs, `cnn_vs_fisher_summary.json`
- **Popeye disk**: `/mnt/home/azonca` is at 100% (500GB). Use `/tmp` (3.4TB node-local) for large files, but `/tmp` is wiped after job ends.
- **Popeye→Expanse transfer**: Both `scp` and `rsync` fail immediately ("connection unexpectedly closed") from Popeye compute nodes to Expanse login. Generate HDF5 directly on Expanse GPU node instead (slower at ~11 maps/s vs 33 maps/s on Popeye, but avoids transfer).

### HDF5 Data on Expanse Lustre

- **Original directory (v1 — D_ℓ bug)**: `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data/`
  - 4 HDF5 files (~244 GB each): fsky1.0_noise0, fsky1.0_noise6, fsky0.1_noise0, fsky0.1_noise6
  - 100K train + 10K val + 1K test maps per file, NSIDE=128
  - **DEPRECATED**: Generated with D_ℓ (CAMB default), not C_ℓ. Wrong amplitudes.
- **v2 directory (C_ℓ fix)**: `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data_v2/`
  - Same structure, generated with `raw_cl=True` in CAMB
  - Regenerated via `slurm/generate_hdf5_nside128_popeye.slurm` → scp to Expanse
- **Striped directory**: `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_striped/`
  - Created with `lfs setstripe -c 16 -S 4M` (16 OSTs, 4 MB stripe size)
  - Expected bandwidth: 8-16x faster (~0.5-1+ GB/s), reducing I/O per epoch from ~40 min to ~3-5 min
  - **Re-striping**: `cp` from `hdf5_data/` to `hdf5_striped/` (Lustre respects target directory striping on new files)
  - Run from login node: `nohup bash -c 'for f in hdf5_data/*.h5; do cp "$f" hdf5_striped/; done' &`
  - Or from compute node via `srun --jobid=XXX --overlap` (faster, ~5 min per file)
  - **After restriping**: update Slurm scripts to point `--hdf5_path` to `hdf5_striped/` instead of `hdf5_data/`
- **Results directories**:
  - v1 (D_ℓ bug): `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results/`
  - v2 (C_ℓ fix): `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2/`
  - v3 (C_ℓ fix + Huber τ loss): `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3/` and `results_v3/` (local)
- **Epoch timing** (single-striped files, batch_size=16, 422M params, V100):
  - I/O: ~40 min (20 chunks × ~120s avg, varying 7-650s due to OST variability)
  - GPU training: ~20 min (6250 batches × ~0.2s)
  - Validation: ~3 min (625 batches)
  - Total: ~60 min/epoch
