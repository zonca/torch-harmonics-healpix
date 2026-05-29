# AGENTS — torch-harmonics-healpix

## Remote Compute

- **Expanse** (`ssh expanse`): GPU jobs only. Use Slurm with `sbatch`. Partition `gpu-shared`, account `sds275`.
- **Popeye** (`ssh popeye`): CPU jobs only. Use Slurm with `sbatch`. Partition `gen`. SSH key auth (no 2FA).
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

### HDF5 Data on Expanse Lustre

- **Original directory**: `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data/`
  - 4 HDF5 files (~244 GB each): fsky1.0_noise0, fsky1.0_noise6, fsky0.1_noise0, fsky0.1_noise6
  - 100K train + 10K val + 1K test maps per file, NSIDE=128
  - **Problem**: `lfs getstripe` shows `stripe_count=1` (single OST). Bandwidth capped at ~80 MB/s per file, causing ~60 min/epoch I/O time
- **Striped directory**: `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_striped/`
  - Created with `lfs setstripe -c 16 -S 4M` (16 OSTs, 4 MB stripe size)
  - Expected bandwidth: 8-16x faster (~0.5-1+ GB/s), reducing I/O per epoch from ~40 min to ~3-5 min
  - **Re-striping**: `cp` from `hdf5_data/` to `hdf5_striped/` (Lustre respects target directory striping on new files)
  - Run from login node: `nohup bash -c 'for f in hdf5_data/*.h5; do cp "$f" hdf5_striped/; done' &`
  - Or from compute node via `srun --jobid=XXX --overlap` (faster, ~5 min per file)
  - **After restriping**: update Slurm scripts to point `--hdf5_path` to `hdf5_striped/` instead of `hdf5_data/`
- **Epoch timing** (single-striped files, batch_size=16, 422M params, V100):
  - I/O: ~40 min (20 chunks × ~120s avg, varying 7-650s due to OST variability)
  - GPU training: ~20 min (6250 batches × ~0.2s)
  - Validation: ~3 min (625 batches)
  - Total: ~60 min/epoch
