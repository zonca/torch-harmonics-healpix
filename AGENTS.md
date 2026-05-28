# AGENTS — torch-harmonics-healpix

## Remote Compute

- **Expanse** (`ssh expanse`): GPU jobs only. Use Slurm with `sbatch`. Partition `gpu-shared`, account `sds166`.
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
- `run_mcmc_benchmark_popeye.slurm` — Popeye CPU: MCMC baseline benchmark (1000 maps)

### NRP Nautilus (GPU — Kubernetes)

Training is run as Kubernetes Jobs, not Slurm. See `../nrp/AGENTS.md` for full details.

- **GPU observed:** NVIDIA A40 (48GB), driver 595.71.05 (first run: GTX 1080 Ti 11GB — OOM with hidden_channels=32)
- **Image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` (packages installed at runtime)
- **Model config for NSIDE=128:** hidden_channels=8, num_blocks=3, batch_size=4 (~30M params)
- **OOM caveat:** Default hidden_channels=32 → 422M params, OOMs on ≤16GB GPUs. Use hidden_channels=8 or request A40.
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
