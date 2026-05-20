# AGENTS — torch-harmonics-healpix

## Remote Compute

- **Expanse** (`ssh expanse`): GPU jobs only. Use Slurm with `sbatch`. Partition `gpu-shared`, account `sds166`.
- **Popeye** (`ssh popeye`): CPU jobs only. Use Slurm with `sbatch`. Partition `gen`. SSH key auth (no 2FA).
- **Code sync:** Always `git commit && git push` from local, then `git pull` on remote. Do NOT use `scp` or `rsync` to copy code around.

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

## Key Technical Notes

- `RealSHT`/`InverseRealSHT`: `lmax` and `mmax` are **dimension sizes**, not max indices. Output is complex.
- `hp.synfast()` does NOT accept `seed` kwarg in healpy 1.19.0 — use `np.random.seed()` save/restore.
- Slurm Python scripts: always set `PYTHONUNBUFFERED=1` for real-time output.
- In Slurm `--output` paths, use `$HOME` not `~` (tilde doesn't expand in SBATCH directives).
