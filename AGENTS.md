# AGENTS — torch-harmonics-healpix

## Remote Compute

- **Expanse** (`ssh expanse`): GPU jobs only. Use Slurm with `sbatch`. Partition `gpu-shared`, account `sds166`.
- **Popeye** (`ssh popeye`): Long CPU calculations. SSH key auth works (no 2FA needed).
- **Code sync:** Always `git commit && git push` from local, then `git pull` on remote. Do NOT use `scp` or `rsync` to copy code around.

## Software Versions (Expanse)

- Python 3.11, PyTorch 2.6.0+cu124, torch-harmonics 0.8.0, healpy 1.19.0, CUDA 12.0
- **torch-harmonics must be pinned to 0.8.0 with `--no-deps`** — version ≥0.9.0 has C++ ABI issue on V100

## Key Technical Notes

- `RealSHT`/`InverseRealSHT`: `lmax` and `mmax` are **dimension sizes**, not max indices. Output is complex.
- `hp.synfast()` does NOT accept `seed` kwarg in healpy 1.19.0 — use `np.random.seed()` save/restore.
- Slurm Python scripts: always set `PYTHONUNBUFFERED=1` for real-time output.
- Use `uv venv` (not `python3 -m venv`) in Slurm scripts to avoid library conflicts.
