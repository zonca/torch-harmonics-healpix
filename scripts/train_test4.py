"""Train SpectralCNN for Test 4: joint estimation of r (tensor-to-scalar ratio) and τ (optical depth) from Q/U polarization maps. Extends Test 3 to 2-parameter estimation relevant to Simons Observatory.

Supports two data modes:
  1. On-the-fly generation (--hdf5_path not set): generates maps via hp.synfast in __getitem__
  2. HDF5 pre-generated (--hdf5_path set): reads from pre-generated HDF5 file on disk

HDF5 mode is strongly recommended for NSIDE≥128 where synfast is the bottleneck.
"""

import argparse
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from torch_harmonics_healpix.data_generation_test4 import (
    generate_r_tau_map,
    precompute_camb_spectra_r_tau,
    R_MIN, R_MAX,
    TAU_MIN, TAU_MAX,
    N_CAMB_SPECTRA,
    R_LOG_EPSILON,
)
from torch_harmonics_healpix.models.spectral_cnn import SpectralCNN


class RTauDataset(Dataset):
    """On-the-fly Q/U map generation for joint r/τ estimation.

    Pre-computes CAMB spectra for N_CAMB_SPECTRA (r, τ) pairs, then
    randomly picks one spectrum for each map. This avoids calling CAMB
    per map and is much faster.

    r is stored in log-space: r_target = log(r + ε) to handle the
    boundary at r = 0 and improve training dynamics across the wide
    dynamic range r ∈ [0, 0.01].
    """

    def __init__(self, n_maps, nside=16, lmax=47,
                 noise_std=0.0, f_sky=1.0, seed=42, mask=None,
                 camb_spectra=None):
        """Initialize the RTauDataset.

        Args:
            n_maps: Number of maps to generate (dataset length).
            nside: HEALPix Nside parameter.
            lmax: Maximum multipole for power spectra.
            noise_std: White noise standard deviation in μK (per pixel).
            f_sky: Sky fraction observed (1.0 = full sky).
            seed: Random seed for reproducibility.
            mask: Pre-computed sky mask (1D HEALPix array); generated if None.
            camb_spectra: Optional tuple of (r_values, tau_values, cl_ee_array, cl_bb_array)
                to reuse pre-computed CAMB spectra across datasets.
        """
        self.n_maps = n_maps
        self.nside = nside
        self.lmax = lmax
        self.noise_std = noise_std
        self.f_sky = f_sky
        self.rng = np.random.default_rng(seed)

        # Use shared CAMB spectra if provided, otherwise pre-compute
        if camb_spectra is not None:
            r_values, tau_values, self.cl_ee_array, self.cl_bb_array = camb_spectra
        else:
            print(f"  Pre-computing {N_CAMB_SPECTRA} CAMB spectra for RTauDataset...")
            r_values, tau_values, self.cl_ee_array, self.cl_bb_array = \
                precompute_camb_spectra_r_tau(N_CAMB_SPECTRA, lmax, seed=seed + 100)

        # Randomly choose from pre-computed spectra for each map
        self.spectrum_indices = self.rng.integers(0, len(r_values), size=n_maps)
        self.r_values = r_values[self.spectrum_indices]
        self.tau_values = tau_values[self.spectrum_indices]

        # Log-transform r for boundary handling
        self.r_target = np.log(self.r_values + R_LOG_EPSILON).astype(np.float32)
        self.tau_target = self.tau_values.astype(np.float32)

        # Use provided mask or generate one (shared mask critical for SpectralCNN)
        from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
        if mask is not None:
            self.mask = mask
        else:
            self.mask = create_sky_mask(f_sky, nside, self.rng).astype(np.float32)

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        """Generate and return a single (QU_map, [r_log, τ]) pair on-the-fly."""
        spec_idx = self.spectrum_indices[idx]
        q, u, _ = generate_r_tau_map(
            self.r_values[idx],
            self.tau_values[idx],
            self.nside, self.lmax, self.noise_std, self.f_sky, self.rng,
            cl_ee=self.cl_ee_array[spec_idx],
            cl_bb=self.cl_bb_array[spec_idx],
        )

        # Stack Q, U, mask as 3 channels: [3, npix]
        qu_map = np.stack([q, u, self.mask], axis=0).astype(np.float32)
        target = torch.tensor([self.r_target[idx], self.tau_target[idx]],
                              dtype=torch.float32)

        return torch.from_numpy(qu_map), target


class HDF5RTauDataset(Dataset):
    """HDF5-backed dataset for pre-generated Q/U maps (Test 4).

    Reads maps from HDF5 files created by scripts/generate_test4_hdf5.py.
    Supports two access modes:
      - Direct HDF5 reads (default): each __getitem__ reads one map from disk.
        Slow on Lustre due to random I/O. Acceptable for small datasets.
      - RAM chunk cache (cache_chunk_size > 0): pre-loads chunks of maps into
        RAM for fast access. When the DataLoader requests a map outside the
        current chunk, the next chunk is loaded. This dramatically speeds up
        training on Lustre by converting random reads into large sequential ones.

    The HDF5 file layout is:
      /train/maps    [N_train, 3, npix] float32  (Q, U, mask channels)
      /train/targets [N_train, 2]       float32  (log(r+ε), τ)
      /val/maps      [N_val, 3, npix]
      /val/targets   [N_val, 2]
      /test/maps     [N_test, 3, npix]
      /test/targets  [N_test, 2]
      /mask          [npix]             float32  (shared mask)

    h5py file handles are opened lazily per-worker to support multiprocessing
    DataLoader (h5py does not allow sharing file handles across forked processes).
    """

    def __init__(self, h5_path, split='train', cache_chunk_size=0):
        """Initialize HDF5RTauDataset.

        Args:
            h5_path: Path to HDF5 file with pre-generated maps.
            split: One of 'train', 'val', 'test'.
            cache_chunk_size: Number of maps to cache in RAM per chunk.
                0 = no caching (read each map from disk on demand).
                >0 = cache this many maps in RAM at a time. Recommended: 5000-10000
                for NSIDE=128 on Lustre (each map ~2.3 MB, so 5K maps ≈ 11.5 GB RAM).
                The DataLoader should shuffle indices so that maps within the same
                chunk are accessed together for best performance.
        """
        import h5py
        self.h5_path = h5_path
        self.split = split
        self._file = None
        self.cache_chunk_size = cache_chunk_size

        # Read metadata without keeping file open
        with h5py.File(h5_path, 'r') as f:
            self.n_maps = f[split]['maps'].shape[0]
            self.npix = f[split]['maps'].shape[2]
            self.nside = int(f.attrs.get('nside', 128))
            self.f_sky = float(f.attrs.get('f_sky', 1.0))

        # Chunk cache state
        self._cached_chunk_idx = -1
        self._maps_cache = None
        self._targets_cache = None

    def _ensure_open(self):
        """Open HDF5 file lazily (safe for multiprocessing workers)."""
        if self._file is None:
            import h5py
            self._file = h5py.File(self.h5_path, 'r')

    def _load_chunk(self, chunk_idx):
        """Load a chunk of maps into RAM from HDF5."""
        self._ensure_open()
        start = chunk_idx * self.cache_chunk_size
        end = min(start + self.cache_chunk_size, self.n_maps)
        print(f"  Loading HDF5 chunk {chunk_idx}: maps [{start}:{end}] into RAM...")
        t0 = time.time()
        self._maps_cache = np.array(self._file[self.split]['maps'][start:end],
                                    dtype=np.float32)
        self._targets_cache = np.array(self._file[self.split]['targets'][start:end],
                                       dtype=np.float32)
        elapsed = time.time() - t0
        print(f"  Loaded {end - start} maps in {elapsed:.1f}s "
              f"({(end - start) / elapsed:.0f} maps/s, "
              f"{self._maps_cache.nbytes / 1e9:.1f} GB)")
        self._cached_chunk_idx = chunk_idx

    def __len__(self):
        return self.n_maps

    def __getitem__(self, idx):
        """Return (qu_map, target) pair from HDF5 (with optional RAM cache)."""
        if self.cache_chunk_size > 0:
            chunk_idx = idx // self.cache_chunk_size
            if chunk_idx != self._cached_chunk_idx:
                self._load_chunk(chunk_idx)
            local_idx = idx - chunk_idx * self.cache_chunk_size
            qu_map = self._maps_cache[local_idx]
            target = self._targets_cache[local_idx]
        else:
            self._ensure_open()
            qu_map = self._file[self.split]['maps'][idx]
            target = self._file[self.split]['targets'][idx]
            qu_map = np.array(qu_map, dtype=np.float32)
            target = np.array(target, dtype=np.float32)

        return torch.from_numpy(qu_map.copy()), torch.tensor(target, dtype=torch.float32)

    def __del__(self):
        if self._file is not None:
            self._file.close()


class ChunkShuffleSampler(Sampler):
    """Sampler that shuffles chunk order and indices within each chunk.

    Designed for use with HDF5RTauDataset's RAM chunk cache. Instead of
    fully randomizing index order (which causes constant cache eviction),
    this sampler:
      1. Shuffles the ORDER of chunks (global randomization across epochs)
      2. Shuffles indices WITHIN each chunk (local randomization)
      3. Yields all indices from one chunk before moving to the next

    This gives the statistical benefits of shuffling while keeping the
    HDF5 chunk cache hit rate at ~100%, dramatically speeding up training
    on Lustre where each chunk load takes 1-3 minutes.

    Args:
        dataset_size: Total number of samples in the dataset.
        chunk_size: Number of samples per chunk (must match cache_chunk_size).
        shuffle_chunks: Whether to randomize chunk order (default: True).
        shuffle_within: Whether to randomize within each chunk (default: True).
    """

    def __init__(self, dataset_size, chunk_size, shuffle_chunks=True,
                 shuffle_within=True):
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within = shuffle_within
        self.n_chunks = (dataset_size + chunk_size - 1) // chunk_size
        self.epoch = 0

    def __iter__(self):
        # Generate chunk indices
        chunk_indices = list(range(self.n_chunks))
        if self.shuffle_chunks:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            chunk_indices = torch.randperm(self.n_chunks, generator=g).tolist()

        # For each chunk, generate shuffled indices within that chunk
        all_indices = []
        for ci in chunk_indices:
            start = ci * self.chunk_size
            end = min(start + self.chunk_size, self.dataset_size)
            chunk_idx = list(range(start, end))
            if self.shuffle_within:
                g = torch.Generator()
                g.manual_seed(self.epoch * 10000 + ci)
                shuffled = torch.randperm(len(chunk_idx), generator=g).tolist()
                chunk_idx = [chunk_idx[j] for j in shuffled]
            all_indices.extend(chunk_idx)

        return iter(all_indices)

    def __len__(self):
        return self.dataset_size

    def set_epoch(self, epoch):
        """Set the epoch for deterministic shuffling across epochs."""
        self.epoch = epoch


def _hdf5_worker_init(worker_id):
    """Reset h5py file handle in each DataLoader worker process.

    When num_workers>0, DataLoader forks the dataset. The parent's h5py
    file handle is invalid in the child process. This function closes
    the inherited handle so _ensure_open() creates a fresh one.
    """
    dataset = torch.utils.data.get_worker_info().dataset
    if hasattr(dataset, '_file') and dataset._file is not None:
        dataset._file.close()
        dataset._file = None


def train_one_epoch(model, dataloader, optimizer, device, grad_clip=0):
    """Train model for one epoch. Returns mean loss.

    Args:
        model: SpectralCNN model to train.
        dataloader: DataLoader yielding (input_maps, targets) batches.
        optimizer: PyTorch optimizer.
        device: torch.device for computation.
        grad_clip: Max gradient norm for clipping (0=disabled).

    Returns:
        Mean MSE loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = nn.functional.mse_loss(pred, batch_y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate joint r/τ estimation. Returns dict with r_pct_error, tau_pct_error, r_bias.

    Args:
        model: SpectralCNN model to evaluate.
        dataloader: DataLoader yielding (input_maps, targets) batches.
        device: torch.device for computation.

    Returns:
        Dict with keys:
            r_pct_error: mean |r_pred - r_true| / r_true × 100 for r_true > 0.001.
            tau_pct_error: mean |τ_pred - τ_true| / |τ_true| × 100 (denom clamped ≥ 0.01).
            r_bias: mean (r_pred - r_true) for r_true < 0.001 (near-zero r detection).
    """
    model.eval()
    r_pct_errors = []
    tau_pct_errors = []
    r_biases = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()

        # batch_y: [batch, 2] with [r_log, tau]
        r_log_true = batch_y[:, 0]
        tau_true = batch_y[:, 1]

        # Convert r_log back to r
        r_pred = torch.exp(pred[:, 0]) - R_LOG_EPSILON
        r_true = torch.exp(r_log_true) - R_LOG_EPSILON
        tau_pred = pred[:, 1]
        tau_true_f = tau_true

        # r percentage error for r_true > 0.001
        r_mask = r_true > 0.001
        if r_mask.any():
            r_pct = (r_pred[r_mask] - r_true[r_mask]).abs() / r_true[r_mask] * 100
            r_pct_errors.extend(r_pct.tolist())

        # r bias for r_true < 0.001 (near-zero r detection)
        r_nearzero_mask = r_true < 0.001
        if r_nearzero_mask.any():
            r_bias = (r_pred[r_nearzero_mask] - r_true[r_nearzero_mask]).mean()
            r_biases.append(r_bias.item())

        # τ percentage error
        tau_denom = tau_true_f.abs().clamp(min=0.01)
        tau_pct = (tau_pred - tau_true_f).abs() / tau_denom * 100
        tau_pct_errors.extend(tau_pct.tolist())

    results = {
        "r_pct_error": float(np.mean(r_pct_errors)) if r_pct_errors else float("inf"),
        "tau_pct_error": float(np.mean(tau_pct_errors)) if tau_pct_errors else float("inf"),
        "r_bias": float(np.mean(r_biases)) if r_biases else 0.0,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train SpectralCNN on Test 4 (joint r/τ estimation)"
    )
    parser.add_argument("--f_sky", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0,
                        help="Noise in μK-arcmin")
    parser.add_argument("--nside", type=int, default=16,
                        help="HEALPix NSIDE (default: 16)")
    parser.add_argument("--lmax", type=int, default=None,
                        help="Maximum multipole (default: 3*NSIDE-1)")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR scheduler: cosine (smooth decay) or plateau (step drops)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (0=disabled)")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--camb_cache", type=str, default=None,
                        help="NPZ file to cache/load CAMB spectra (saves ~1.5h on repeat runs)")
    parser.add_argument("--hdf5_path", type=str, default=None,
                        help="Path to pre-generated HDF5 dataset. When set, reads maps from disk "
                             "instead of on-the-fly generation. Much faster for NSIDE>=128. "
                             "The HDF5 file must contain train/val/test splits as created by "
                             "scripts/generate_test4_hdf5.py. Ignores --n_train/--n_val/--n_test "
                             "(sizes come from the HDF5 file).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers. 0 = main thread (safe on gpu-shared). "
                             "2-4 recommended with HDF5 on nodes with multiple CPU cores.")
    parser.add_argument("--cache_chunk_size", type=int, default=0,
                        help="Number of maps to cache in RAM per chunk for HDF5 mode. "
                             "0 = no caching (read each map from disk). "
                             "Recommended for NSIDE=128 on Lustre: 5000-10000 "
                             "(5K maps ≈ 11.5 GB RAM at NSIDE=128). "
                             "Converts slow random I/O into fast sequential reads.")
    args = parser.parse_args()

    if args.lmax is None:
        args.lmax = 3 * args.nside - 1

    # μK-arcmin → μK/pixel: σ_pix = σ_arcmin / sqrt(Ω_pix [arcmin²])
    noise_arcmin = args.noise_std
    npix = 12 * args.nside**2
    pixel_area_sr = 4.0 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_sr * (180.0 * 60.0 / np.pi)**2
    noise_uK = noise_arcmin / np.sqrt(pixel_area_arcmin2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"NSIDE = {args.nside}, LMAX = {args.lmax}")
    print(f"Noise σ_n = {noise_arcmin} μK-arcmin = {noise_uK:.4f} μK, f_sky = {args.f_sky}")
    print(f"r range: [{R_MIN}, {R_MAX}], τ range: [{TAU_MIN}, {TAU_MAX}]")
    print(f"Training maps: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Data mode: {'HDF5' if args.hdf5_path else 'on-the-fly'}, num_workers={args.num_workers}")
    if args.scheduler == "cosine":
        print(f"LR schedule: CosineAnnealingLR (T_max={args.max_epochs}, eta_min={args.lr*1e-4:.1e})")
    else:
        print(f"LR schedule: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Gradient clipping: {args.grad_clip if args.grad_clip > 0 else 'disabled'}")
    print(f"Early stopping: patience={args.patience}")

    # Create datasets (HDF5 or on-the-fly)
    use_hdf5 = args.hdf5_path is not None
    num_workers = args.num_workers

    if use_hdf5:
        # HDF5 mode: read pre-generated maps from disk (fast, no synfast bottleneck)
        print(f"\nUsing HDF5 dataset: {args.hdf5_path}")
        import h5py
        with h5py.File(args.hdf5_path, 'r') as f:
            print(f"  HDF5 attrs: nside={f.attrs.get('nside')}, f_sky={f.attrs.get('f_sky')}, "
                  f"noise_std={f.attrs.get('noise_std')}")
            print(f"  train: {f['train/maps'].shape}, val: {f['val/maps'].shape}, "
                  f"test: {f['test/maps'].shape}")
        train_dataset = HDF5RTauDataset(args.hdf5_path, split='train',
                                        cache_chunk_size=args.cache_chunk_size)
        val_dataset = HDF5RTauDataset(args.hdf5_path, split='val',
                                      cache_chunk_size=args.cache_chunk_size)
        test_dataset = HDF5RTauDataset(args.hdf5_path, split='test',
                                       cache_chunk_size=args.cache_chunk_size)
        # Override n_train/n_val/n_test from HDF5 sizes
        args.n_train = len(train_dataset)
        args.n_val = len(val_dataset)
        args.n_test = len(test_dataset)
        print(f"  Dataset sizes: train={args.n_train}, val={args.n_val}, test={args.n_test}")
    else:
        # On-the-fly mode: generate maps via hp.synfast in __getitem__
        print("\nGenerating datasets on-the-fly (CAMB spectra needed)...")
        # Shared mask for train/val/test (critical for SpectralCNN)
        from torch_harmonics_healpix.data_generation_test2 import create_sky_mask
        shared_rng = np.random.default_rng(0)
        shared_mask = create_sky_mask(args.f_sky, args.nside, shared_rng).astype(np.float32)

        # Pre-compute CAMB spectra ONCE (with optional FITS disk cache)
        # FITS cache layout: PrimaryHDU + ImageHDU "R_VALUES"/"TAU_VALUES" + ImageHDU "CL_EE"/"CL_BB"
        if args.camb_cache and os.path.exists(args.camb_cache):
            print(f"  Loading cached CAMB spectra from {args.camb_cache}")
            from astropy.io import fits as pf
            with pf.open(args.camb_cache) as hdul:
                r_values = np.array(hdul["R_VALUES"].data, dtype=np.float32)
                tau_values = np.array(hdul["TAU_VALUES"].data, dtype=np.float32)
                cl_ee_array = np.array(hdul["CL_EE"].data, dtype=np.float64)
                cl_bb_array = np.array(hdul["CL_BB"].data, dtype=np.float64)
            shared_camb = (r_values, tau_values, cl_ee_array, cl_bb_array)
            print(f"  Loaded {len(shared_camb[0])} spectra from cache")
        else:
            print(f"  Pre-computing {N_CAMB_SPECTRA} CAMB spectra (shared across all datasets)...")
            shared_camb = precompute_camb_spectra_r_tau(N_CAMB_SPECTRA, args.lmax, seed=142)
            if args.camb_cache:
                print(f"  Saving CAMB spectra to {args.camb_cache}")
                from astropy.io import fits as pf
                hdu_r = pf.ImageHDU(shared_camb[0])
                hdu_r.name = "R_VALUES"
                hdu_tau = pf.ImageHDU(shared_camb[1])
                hdu_tau.name = "TAU_VALUES"
                hdu_ee = pf.ImageHDU(shared_camb[2])
                hdu_ee.name = "CL_EE"
                hdu_bb = pf.ImageHDU(shared_camb[3])
                hdu_bb.name = "CL_BB"
                hdul = pf.HDUList([pf.PrimaryHDU(), hdu_r, hdu_tau, hdu_ee, hdu_bb])
                hdul.writeto(args.camb_cache, overwrite=True)
                print(f"  Cache saved ({os.path.getsize(args.camb_cache)/1e6:.1f} MB)")

        train_dataset = RTauDataset(
            args.n_train, args.nside, args.lmax,
            noise_uK, args.f_sky, seed=42,
            mask=shared_mask, camb_spectra=shared_camb,
        )
        val_dataset = RTauDataset(
            args.n_val, args.nside, args.lmax,
            noise_uK, args.f_sky, seed=1234,
            mask=shared_mask, camb_spectra=shared_camb,
        )
        test_dataset = RTauDataset(
            args.n_test, args.nside, args.lmax,
            noise_uK, args.f_sky, seed=9999,
            mask=shared_mask, camb_spectra=shared_camb,
        )
        # On-the-fly must use num_workers=0 on gpu-shared (1 CPU core)
        num_workers = 0

    # DataLoader with optional multiprocessing for HDF5
    worker_init_fn = _hdf5_worker_init if use_hdf5 and num_workers > 0 else None

    # Use ChunkShuffleSampler when cache_chunk_size is set — it groups indices
    # by chunk so the HDF5 cache stays hot (100% hit rate vs ~0% with random shuffle)
    train_sampler = None
    if use_hdf5 and args.cache_chunk_size > 0:
        train_sampler = ChunkShuffleSampler(len(train_dataset),
                                            args.cache_chunk_size)
        print(f"Using ChunkShuffleSampler: {train_sampler.n_chunks} chunks "
              f"of {args.cache_chunk_size} maps each")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=num_workers, worker_init_fn=worker_init_fn)

    # Model: 3 input channels (Q, U, mask), 2 outputs (r_log, τ)
    # Enable inpainting for partial-sky observations
    model = SpectralCNN(
        in_channels=3,
        out_channels=2,
        nside=args.nside,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        inpaint=(args.f_sky < 1.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.lr * 1e-4
        )
        use_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor
        )
        use_plateau = True

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_reached = 0

    print(f"\nEpoch | Train Loss | r %err | τ %err | LR | No Imp")
    print("-" * 70)

    train_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        # Update sampler epoch so each epoch gets a different chunk shuffle
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        epoch_time = time.time() - t0

        val_results = evaluate(model, val_loader, device)
        val_loss = val_results["r_pct_error"] + val_results["tau_pct_error"]  # Combined validation metric for early stopping (equal weight to both parameters)

        if use_plateau:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            # Save best checkpoint to disk (survives walltime kills)
            ckpt_dir = os.path.dirname(args.output) or "."
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = args.output.replace(".json", f"_nside{args.nside}_best.pt")
            torch.save(best_state, ckpt_path)
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_reached = epoch
        r_err = val_results["r_pct_error"]
        tau_err = val_results["tau_pct_error"]
        print(f"{epoch:4d} | {train_loss:10.6f} | {r_err:5.1f}% | {tau_err:5.1f}% | {lr:9.6f} | {epochs_no_improve:3d}  ({epoch_time:.1f}s)")

        if epochs_no_improve >= args.patience:
            print(f"\n*** Early stopping at epoch {epoch} ***")
            break

    train_time = time.time() - train_start

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    print(f"\n=== Evaluation on {args.n_test} test maps ===")
    results = evaluate(model, test_loader, device)

    print(f"\nCNN r mean % error: {results['r_pct_error']:.1f}%")
    print(f"CNN τ mean % error: {results['tau_pct_error']:.1f}%")
    print(f"CNN r bias (near-zero r): {results['r_bias']:.6f}")

    # Fisher forecast comparison (placeholder)
    print(f"\nFisher forecast comparison:")
    print(f"  (Fisher forecasts for joint r/τ not yet implemented — placeholder)")

    # GPU memory usage
    gpu_memory_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nPeak GPU memory: {gpu_memory_mb:.1f} MB")

    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "test": "test4_joint_r_tau",
            "data_mode": "hdf5" if args.hdf5_path else "on-the-fly",
            "hdf5_path": args.hdf5_path,
            "r_pct_error": float(results["r_pct_error"]),
            "tau_pct_error": float(results["tau_pct_error"]),
            "r_bias": float(results["r_bias"]),
            "n_params": int(n_params),
            "noise_std_uK_arcmin": float(noise_arcmin),
            "noise_std_uK": float(noise_uK),
            "f_sky": float(args.f_sky),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "epochs_reached": epoch_reached,
            "best_val_loss": float(best_val_loss),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_schedule": args.scheduler,
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "early_stopping_patience": args.patience,
            "hidden_channels": args.hidden_channels,
            "num_blocks": args.num_blocks,
            "inpaint": bool(args.f_sky < 1.0),
            "nside": args.nside,
            "lmax": args.lmax,
            "train_time_s": train_time,
            "gpu_memory_mb": gpu_memory_mb,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save best model weights (include nside in filename)
    if best_state is not None:
        model_path = args.output.replace(".json", f"_nside{args.nside}.pt")
        torch.save(best_state, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
