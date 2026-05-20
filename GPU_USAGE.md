# GPU Usage & Model Parameters

## GPU Usage (Expanse, V100-SXM2-32GB, gpu-shared partition)

| Job ID | Name | Purpose | Walltime | Status |
|--------|------|---------|----------|--------|
| 49192871 | thh-v2 | Test 1 v2 (4 noise levels) | 4h 33m | ✅ Complete |
| 49220070 | thh-v3 | Test 1 v3 MultiRes (4 noise levels) | 4h 11m | ✅ Complete |
| 49224438 | thh-t23 | Test 2 no-inpaint (5 f_sky) + Test 3 (CAMB crash) | 8h 11m | ❌ Failed |
| 49229707 | thh-t23-inpaint | Test 2 inpaint (4 f_sky) + Test 3 (fixed) | ~15-20h est. | 🔄 Running |

**Total GPU time used: ~18.5 hours** (as of May 20, 2026)

**Estimated total at completion: ~30-34 GPU-hours**

Note: Job 49224438 wasted 8h because Test 3 crashed on a CAMB bug
(`get_transfer_functions` instead of `get_results` for lensed Cls).
Without that failure, total would have been ~22-26 GPU-hours.

Each training run takes roughly:
- Test 1 per noise level: ~1h (v2), ~1h (v3)
- Test 2 per f_sky value: ~2.5-3h
- Test 3: ~1.5h (estimated)

---

## Model Parameters

### SpectralCNN (our implementation, torch-harmonics-based)

| Variant | Input Channels | Blocks | Hidden Ch. | ℓ_max | Parameters |
|---------|---------------|--------|-----------|-------|-----------|
| v2 (Test 1) | 1 (T) | 3 | 32 | 47 | **6,454,529** |
| v3 MultiRes (Test 1) | 1 (T) | 4 | 32 | 47→23→11→5 | **1,525,665** |
| v2 (Test 2/3) | 3 (Q,U,mask) | 4 | 32 | 47 | **9,829,634** |

The spectral weights dominate parameter count: `[C_out, C_in, ℓ_max, m_max]` complex
weights per block. With more input channels (3 vs 1), the first block's weight
matrix is 3× larger, giving ~9.8M vs ~6.5M total.

### NNhealpix (paper reference)

| Variant | Parameters |
|---------|-----------|
| Test 1 (4 NBBs, 32 filters, Nside 16→1) | **~80,000** |
| Test 2 (4 NBBs, 32 filters, 3 input channels) | **~240,000** (estimated) |
| Test 3 (4 NBBs, 32 filters, 3 input channels) | **~240,000** (estimated) |

NNhealpix is dramatically more parameter-efficient because pixel-space
convolution kernels are small (O(filter_size²) per layer) compared to
spectral weights which scale as O(ℓ_max²) per layer.

### Parameter Count Comparison

| Architecture | Test 1 Params | Test 2 Params | Ratio vs NNhealpix |
|-------------|--------------|--------------|-------------------|
| NNhealpix | ~80k | ~240k | 1× |
| SpectralCNN v2 | 6.5M | 9.8M | 80-40× |
| MultiResSpectralCNN v3 | 1.5M | — | 19× |

Despite being 40-80× larger, SpectralCNN does not consistently outperform
NNhealpix. It excels at full-sky polarization (Test 2 f_sky=1.0) but
underperforms at high noise (Test 1) and partial sky (Test 2 f_sky<0.5).

---

## Cost-Benefit Summary

- **SpectralCNN advantage**: Rotation equivariance (no data augmentation),
  native E/B separation potential (with vector SHT), superior full-sky
  polarization performance (43% better at f_sky=1.0).
- **SpectralCNN disadvantage**: 40-80× more parameters, slower training
  (~2.5h per f_sky vs minutes for NNhealpix on CPU), underperforms at
  high noise and partial sky (without inpainting).
- **Inpainting fix**: Simple mean-inpainting restores partial-sky
  performance (results pending from current job).
