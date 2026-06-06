# N128 Training Results (v2 — C_ℓ fix)

## Training Summary

All 4 configs trained for 36h walltime (~18 epochs max). Jobs expired before completing full 25 epochs.

| Config | Job | Final Epoch | Best Epoch | Final Loss | r %err (best) | τ %err (best) |
|--------|-----|-------------|------------|------------|---------------|---------------|
| fsky=1.0, noise=0 | 50387243 | 16 | 13 | 0.095 | 58.9% | 24.6% |
| fsky=1.0, noise=6 | 50387244 | 16 | 12 | 0.102 | 61.2% | 24.6% |
| fsky=0.1, noise=0 | 50387246 | 18 | 14 | 1.129 | 64.6% | 25.7% |
| fsky=0.1, noise=6 | 50433550 | 17 | 12 | 1.134 | 62.3% | 24.6% |

## Test Set Evaluation (1000 maps per config)

Best checkpoints evaluated on held-out test sets:

| Config | r median %err | r mean %err | τ median %err | τ mean %err |
|--------|---------------|-------------|---------------|-------------|
| fsky=1.0, noise=0 | 83.1% | 1947% | 24.5% | 24.7% |
| fsky=1.0, noise=6 | 124.6% | 2453% | 21.7% | 24.7% |
| fsky=0.1, noise=0 | 83.2% | 2032% | 22.1% | 25.0% |
| fsky=0.1, noise=6 | 84.0% | 1426% | 23.4% | 24.3% |

**Note**: Mean r error is heavily skewed by extreme outliers (p95: 4700-10000%). Median is the representative metric.

## Fisher Information Comparison (Cramér-Rao bounds)

Fisher limits at fiducial r=0.003, τ=0.054:

| Config | Fisher σ_r% | Fisher σ_τ% | CNN r median% | CNN τ median% | r ratio | τ ratio |
|--------|-------------|-------------|---------------|---------------|---------|---------|
| fsky=1.0, noise=0 | 7.5% | 2.0% | 83.1% | 24.5% | 11.1× | 12.3× |
| fsky=1.0, noise=6 | 18.5% | 2.5% | 124.6% | 21.7% | 6.7× | 8.7× |
| fsky=0.1, noise=0 | 23.8% | 2.5% | 83.2% | 22.1% | 3.5× | 8.8× |
| fsky=0.1, noise=6 | 58.5% | 7.8% | 84.0% | 23.4% | 1.4× | 3.0× |

**Key finding**: N128 results are significantly worse than Fisher limits (3.5-11× for r). This is expected because:
1. Models are undertrained (16-18 epochs vs target 25+)
2. 422M parameter model needs many more epochs to converge
3. Validation-based checkpointing may not optimize test r error directly

## N16 vs N128 Comparison

N16 (6.6M params, fully trained) vs N128 (422M params, undertrained):

| Config | N16 r% | N128 r median% | N16 τ% | N128 τ median% |
|--------|--------|----------------|--------|-----------------|
| fsky=1.0, noise=0 | 24.9% | 83.1% | 17.8% | 24.5% |
| fsky=1.0, noise=6 | 35.4% | 124.6% | 18.6% | 21.7% |
| fsky=0.1, noise=0 | 58.0% | 83.2% | 25.7% | 22.1% |
| fsky=0.1, noise=6 | 59.7% | 84.0% | 23.7% | 23.4% |

N16 consistently outperforms N128 for r despite fewer parameters. N128 needs more training epochs to realize its potential advantage.

## Checkpoints

Best model checkpoints saved to:
- `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2/test4_nside128_fsky*_noise*`

Eval JSONs:
- `/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v2/test4_nside128_*_eval.json`

## Next Steps

To improve N128 results:
1. Add resume support to training script (currently doesn't support continuing from checkpoint)
2. Train for 25+ epochs with proper learning rate scheduling
3. Consider reducing batch size or using gradient accumulation for better convergence
4. Evaluate whether the 422M param model is appropriate for 100K training maps

