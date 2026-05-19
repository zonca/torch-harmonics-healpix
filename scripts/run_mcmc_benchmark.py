#!/usr/bin/env python3
"""Run MCMC baseline benchmark for Test 1 (ℓ_p estimation).

Reproduces the maximum-likelihood baseline from Krachmalnicoff & Tomasi (2019)
Table 1. Generates maps at each noise level, estimates ℓ_p via MCMC,
and reports mean percentage error.
"""

import numpy as np
import sys
import time

from torch_harmonics_healpix.data_generation import (
    generate_map,
    NSIDE,
    LMAX,
    SIGMA_P,
)
from torch_harmonics_healpix.mcmc_baseline import mcmc_estimate_ell_p


N_TEST = 50  # Small test set for quick run; use 1000 for full benchmark
NOISE_LEVELS = [0, 5, 10, 15]


def run_benchmark(n_test=N_TEST, seed=42):
    """Run MCMC baseline for all noise levels."""
    rng = np.random.default_rng(seed)

    print(f"MCMC Baseline Benchmark (n_test={n_test})")
    print(f"{'Noise σ_n':>10} | {'MCMC % error':>12} | {'Time (s)':>10}")
    print("-" * 40)

    for noise_std in NOISE_LEVELS:
        # Generate random ℓ_p values
        ell_p_true = rng.uniform(5, 20, size=n_test).astype(np.float32)

        errors = []
        t0 = time.time()

        for i in range(n_test):
            m = generate_map(
                ell_p_true[i],
                nside=NSIDE,
                lmax=LMAX,
                sigma_p=SIGMA_P,
                noise_std=noise_std,
                rng=rng,
            )
            ell_p_est = mcmc_estimate_ell_p(
                m, sigma_p=SIGMA_P, lmax=LMAX, noise_std=noise_std, nside=NSIDE
            )
            pct_error = abs(ell_p_est - ell_p_true[i]) / ell_p_true[i] * 100
            errors.append(pct_error)

        elapsed = time.time() - t0
        mean_error = np.mean(errors)

        print(f"{noise_std:>10} | {mean_error:>11.1f}% | {elapsed:>9.1f}")

    # Paper baselines for reference
    print("\nPaper baselines (Krachmalnicoff & Tomasi 2019, Table 1):")
    print(f"{'Noise σ_n':>10} | {'NNhealpix':>10} | {'MCMC':>10}")
    print("-" * 38)
    for noise, nnh, mcmc in [(0, "1.3%", "0.7%"), (5, "2.9%", "2.5%"),
                              (10, "5.2%", "4.8%"), (15, "8.4%", "7.8%")]:
        print(f"{noise:>10} | {nnh:>10} | {mcmc:>10}")


if __name__ == "__main__":
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else N_TEST
    run_benchmark(n_test=n_test)
