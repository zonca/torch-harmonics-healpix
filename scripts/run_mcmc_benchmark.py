#!/usr/bin/env python3
"""Run MCMC baseline benchmark for Test 1 (ℓ_p estimation).

Reproduces the maximum-likelihood baseline from Krachmalnicoff & Tomasi (2019)
Table 1. Generates maps at each noise level, estimates ℓ_p via MCMC,
and reports mean percentage error.

v4 (2026-10): optionally writes results to JSON (--output), matching the
layout of the Test 2/3 baselines for BENCHMARKS.md aggregation.
"""

import argparse
import json
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


NOISE_LEVELS = [0, 5, 10, 15]


def run_benchmark(n_test=50, seed=42, output=None):
    """Run MCMC baseline for all noise levels.

    Args:
        n_test (int): Number of test maps per noise level.
        seed (int): Random seed for reproducibility.
        output (str): Optional path to write JSON results.
    """
    rng = np.random.default_rng(seed)

    print(f"MCMC Baseline Benchmark (n_test={n_test})")
    print(f"{'Noise σ_n':>10} | {'MCMC % error':>12} | {'Time (s)':>10}")
    print("-" * 40)

    results = {"version": "v4_exact_lik", "n_test": n_test, "seed": seed, "levels": []}

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
        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))

        print(f"{noise_std:>10} | {mean_error:>11.2f}% | {elapsed:>9.1f}")
        results["levels"].append({
            "noise_std": noise_std,
            "mean_pct_error": mean_error,
            "median_pct_error": median_error,
            "time_per_map_s": elapsed / n_test,
        })

    # Paper baselines for reference
    print("\nPaper baselines (Krachmalnicoff & Tomasi 2019, Table 1):")
    print(f"{'Noise σ_n':>10} | {'NNhealpix':>10} | {'MCMC':>10}")
    print("-" * 38)
    for noise, nnh, mcmc in [(0, "1.3%", "0.7%"), (5, "2.9%", "2.5%"),
                              (10, "5.2%", "4.8%"), (15, "8.4%", "7.8%")]:
        print(f"{noise:>10} | {nnh:>10} | {mcmc:>10}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n_test", type=int, nargs="?", default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    run_benchmark(n_test=args.n_test, seed=args.seed, output=args.output)
