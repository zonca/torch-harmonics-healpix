#!/usr/bin/env python3
"""Summarize all benchmark results from the torch-harmonics-healpix project.

Reads result JSON files from the results/ directory and produces
a comprehensive comparison table.

Usage:
    python scripts/summarize_results.py [--results_dir results/]
"""

import json
import os
import sys
import argparse
from pathlib import Path


def load_json(path):
    """Load JSON file, return None if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    """Load all result JSONs and print summary comparison tables."""
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    rdir = Path(args.results_dir)

    # ============================================================
    # Test 1: ℓ_p estimation from scalar (T) maps
    # ============================================================
    print("=" * 70)
    print("TEST 1: ℓ_p estimation from scalar (T) maps")
    print("=" * 70)

    # Paper baselines
    nnhealpix = {0: 1.3, 5: 2.9, 10: 5.2, 15: 8.4}
    mcmc_paper = {0: 0.7, 5: 2.5, 10: 4.8, 15: 7.8}

    # v1 results
    v1 = load_json(rdir / "test1_spectralcnn_v1.json")
    # v2 results
    v2 = load_json(rdir / "test1_spectralcnn_v2.json")

    # v3 results (separate files per noise level)
    v3 = {}
    for noise in [0, 5, 10, 15]:
        data = load_json(rdir / f"test1_v3_noise{noise}.json")
        if data:
            v3[noise] = data.get("cnn_pct_error")

    # MCMC baseline (1000 maps on Popeye)
    mcmc_ours = load_json(rdir / "mcmc_1000maps_popeye.json")

    # Print comparison table
    print(f"\n{'σ_n':>4} | {'NNhealpix':>10} | {'MCMC(paper)':>11} | {'MCMC(ours)':>10} | "
          f"{'v1(σp=3)':>9} | {'v2(σp=5)':>9} | {'v3(multi)':>10}")
    print("-" * 80)

    for noise in [0, 5, 10, 15]:
        nnh = nnhealpix.get(noise, "-")
        mcmcp = mcmc_paper.get(noise, "-")
        mcmco = "-"
        if mcmc_ours and f"noise_{noise}" in mcmc_ours.get("results", {}):
            mcmco = f"{mcmc_ours['results'][f'noise_{noise}']['mcmc_pct_error']:.1f}%"

        v1_val = "-"
        if v1 and f"noise_{noise}" in v1.get("results", {}):
            v1_val = f"{v1['results'][f'noise_{noise}']['cnn_pct_error']:.1f}%"

        v2_val = "-"
        if v2 and f"noise_{noise}" in v2.get("results", {}):
            r = v2["results"][f"noise_{noise}"]
            v2_val = f"{r['cnn_pct_error']:.1f}%"

        v3_val = "-"
        if noise in v3 and v3[noise] is not None:
            v3_val = f"{v3[noise]:.1f}%"

        print(f"{noise:4d} | {nnh:>9}% | {mcmcp:>10}% | {mcmco:>10} | "
              f"{v1_val:>9} | {v2_val:>9} | {v3_val:>10}")

    # ============================================================
    # Test 2: ℓ_Ep/ℓ_Bp from Q/U polarization maps
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: ℓ_Ep/ℓ_Bp from Q/U polarization maps")
    print("=" * 70)

    # Paper baselines (NNhealpix, full sky, no noise)
    paper_t2 = {
        1.0:  (2.7, 2.7),
        0.5:  (3.9, 3.9),
        0.2:  (5.3, 5.3),
        0.1:  (6.4, 6.4),
        0.05: (8.4, 8.4),
    }

    print(f"\n{'f_sky':>6} | {'NNhealpix Ep/Bp':>16} | {'SpectralCNN Ep':>15} | {'SpectralCNN Bp':>15}")
    print("-" * 65)

    for fsky in [1.0, 0.5, 0.2, 0.1, 0.05]:
        nnh = paper_t2.get(fsky, ("-", "-"))

        data = load_json(rdir / f"test2_v2_fsky{fsky}.json")
        if data:
            ep = f"{data.get('ep_pct_error', 0):.1f}%"
            bp = f"{data.get('bp_pct_error', 0):.1f}%"
        else:
            ep = bp = "-"

        print(f"{fsky:6.2f} | {nnh[0]:>7.1f}%/{nnh[1]:>5.1f}% | {ep:>15} | {bp:>15}")

    # ============================================================
    # Test 3: τ estimation from Q/U maps
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: τ estimation from Q/U maps")
    print("=" * 70)

    print(f"\n{'Method':<20} | {'Mean % error':>12}")
    print("-" * 35)
    print(f"{'NNhealpix':<20} | {'4.0%':>12}")
    print(f"{'MCMC (paper)':<20} | {'2.8%':>12}")

    data = load_json(rdir / "test3_v2.json")
    if data:
        print(f"{'SpectralCNN v2':<20} | {data.get('tau_pct_error', 0):>11.1f}%")
    else:
        print(f"{'SpectralCNN v2':<20} | {'-':>12}")

    # ============================================================
    # Architecture comparison summary
    # ============================================================
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 70)
    print("""
  SpectralCNN (v2): Matches NNhealpix at σ_n=0 (1.3% vs 1.3%)
                    Underperforms at high noise (11.8% vs 8.4% at σ_n=15)
                    Gap grows linearly with noise level
                    Root cause: fixed-resolution spectral blocks lack
                    multi-scale information that NNhealpix's progressive
                    pooling provides

  MultiResSpectralCNN (v3): Tests multi-resolution hypothesis
                    Decreasing ℓ_max (47→23→11→5) mimics NNhealpix pooling
                    1.5M params vs 6.4M (more parameter-efficient)
                    Complete: no meaningful improvement over v2
                    σ_n=0: 1.5%, σ_n=5: 3.5%, σ_n=10: 6.7%, σ_n=15: 11.3%
                    Multi-resolution is NOT the missing ingredient

  SpectralCNN on Polarization: Significantly outperforms NNhealpix!
                    f_sky=1.0: ℓ_Ep=1.5%, ℓ_Bp=1.6% (NNhealpix: 2.7%/2.7%)
                    ~43% improvement from global SHT features on Q/U maps

  Key limitation: torch-harmonics 0.8.0 VectorSHT (spin-2) is too slow
  for practical training. Both architectures use scalar Q/U processing
  for Test 2, missing E/B mode separation advantage.
""")


if __name__ == "__main__":
    main()
