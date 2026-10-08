#!/usr/bin/env python3
"""Run MCMC baselines for Tests 2 and 3 (production, 1000 maps each).

Test 2: ℓ_Ep/ℓ_Bp estimation at f_sky = 1.0, 0.5, 0.2, 0.1, 0.05
        (KT19 masks: circular caps of 90/53/37/26° radius → 50/20/10/5%).
Test 3: τ estimation with the full 5000-spectrum CAMB grid.

Writes one JSON per test into --output-dir (default: results_v4/).
"""

import argparse
import json
import os
import time

import numpy as np

from torch_harmonics_healpix.mcmc_baselines_test2_3 import (
    evaluate_mcmc_test2,
    evaluate_mcmc_test3,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_maps", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="results_v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsky", type=float, nargs="*", default=None,
                   help="Run only these f_sky values for Test 2")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fsky_values = args.fsky if args.fsky else [1.0, 0.5, 0.2, 0.1, 0.05]

    # ---------------- Test 2 ----------------
    t2 = {"version": "v4_exact_lik", "n_maps": args.n_maps, "seed": args.seed, "fsky": {}}
    for fsky in fsky_values:
        print(f"\n=== Test 2: f_sky = {fsky} ===", flush=True)
        t0 = time.time()
        res = evaluate_mcmc_test2(n_maps=args.n_maps, f_sky=fsky, seed=args.seed)
        res["wall_time_s"] = time.time() - t0
        print(f"  ep {res['ep_pct_error']:.2f}%  bp {res['bp_pct_error']:.2f}%  "
              f"({res['time_per_map']:.2f}s/map)", flush=True)
        t2["fsky"][str(fsky)] = res
        # Write incrementally so a preemption/timelimit keeps partial results
        with open(os.path.join(args.output_dir, "mcmc_test2_v4.json"), "w") as f:
            json.dump(t2, f, indent=2)

    # ---------------- Test 3 ----------------
    print(f"\n=== Test 3: {args.n_maps} maps ===", flush=True)
    t0 = time.time()
    res = evaluate_mcmc_test3(n_maps=args.n_maps, seed=args.seed)
    res["wall_time_s"] = time.time() - t0
    print(f"  tau {res['tau_pct_error']:.2f}%  ({res['time_per_map']:.2f}s/map)", flush=True)
    with open(os.path.join(args.output_dir, "mcmc_test3_v4.json"), "w") as f:
        json.dump({"version": "v4_exact_lik", "n_maps": args.n_maps,
                    "seed": args.seed, **res}, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
