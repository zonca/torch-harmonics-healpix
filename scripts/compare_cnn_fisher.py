#!/usr/bin/env python3
"""Compare CNN results against Fisher information matrix bounds.

Loads training results (JSON) and Fisher baselines (JSON) from results_v3/
and produces a comparison table showing how close the CNN gets to the
optimal Cramér-Rao bound.

Usage:
    python3 scripts/compare_cnn_fisher.py --results_dir results_v3
"""

import argparse
import json
import glob
import os
import sys


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare CNN results against Fisher information bounds"
    )
    parser.add_argument("--results_dir", type=str, default="results_v3",
                        help="Directory containing results JSON files")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: results directory {results_dir} not found")
        sys.exit(1)

    # Load Fisher results
    fisher_files = sorted(glob.glob(os.path.join(results_dir, "test4_fisher_nside*.json")))
    if not fisher_files:
        print("No Fisher results found. Run run_fisher_mcmc_test4.py first.")
        sys.exit(1)

    fisher = {}
    for f in fisher_files:
        data = load_json(f)
        config = data.get('config', f"fsky{data['f_sky']}_noise{int(data.get('noise_arcmin', 0))}")
        key = f"nside{data['nside']}_{config}"
        fisher[key] = data

    # Load CNN results
    cnn_files = sorted(glob.glob(os.path.join(results_dir, "test4_nside*.json")))
    # Filter out Fisher files
    cnn_files = [f for f in cnn_files if "fisher" not in f and "mcmc" not in f]
    if not cnn_files:
        print("No CNN results found. Run train_test4.py first.")
        sys.exit(1)

    cnn = {}
    for f in cnn_files:
        data = load_json(f)
        # CNN JSONs may not have 'config' — construct from f_sky and noise
        if 'config' not in data:
            noise_val = data.get('noise_std_uK_arcmin', 0)
            cfg_label = f"fsky{data['f_sky']}_noise{int(noise_val)}"
            data['config'] = cfg_label
        key = f"nside{data['nside']}_{data['config']}"
        cnn[key] = data

    # MCMC results (optional)
    mcmc_files = sorted(glob.glob(os.path.join(results_dir, "test4_mcmc_mh_nside*.json")))
    mcmc = {}
    for f in mcmc_files:
        data = load_json(f)
        key = f"nside{data['nside']}_{data['config']}"
        mcmc[key] = data

    # Print comparison table
    print("=" * 100)
    print("CNN vs Fisher Information Matrix Bounds")
    print("=" * 100)
    print("\nNOTE: MCMC pseudo-C_ℓ chains drift to r→r_max boundary (~230% error).")
    print("      This is a known limitation: single-realization C_ℓ estimates are too")
    print("      noisy for pseudo-C_ℓ MCMC to constrain r. Fisher matrix is the")
    print("      correct Cramér-Rao baseline (standard in CMB literature).")
    print("      CNN N128 results use hidden_channels=8 (underfitted); hc=32 training pending.\n")

    configs = sorted(set(list(fisher.keys()) + list(cnn.keys())))

    for key in configs:
        if key not in fisher:
            print(f"\n{key}: No Fisher baseline")
            continue

        f = fisher[key]
        f_r_pct = f.get('r_fisher_pct', f.get('r_pct_error', 0))
        f_tau_pct = f.get('tau_fisher_pct', f.get('tau_pct_error', 0))
        f_sigma_r = f.get('sigma_r_fisher', f.get('sigma_r', 0))
        f_sigma_tau = f.get('sigma_tau_fisher', f.get('sigma_tau', 0))
        print(f"\n--- {key} (f_sky={f['f_sky']}, noise={f['noise_arcmin']} μK-arcmin) ---")
        print(f"  Fisher: σ_r={f_sigma_r:.6f} ({f_r_pct:.1f}%), "
              f"σ_τ={f_sigma_tau:.6f} ({f_tau_pct:.1f}%)")

        if key in cnn:
            c = cnn[key]
            cnn_r_factor = c['r_pct_error'] / f_r_pct if f_r_pct > 0 else float('inf')
            cnn_tau_factor = c['tau_pct_error'] / f_tau_pct if f_tau_pct > 0 else float('inf')
            print(f"  CNN:    r_err={c['r_pct_error']:.1f}% ({cnn_r_factor:.2f}× Fisher), "
                  f"τ_err={c['tau_pct_error']:.1f}% ({cnn_tau_factor:.2f}× Fisher)")
            print(f"         n_params={c['n_params']:,}, epochs={c['epochs_reached']}, "
                  f"hidden={c['hidden_channels']}, blocks={c['num_blocks']}")

        if key in mcmc:
            m = mcmc[key]
            mcmc_r_factor = m['r_pct_error'] / f_r_pct if f_r_pct > 0 else float('inf')
            mcmc_tau_factor = m['tau_pct_error'] / f_tau_pct if f_tau_pct > 0 else float('inf')
            print(f"  MCMC:   r_err={m['r_pct_error']:.1f}% ({mcmc_r_factor:.2f}× Fisher), "
                  f"τ_err={m['tau_pct_error']:.1f}% ({mcmc_tau_factor:.2f}× Fisher)")
            print(f"         method={m['method']}, n_test={m['n_test']}, "
                  f"steps={m['mcmc_steps']}, cpu_time={m['cpu_time_seconds']:.0f}s")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (r % error / Fisher σ_r %)")
    print("=" * 100)
    print(f"{'Config':<25} {'Fisher σ_r%':>12} {'CNN r%':>8} {'CNN/Fisher':>10} {'MCMC r%':>8} {'MCMC/Fisher':>10}")
    print("-" * 75)

    for key in configs:
        if key not in fisher:
            continue
        f = fisher[key]
        fisher_r_pct = f.get('r_fisher_pct', f.get('r_pct_error', 0))

        if key in cnn:
            c = cnn[key]
            cnn_r = c['r_pct_error']
            cnn_ratio = f"{cnn_r / fisher_r_pct:.2f}×" if fisher_r_pct > 0 else "—"
        else:
            cnn_r = "—"
            cnn_ratio = "—"

        if key in mcmc:
            m = mcmc[key]
            mcmc_r = m['r_pct_error']
            mcmc_ratio = f"{mcmc_r / fisher_r_pct:.2f}×" if fisher_r_pct > 0 else "—"
        else:
            mcmc_r = "—"
            mcmc_ratio = "—"

        label = f"n{key.split('_')[0].replace('nside', '')} {key.split('_')[1]} {key.split('_')[2]}"
        cnn_r_str = f"{cnn_r:.1f}%" if isinstance(cnn_r, (int, float)) else cnn_r
        mcmc_r_str = f"{mcmc_r:.1f}%" if isinstance(mcmc_r, (int, float)) else mcmc_r
        print(f"{label:<25} {fisher_r_pct:>11.1f}% {cnn_r_str:>8} {str(cnn_ratio):>10} {mcmc_r_str:>8} {str(mcmc_ratio):>10}")

    # Save summary as JSON
    summary = {}
    for key in configs:
        if key not in fisher:
            continue
        fdata = fisher[key]
        fisher_r_pct = fdata.get('r_fisher_pct', fdata.get('r_pct_error', 0))
        fisher_tau_pct = fdata.get('tau_fisher_pct', fdata.get('tau_pct_error', 0))
        entry = {
            "fisher_sigma_r": fdata.get('sigma_r_fisher', fdata.get('sigma_r', 0)),
            "fisher_sigma_tau": fdata.get('sigma_tau_fisher', fdata.get('sigma_tau', 0)),
            "fisher_r_pct": fisher_r_pct,
            "fisher_tau_pct": fisher_tau_pct,
        }
        if key in cnn:
            c = cnn[key]
            entry.update({
                "cnn_r_pct": c['r_pct_error'],
                "cnn_tau_pct": c['tau_pct_error'],
                "cnn_r_ratio": c['r_pct_error'] / fisher_r_pct if fisher_r_pct > 0 else None,
                "cnn_tau_ratio": c['tau_pct_error'] / fisher_tau_pct if fisher_tau_pct > 0 else None,
                "cnn_n_params": c['n_params'],
            })
        if key in mcmc:
            m = mcmc[key]
            entry.update({
                "mcmc_r_pct": m['r_pct_error'],
                "mcmc_tau_pct": m['tau_pct_error'],
                "mcmc_r_ratio": m['r_pct_error'] / fisher_r_pct if fisher_r_pct > 0 else None,
                "mcmc_tau_ratio": m['tau_pct_error'] / fisher_tau_pct if fisher_tau_pct > 0 else None,
            })
        summary[key] = entry

    summary_path = os.path.join(results_dir, "cnn_vs_fisher_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
