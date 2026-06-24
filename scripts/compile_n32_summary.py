#!/usr/bin/env python3
"""Compile all NSIDE=32 results into a single summary JSON.

Reads:
  - Fisher bounds: results_v3/fisher_fixed_lmax_verification.json
  - CNN hc=32: results_v3/test4_nside32_hc32_*.json
  - CNN hc=64: results_v3/test4_nside32_hc64_*.json

Writes:
  - results_v3/n32_comprehensive_summary.json
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_v3')

CONFIGS = [
    (1.0, 0, "fsky1.0_noise0"),
    (1.0, 6, "fsky1.0_noise6"),
    (0.1, 0, "fsky0.1_noise0"),
    (0.1, 6, "fsky0.1_noise6"),
]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    fisher = load_json(os.path.join(RESULTS_DIR, 'fisher_fixed_lmax_verification.json'))
    summary = {}

    for fsky, noise, label in CONFIGS:
        fisher_key = f"nside32_fsky{fsky}_noise{noise}"
        f = fisher[fisher_key]

        entry = {
            "nside": 32,
            "f_sky": fsky,
            "noise_arcmin": noise,
            "fisher": {
                "sigma_r": f["sigma_r"],
                "sigma_tau": f["sigma_tau"],
                "r_pct_error": f["r_pct_error"],
                "tau_pct_error": f["tau_pct_error"],
                "correlation_r_tau": f.get("correlation_r_tau", 0),
            },
        }

        # hc=32 results
        hc32_path = os.path.join(RESULTS_DIR, f"test4_nside32_hc32_{label}.json")
        if os.path.exists(hc32_path):
            hc32 = load_json(hc32_path)
            entry["hc32"] = {
                "r_pct_error": hc32["r_pct_error"],
                "tau_pct_error": hc32["tau_pct_error"],
                "r_bias": hc32.get("r_bias", None),
                "n_params": hc32.get("n_params", 26480834),
                "epochs": hc32.get("epochs_reached", None),
                "cnn_fisher_ratio_r": hc32["r_pct_error"] / f["r_pct_error"],
            }

        # hc=64 results
        hc64_path = os.path.join(RESULTS_DIR, f"test4_nside32_hc64_{label}.json")
        if os.path.exists(hc64_path):
            hc64 = load_json(hc64_path)
            entry["hc64"] = {
                "r_pct_error": hc64["r_pct_error"],
                "tau_pct_error": hc64["tau_pct_error"],
                "r_bias": hc64.get("r_bias", None),
                "n_params": hc64.get("n_params", 103547074),
                "epochs": hc64.get("epochs_reached", None),
                "cnn_fisher_ratio_r": hc64["r_pct_error"] / f["r_pct_error"],
            }

        summary[label] = entry

    out_path = os.path.join(RESULTS_DIR, 'n32_comprehensive_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")

    # Print summary table
    print("\n=== NSIDE=32 Comprehensive Summary ===\n")
    print(f"{'Config':<20} {'Fisher σr%':<12} {'hc32 r%':<10} {'hc64 r%':<10} {'hc32/hc64':<10}")
    print("-" * 65)
    for label, entry in summary.items():
        fisher_r = entry["fisher"]["r_pct_error"]
        hc32_r = entry.get("hc32", {}).get("r_pct_error", "N/A")
        hc64_r = entry.get("hc64", {}).get("r_pct_error", "N/A")
        ratio = "N/A"
        if isinstance(hc32_r, float) and isinstance(hc64_r, float):
            ratio = f"{hc32_r/hc64_r:.2f}×"
        print(f"{label:<20} {fisher_r:<12.2f} {str(hc32_r):<10} {str(hc64_r):<10} {ratio:<10}")

if __name__ == '__main__':
    main()
