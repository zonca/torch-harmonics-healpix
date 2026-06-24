#!/usr/bin/env python3
"""Analyze N32 CNN capacity scaling results.

Compares CNN performance across hidden_channels = 32, 64, 128
against corrected Fisher bounds. Generates a summary JSON and
markdown report.

Usage:
    python scripts/analyze_n32_capacity_scaling.py \
        --results_dir /path/to/results_v3 \
        --fisher_json /path/to/fisher_fixed_lmax_verification.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import glob
import numpy as np


def load_cnn_results(results_dir, nside=32):
    """Load all N32 CNN result JSONs, grouped by hidden_channels."""
    pattern = os.path.join(results_dir, f"test4_nside{nside}_hc*_fsky*.json")
    files = glob.glob(pattern)

    results = {}  # {hc: {config_name: data}}

    for f in files:
        basename = os.path.basename(f)
        # Parse: test4_nside32_hc32_fsky1.0_noise0.json
        parts = basename.replace(".json", "").split("_")
        hc_str = [p for p in parts if p.startswith("hc")][0]
        hc = int(hc_str.replace("hc", ""))

        # Extract fsky and noise from filename
        fsky_part = [p for p in parts if p.startswith("fsky")][0]
        noise_part = [p for p in parts if p.startswith("noise")][0]
        fsky = float(fsky_part.replace("fsky", ""))
        noise = int(noise_part.replace("noise", ""))

        config_name = f"fsky{fsky}_noise{noise}"

        with open(f, "r") as fh:
            data = json.load(fh)

        if hc not in results:
            results[hc] = {}
        results[hc][config_name] = data

    return results


def load_fisher_results(fisher_json_path):
    """Load corrected Fisher bounds."""
    with open(fisher_json_path, "r") as f:
        return json.load(f)


def analyze_capacity_scaling(cnn_results, fisher_results, nside=32):
    """Generate capacity scaling analysis."""
    analysis = {
        "nside": nside,
        "configs": [],
        "capacity_scaling": {},
    }

    # Get sorted list of hidden channel counts
    hc_list = sorted(cnn_results.keys())

    # Get all config names
    all_configs = set()
    for hc in hc_list:
        all_configs.update(cnn_results[hc].keys())
    config_list = sorted(all_configs)

    # For each config, compare across capacities
    for config in config_list:
        config_data = {"config": config, "capacities": []}

        for hc in hc_list:
            if config not in cnn_results[hc]:
                continue

            cnn = cnn_results[hc][config]
            r_fid = 0.003

            config_data["capacities"].append({
                "hidden_channels": hc,
                "n_params": cnn.get("n_params", 0),
                "r_pct_error": cnn.get("r_pct_error", 0),
                "tau_pct_error": cnn.get("tau_pct_error", 0),
                "r_bias": cnn.get("r_bias", 0),
                "epochs_reached": cnn.get("epochs_reached", 0),
                "train_time_s": cnn.get("train_time_s", 0),
            })

        # Find matching Fisher bound
        fisher_r_pct = None
        fisher_tau_pct = None
        for key, fisher_data in fisher_results.items():
            if nside in key or str(nside) in key:
                fsky_match = False
                noise_match = False
                if f"fsky{config.split('_')[0].replace('fsky', '')}" in str(fisher_data):
                    fsky_match = True
                # Try to match config
                if "configs" in fisher_data:
                    for fc in fisher_data["configs"]:
                        if fc.get("name") == config or fc.get("config") == config:
                            fisher_r_pct = fc.get("sigma_r_pct", fc.get("r_pct_error"))
                            fisher_tau_pct = fc.get("sigma_tau_pct", fc.get("tau_pct_error"))
                            break

        config_data["fisher_r_pct"] = fisher_r_pct
        config_data["fisher_tau_pct"] = fisher_tau_pct

        analysis["configs"].append(config_data)

    # Generate capacity scaling summary
    for hc in hc_list:
        hc_data = {"hidden_channels": hc, "configs": {}}
        for config in config_list:
            if config in cnn_results.get(hc, {}):
                cnn = cnn_results[hc][config]
                hc_data["configs"][config] = {
                    "r_pct_error": cnn.get("r_pct_error", 0),
                    "tau_pct_error": cnn.get("tau_pct_error", 0),
                    "r_bias": cnn.get("r_bias", 0),
                    "n_params": cnn.get("n_params", 0),
                }
        analysis["capacity_scaling"][f"hc{hc}"] = hc_data

    return analysis


def generate_markdown_report(analysis, output_path):
    """Generate a publication-ready markdown report."""
    lines = []
    lines.append("# N32 CNN Capacity Scaling Study")
    lines.append("")
    lines.append("**Date:** 2026-06-21")
    lines.append("**Pipeline:** v3 (C_ℓ fix + Huber τ loss + Fisher lmax fix)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("This study tests whether increasing SpectralCNN capacity (hidden_channels)")
    lines.append("can break through the ~56–59% r error ceiling observed with hc=32.")
    lines.append("")
    lines.append("Tested configurations:")
    lines.append("- hc=32 (26.5M params) — baseline")
    lines.append("- hc=64 (~106M params) — 4× capacity")
    lines.append("- hc=128 (~424M params) — 16× capacity")
    lines.append("")
    lines.append("All configs trained on N32 HDF5 data (100K train, 10K val, 1K test)")
    lines.append("with cosine LR schedule, Huber τ loss, and early stopping (patience=15).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Results: r % Error vs Capacity")
    lines.append("")
    lines.append("| Config | hc=32 | hc=64 | hc=128 | Fisher |")
    lines.append("|--------|-------|-------|--------|--------|")

    for config_data in analysis["configs"]:
        config = config_data["config"]
        row = [config]
        for hc in [32, 64, 128]:
            found = False
            for cap in config_data["capacities"]:
                if cap["hidden_channels"] == hc:
                    row.append(f"{cap['r_pct_error']:.1f}%")
                    found = True
                    break
            if not found:
                row.append("—")
        fisher_r = config_data.get("fisher_r_pct")
        row.append(f"{fisher_r:.1f}%" if fisher_r else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Results: τ % Error vs Capacity")
    lines.append("")
    lines.append("| Config | hc=32 | hc=64 | hc=128 | Fisher |")
    lines.append("|--------|-------|-------|--------|--------|")

    for config_data in analysis["configs"]:
        config = config_data["config"]
        row = [config]
        for hc in [32, 64, 128]:
            found = False
            for cap in config_data["capacities"]:
                if cap["hidden_channels"] == hc:
                    row.append(f"{cap['tau_pct_error']:.1f}%")
                    found = True
                    break
            if not found:
                row.append("—")
        fisher_tau = config_data.get("fisher_tau_pct")
        row.append(f"{fisher_tau:.1f}%" if fisher_tau else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Results: r_bias vs Capacity")
    lines.append("")
    lines.append("| Config | hc=32 | hc=64 | hc=128 |")
    lines.append("|--------|-------|-------|--------|")

    for config_data in analysis["configs"]:
        config = config_data["config"]
        row = [config]
        for hc in [32, 64, 128]:
            found = False
            for cap in config_data["capacities"]:
                if cap["hidden_channels"] == hc:
                    row.append(f"{cap['r_bias']:.6f}")
                    found = True
                    break
            if not found:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Capacity ceiling:** If hc=64 and hc=128 show similar r error to hc=32,")
    lines.append("   the bottleneck is NOT model capacity but likely training data diversity")
    lines.append("   or loss function design.")
    lines.append("")
    lines.append("2. **Bias stability:** The Jensen's inequality r_bias (~0.0007) should remain")
    lines.append("   constant across capacities if the bias is purely from the log-r transform.")
    lines.append("")
    lines.append("3. **Fisher gap:** The gap between CNN r error and Fisher bound indicates")
    lines.append("   how much room for improvement exists. If larger models close this gap,")
    lines.append("   capacity was the limiting factor.")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze N32 CNN capacity scaling results"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with CNN result JSONs")
    parser.add_argument("--fisher_json", type=str, required=True,
                        help="Path to corrected Fisher bounds JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis")
    parser.add_argument("--nside", type=int, default=32,
                        help="HEALPix NSIDE (default: 32)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading CNN results...")
    cnn_results = load_cnn_results(args.results_dir, nside=args.nside)
    print(f"  Found results for hidden_channels: {sorted(cnn_results.keys())}")

    print("Loading Fisher results...")
    fisher_results = load_fisher_results(args.fisher_json)

    # Analyze
    print("Analyzing capacity scaling...")
    analysis = analyze_capacity_scaling(cnn_results, fisher_results, nside=args.nside)

    # Save JSON
    json_path = os.path.join(args.output_dir, "n32_capacity_scaling_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved analysis JSON: {json_path}")

    # Generate markdown report
    md_path = os.path.join(args.output_dir, "n32_capacity_scaling_report.md")
    generate_markdown_report(analysis, md_path)
    print(f"  Saved markdown report: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Capacity Scaling Summary")
    print("=" * 80)

    for config_data in analysis["configs"]:
        config = config_data["config"]
        print(f"\n  {config}:")
        for cap in config_data["capacities"]:
            hc = cap["hidden_channels"]
            r_pct = cap["r_pct_error"]
            tau_pct = cap["tau_pct_error"]
            r_bias = cap["r_bias"]
            n_params = cap["n_params"]
            print(f"    hc={hc:3d} ({n_params/1e6:.1f}M): r={r_pct:.1f}%, τ={tau_pct:.1f}%, bias={r_bias:.6f}")


if __name__ == "__main__":
    main()
