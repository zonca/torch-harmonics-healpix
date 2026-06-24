#!/usr/bin/env python3
"""Generate publication-ready figures for NSIDE=32 analysis.

Produces:
  - Fisher vs CNN comparison bar chart (hc=32 vs hc=64)
  - Capacity scaling: r error vs hidden_channels
  - r_bias vs configuration

Reads from results_v3/ JSONs. Run after capacity scaling job completes.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results_v3')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---- Data loading helpers ----

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_fisher():
    return load_json(os.path.join(RESULTS_DIR, 'fisher_fixed_lmax_verification.json'))

def load_cnn_result(nside, fsky, noise, hc):
    """Load CNN result JSON from Expanse results_v3."""
    fname = f"test4_nside{nside}_hc{hc}_fsky{fsky}_noise{noise}.json"
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        return load_json(path)
    return None

def load_hc32_result(nside, fsky, noise):
    """Load original hc=32 CNN result."""
    fname = f"test4_nside{nside}_hc32_fsky{fsky}_noise{noise}.json"
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        return load_json(path)
    return None

CONFIGS = [
    (1.0, 0, "fsky=1.0\nnoise=0"),
    (1.0, 6, "fsky=1.0\nnoise=6"),
    (0.1, 0, "fsky=0.1\nnoise=0"),
    (0.1, 6, "fsky=0.1\nnoise=6"),
]

NSIDE = 32

# ---- Figure 1: Fisher vs CNN (hc=32 and hc=64) bar chart ----

def fig1_fisher_vs_cnn():
    fisher = load_fisher()
    labels = [c[2] for c in CONFIGS]
    x = np.arange(len(labels))
    width = 0.22

    fisher_r = []
    cnn32_r = []
    cnn64_r = []

    for fsky, noise, _ in CONFIGS:
        key = f"nside{NSIDE}_fsky{fsky}_noise{noise}"
        fisher_r.append(fisher[key]["r_pct_error"])

        hc32 = load_hc32_result(NSIDE, fsky, noise)
        cnn32_r.append(hc32["r_pct_error"] if hc32 else np.nan)

        hc64 = load_cnn_result(NSIDE, fsky, noise, 64)
        cnn64_r.append(hc64["r_pct_error"] if hc64 else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, fisher_r, width, label='Fisher (Cramér-Rao)', color='#2196F3', alpha=0.85)
    ax.bar(x, cnn32_r, width, label=f'CNN hc=32 (26.5M)', color='#FF9800', alpha=0.85)
    has64 = not all(np.isnan(cnn64_r))
    if has64:
        ax.bar(x + width, cnn64_r, width, label=f'CNN hc=64 (103.5M)', color='#4CAF50', alpha=0.85)

    ax.set_ylabel(r'$\sigma_r / r$  [%]', fontsize=12)
    ax.set_title(f'NSIDE={NSIDE}: Fisher Lower Bound vs CNN Estimation Error', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, max(max(fisher_r), max(np.nan_to_num(cnn32_r))) * 1.25)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_fisher_vs_cnn_n32.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ---- Figure 2: Capacity scaling (r error vs hidden_channels) ----

def fig2_capacity_scaling():
    fisher = load_fisher()
    hcs = [32, 64]
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (fsky, noise, label) in enumerate(CONFIGS):
        errs = []
        for hc in hcs:
            if hc == 32:
                res = load_hc32_result(NSIDE, fsky, noise)
            else:
                res = load_cnn_result(NSIDE, fsky, noise, hc)
            errs.append(res["r_pct_error"] if res else np.nan)

        key = f"nside{NSIDE}_fsky{fsky}_noise{noise}"
        fisher_val = fisher[key]["r_pct_error"]

        color = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800'][i]
        ax.plot(hcs, errs, 'o-', color=color, label=label, linewidth=2, markersize=8)
        ax.axhline(y=fisher_val, color=color, linestyle='--', alpha=0.4)

    ax.set_xlabel('Hidden Channels', fontsize=12)
    ax.set_ylabel(r'$\sigma_r / r$  [%]', fontsize=12)
    ax.set_title(f'NSIDE={NSIDE}: Capacity Scaling Study', fontsize=13)
    ax.set_xticks(hcs)
    ax.set_xticklabels([f'hc={h}\n({h*416000//1000000}M params)' for h in hcs], fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_capacity_scaling_n32.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ---- Figure 3: r_bias across configs ----

def fig3_r_bias():
    fisher = load_fisher()
    labels = [c[2] for c in CONFIGS]
    x = np.arange(len(labels))

    bias32 = []
    bias64 = []

    for fsky, noise, _ in CONFIGS:
        hc32 = load_hc32_result(NSIDE, fsky, noise)
        bias32.append(hc32.get("r_bias", np.nan) if hc32 else np.nan)

        hc64 = load_cnn_result(NSIDE, fsky, noise, 64)
        bias64.append(hc64.get("r_bias", np.nan) if hc64 else np.nan)

    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, bias32, width, label='hc=32', color='#FF9800', alpha=0.85)
    has64 = not all(np.isnan(bias64))
    if has64:
        ax.bar(x + width/2, bias64, width, label='hc=64', color='#4CAF50', alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel(r'$r_{\rm bias}$', fontsize=12)
    ax.set_title(f'NSIDE={NSIDE}: Systematic $r$ Bias', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_r_bias_n32.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ---- Main ----

if __name__ == '__main__':
    fig1_fisher_vs_cnn()
    fig2_capacity_scaling()
    fig3_r_bias()
    print("All figures generated.")
