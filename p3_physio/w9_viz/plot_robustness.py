"""
W9: Publication-quality robustness figure from W6 results.

Produces fig6_robustness.png — grouped bar chart showing AUC drop
under different perturbations.

Usage:
    python w9_viz/plot_robustness.py \
        --json_path /kaggle/working/robustness/robustness_results.json \
        --out_dir /kaggle/working/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 150,
})

GROUPS = {
    "JPEG": ["jpeg_q50", "jpeg_q30", "jpeg_q10"],
    "Blur": ["blur_s1", "blur_s2", "blur_s3"],
    "Noise": ["noise_s5", "noise_s10", "noise_s20"],
    "Downscale": ["downscale_2x", "downscale_4x"],
}
GROUP_COLORS = {"JPEG": "#2196F3", "Blur": "#F44336", "Noise": "#FF9800", "Downscale": "#9C27B0"}
LABELS = {
    "jpeg_q50": "Q50", "jpeg_q30": "Q30", "jpeg_q10": "Q10",
    "blur_s1": "σ=1", "blur_s2": "σ=2", "blur_s3": "σ=3",
    "noise_s5": "σ=5", "noise_s10": "σ=10", "noise_s20": "σ=20",
    "downscale_2x": "2×", "downscale_4x": "4×",
}


def main(args):
    with open(args.json_path) as f:
        results = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_auc = results["clean"]["auc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: AUC values
    x_labels, auc_vals, colors = [], [], []
    for group_name, keys in GROUPS.items():
        for k in keys:
            if k in results:
                x_labels.append(f"{group_name}\n{LABELS[k]}")
                auc_vals.append(results[k]["auc"])
                colors.append(GROUP_COLORS[group_name])

    x = np.arange(len(auc_vals))
    bars = ax1.bar(x, auc_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=clean_auc, color="green", linestyle="--", linewidth=2, label=f"Clean ({clean_auc:.3f})")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_ylabel("Test AUC")
    ax1.set_title("AUC Under Perturbations", fontweight="bold")
    ax1.set_ylim(0.4, 0.8)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, auc_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Panel 2: AUC drop
    drops = [v - clean_auc for v in auc_vals]
    bars2 = ax2.bar(x, drops, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=-0.05, color="orange", linestyle="--", alpha=0.7, label="Target: ≤5% drop")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_ylabel("AUC Change")
    ax2.set_title("AUC Degradation vs Clean", fontweight="bold")
    ax2.set_ylim(-0.25, 0.02)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, drops):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.008,
                 f"{val:+.3f}", ha="center", va="top", fontsize=7, fontweight="bold")

    fig.suptitle("P3: Robustness Analysis — Model Performance Under Perturbations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig6_robustness.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'fig6_robustness.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", required=True)
    p.add_argument("--out_dir", default="./figures")
    main(p.parse_args())
