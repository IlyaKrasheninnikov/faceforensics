"""
W9: Publication-quality ablation bar charts from ablation_results.json.

Produces:
  - fig2_ablation_auc.png/pdf — AUC bar chart per variant
  - fig3_per_manip_heatmap.png/pdf — Per-manipulation AUC heatmap

Usage:
    python w9_viz/plot_ablation_bars.py \
        --json_path /kaggle/working/ablation/ablation_results.json \
        --out_dir ./figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

VARIANT_ORDER = [
    "1_backbone_only", "2_backbone+rppg", "3_backbone+blink",
    "4_backbone+rppg+blink", "5_rppg_only", "6_blink_only", "7_fakecatcher_svm",
]
VARIANT_SHORT = [
    "Backbone", "+rPPG", "+Blink", "Full\nfusion",
    "rPPG\nonly", "Blink\nonly", "FakeCatcher\nSVM",
]
MANIP_ORDER = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]


def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_auc_bars(results, out_dir):
    """Figure 2: AUC + EER bar chart for all variants."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    aucs, eers, labels = [], [], []
    for i, vname in enumerate(VARIANT_ORDER):
        if vname not in results:
            continue
        r = results[vname]
        aucs.append(r["test_auc"])
        eers.append(r["test_eer"])
        labels.append(VARIANT_SHORT[i])

    x = np.arange(len(aucs))
    bars1 = ax1.bar(x, aucs, color=COLORS[:len(aucs)], edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Test AUC")
    ax1.set_title("Test AUC by Ablation Variant", fontweight="bold")
    ax1.set_ylim(0.4, 0.85)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax1.legend(fontsize=9)
    for bar, val in zip(bars1, aucs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    bars2 = ax2.bar(x, eers, color=COLORS[:len(eers)], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Test EER")
    ax2.set_title("Test EER by Ablation Variant (lower = better)", fontweight="bold")
    ax2.set_ylim(0.2, 0.55)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, eers):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("P3: Ablation Study — Physiological Signal Contributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_ablation_auc.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'fig2_ablation_auc.png'}")


def plot_per_manip_heatmap(results, out_dir):
    """Figure 3: Per-manipulation AUC heatmap."""
    matrix = []
    row_labels = []
    for i, vname in enumerate(VARIANT_ORDER):
        if vname not in results:
            continue
        r = results[vname]
        pm = r.get("test_per_manip", {})
        row = []
        for m in MANIP_ORDER:
            if m in pm and "auc" in pm[m]:
                row.append(pm[m]["auc"])
            else:
                row.append(0.5)
        matrix.append(row)
        row_labels.append(VARIANT_SHORT[i])

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.45, vmax=0.82)

    ax.set_xticks(np.arange(len(MANIP_ORDER)))
    ax.set_xticklabels(MANIP_ORDER, fontsize=10, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(MANIP_ORDER)):
            val = matrix[i, j]
            color = "white" if val < 0.55 or val > 0.76 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)
    ax.set_title("Per-Manipulation AUC — Ablation Heatmap (FF++ c23 Test)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_per_manip_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'fig3_per_manip_heatmap.png'}")


def main(args):
    results = load_results(args.json_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_auc_bars(results, out_dir)
    plot_per_manip_heatmap(results, out_dir)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W9: Ablation bar charts + heatmap")
    p.add_argument("--json_path", required=True, help="Path to ablation_results.json")
    p.add_argument("--out_dir", default="./figures")
    main(p.parse_args())
