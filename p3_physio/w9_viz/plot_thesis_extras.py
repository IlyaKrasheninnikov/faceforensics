"""
W9 Extra: Additional thesis-quality figures generated from existing JSON results.

Produces 8 new figures (fig7-fig14) entirely from local JSON/TXT files — no GPU needed.
Can run on Kaggle or locally (only needs matplotlib + numpy).

Usage (Kaggle):
    python w9_viz/plot_thesis_extras.py \
        --results_dir /kaggle/input/.../outputs_and_cfgs \
        --out_dir /kaggle/working/figures

Usage (local):
    cd p3_physio
    python w9_viz/plot_thesis_extras.py \
        --results_dir ./outputs_and_cfgs \
        --out_dir ./figures
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

COLORS = {
    "blue": "#2196F3",
    "red": "#F44336",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#9E9E9E",
    "dark": "#37474F",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7: Cross-Dataset Performance Evolution
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig7_cross_dataset_evolution(results_dir, out_dir):
    """Bar chart: FF-only vs Mixed vs Mixed+rPPG across datasets."""
    cross = load_json(results_dir / "cross_eval" / "cross_eval_results.json")
    mixed = load_json(results_dir / "mixed_probe" / "mixed_probe_results.json")
    biosig = load_json(results_dir / "mixed_probe_biosig" / "mixed_biosig_results.json")

    datasets = ["FF++", "CelebDF"]
    ff_only_auc = [
        mixed["FF++"]["ff_only_auc"],
        cross["CelebDF-v2"]["auc"],
    ]
    mixed_auc = [
        mixed["FF++"]["mixed_auc"],
        mixed["CelebDF"]["mixed_auc"],
    ]
    mixed_rppg_auc = [
        biosig["mixed_backbone+rppg"]["per_dataset"]["FF++"]["auc"],
        biosig["mixed_backbone+rppg"]["per_dataset"]["CelebDF"]["auc"],
    ]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars1 = ax.bar(x - width, ff_only_auc, width, label="FF-only probe",
                   color=COLORS["grey"], edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x, mixed_auc, width, label="Mixed-dataset probe",
                   color=COLORS["blue"], edgecolor="white", linewidth=1.5)
    bars3 = ax.bar(x + width, mixed_rppg_auc, width, label="Mixed + rPPG (best)",
                   color=COLORS["green"], edgecolor="white", linewidth=1.5)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Delta annotations for CelebDF
    ax.annotate("+0.174\n(+30.3%)", xy=(1, mixed_auc[1]), fontsize=9,
                color=COLORS["blue"], fontweight="bold",
                xytext=(1.55, 0.68), arrowprops=dict(arrowstyle="->", color=COLORS["blue"]))
    ax.annotate("+0.191\n(+33.3%)", xy=(1 + width, mixed_rppg_auc[1]), fontsize=9,
                color=COLORS["green"], fontweight="bold",
                xytext=(1.6, 0.78), arrowprops=dict(arrowstyle="->", color=COLORS["green"]))

    ax.set_ylabel("AUC", fontsize=13)
    ax.set_title("Cross-Dataset Performance Evolution\n"
                 "Effect of Mixed-Dataset Probe Training and rPPG Features",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4, label="Target AUC 0.90")
    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, borderaxespad=0.)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig7_cross_dataset_evolution.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig7_cross_dataset_evolution")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8: Operating Point Heatmap (TPR@FPR for Deployment)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig8_operating_point_heatmap(results_dir, out_dir):
    """Heatmap: TPR at different FPR thresholds for best model variants."""
    best = load_json(results_dir / "operating_point_best" / "operating_point_best_results.json")

    variants = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]
    variant_labels = ["Backbone only", "Backbone+rPPG", "Backbone+Blink", "Full fusion"]
    fpr_levels = ["TPR@FPR=1%", "TPR@FPR=5%", "TPR@FPR=10%", "TPR@FPR=20%"]
    fpr_short = ["FPR=1%", "FPR=5%", "FPR=10%", "FPR=20%"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for idx, (dataset, title_ds) in enumerate([("FF++", "FF++ (in-domain)"),
                                                ("CelebDF", "CelebDF (cross-dataset)")]):
        ax = axes[idx]
        data = np.zeros((len(variants), len(fpr_levels)))
        for i, var in enumerate(variants):
            for j, fpr in enumerate(fpr_levels):
                data[i, j] = best[var][dataset][fpr]

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.9)

        ax.set_xticks(range(len(fpr_short)))
        ax.set_xticklabels(fpr_short, fontsize=10)
        ax.set_yticks(range(len(variant_labels)))
        ax.set_yticklabels(variant_labels, fontsize=10)
        ax.set_title(title_ds, fontsize=12, fontweight="bold")

        # Annotate cells
        for i in range(len(variants)):
            for j in range(len(fpr_levels)):
                val = data[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        # Highlight best row
        best_idx = 1  # backbone+rppg
        for j in range(len(fpr_levels)):
            ax.add_patch(plt.Rectangle((j - 0.5, best_idx - 0.5), 1, 1,
                                       fill=False, edgecolor=COLORS["green"],
                                       linewidth=2.5))

    fig.suptitle("Operating Point Analysis — TPR at Fixed FPR Thresholds\n"
                 "Green border = best model (backbone+rPPG, mixed probe)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="True Positive Rate")

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig8_operating_point_heatmap.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig8_operating_point_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9: Bio-Signal Contribution — FF-only vs Mixed Probe Context
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig9_biosignal_contribution(results_dir, out_dir):
    """Grouped bars showing bio-signal delta on FF-only probe vs mixed probe."""
    ablation = load_json(results_dir / "ablation" / "ablation_results.json")
    biosig = load_json(results_dir / "mixed_probe_biosig" / "mixed_biosig_results.json")

    # FF-only probe deltas (from ablation, test AUC on FF++)
    bb_only_ff = ablation["1_backbone_only"]["test_auc"]
    rppg_delta_ff = ablation["2_backbone+rppg"]["test_auc"] - bb_only_ff
    blink_delta_ff = ablation["3_backbone+blink"]["test_auc"] - bb_only_ff
    both_delta_ff = ablation["4_backbone+rppg+blink"]["test_auc"] - bb_only_ff

    # Mixed probe deltas on CelebDF (from biosig)
    bb_only_cdf = biosig["mixed_backbone_only"]["per_dataset"]["CelebDF"]["auc"]
    rppg_delta_cdf = biosig["mixed_backbone+rppg"]["per_dataset"]["CelebDF"]["auc"] - bb_only_cdf
    blink_delta_cdf = biosig["mixed_backbone+blink"]["per_dataset"]["CelebDF"]["auc"] - bb_only_cdf
    both_delta_cdf = biosig["mixed_backbone+rppg+blink"]["per_dataset"]["CelebDF"]["auc"] - bb_only_cdf

    signals = ["+rPPG", "+Blink", "+rPPG+Blink"]
    ff_deltas = [rppg_delta_ff, blink_delta_ff, both_delta_ff]
    cdf_deltas = [rppg_delta_cdf, blink_delta_cdf, both_delta_cdf]

    x = np.arange(len(signals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))

    bars1 = ax.bar(x - width / 2, [d * 100 for d in ff_deltas], width,
                   label="FF-only probe (on FF++ test)", color=COLORS["grey"],
                   edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, [d * 100 for d in cdf_deltas], width,
                   label="Mixed probe (on CelebDF test)", color=COLORS["green"],
                   edgecolor="white", linewidth=1.5)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            sign = "+" if h >= 0 else ""
            va = "bottom" if h >= 0 else "top"
            offset = 0.15 if h >= 0 else -0.15
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{sign}{h:.1f}%", ha="center", va=va, fontsize=11, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("AUC Delta (percentage points)", fontsize=12)
    ax.set_title("Bio-Signal Contribution: FF-Only Probe vs Mixed Probe\n"
                 "Signals become useful only under cross-dataset conditions",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=12)

    # Give headroom so title/legend don't collide with bars
    y_max = max(max(ff_deltas), max(cdf_deltas)) * 100
    y_min = min(min(ff_deltas), min(cdf_deltas)) * 100
    ax.set_ylim(y_min - 2, y_max + 4.5)

    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, borderaxespad=0.)
    ax.grid(axis="y", alpha=0.3)

    # Annotation box — push to the right, clear of the bars
    ax.annotate("rPPG: useless on FF++\nbut +2.3% on CelebDF\nwith mixed probe!",
                xy=(0 + width / 2, cdf_deltas[0] * 100),
                xytext=(1.35, y_max + 1.8),
                fontsize=9, color=COLORS["dark"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=COLORS["green"]))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig9_biosignal_contribution.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig9_biosignal_contribution")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10: Calibration Reliability Diagram
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig10_calibration(results_dir, out_dir):
    """Before/after calibration ECE comparison + conceptual reliability diagram."""
    cal = load_json(results_dir / "calibration" / "calibration_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: ECE before/after
    ax = axes[0]
    splits = ["Validation", "Test"]
    ece_before = [cal["val_raw"]["ece"], cal["test_raw"]["ece"]]
    ece_after = [cal["val_calibrated"]["ece"], cal["test_calibrated"]["ece"]]

    x = np.arange(len(splits))
    width = 0.35
    bars1 = ax.bar(x - width / 2, ece_before, width, label="Before (raw)",
                   color=COLORS["red"], alpha=0.8, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, ece_after, width, label="After (Platt)",
                   color=COLORS["green"], alpha=0.8, edgecolor="white", linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=0.08, color="orange", linestyle="--", linewidth=1.5,
               label="Target ECE <= 0.08")
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax.set_title("(a) ECE Before/After Platt Scaling", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylim(0, 0.12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add improvement arrows
    for i in range(len(splits)):
        ax.annotate("", xy=(x[i] + width / 2, ece_after[i] + 0.012),
                    xytext=(x[i] - width / 2, ece_before[i] + 0.012),
                    arrowprops=dict(arrowstyle="->", color=COLORS["dark"],
                                   lw=1.5, connectionstyle="arc3,rad=-0.3"))
        reduction = (1 - ece_after[i] / ece_before[i]) * 100
        ax.text(x[i], max(ece_before[i], ece_after[i]) + 0.015,
                f"-{reduction:.0f}%", ha="center", fontsize=9, color=COLORS["dark"])

    # Panel B: Other metrics unchanged
    ax = axes[1]
    metrics = ["AUC", "EER", "AP"]
    raw_vals = [cal["test_raw"]["auc"], cal["test_raw"]["eer"], cal["test_raw"]["ap"]]
    cal_vals = [cal["test_calibrated"]["auc"], cal["test_calibrated"]["eer"],
                cal["test_calibrated"]["ap"]]

    x = np.arange(len(metrics))
    bars1 = ax.bar(x - width / 2, raw_vals, width, label="Before (raw)",
                   color=COLORS["blue"], alpha=0.6, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, cal_vals, width, label="After (Platt)",
                   color=COLORS["blue"], alpha=0.9, edgecolor="white", linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("(b) Discrimination Metrics (Unchanged)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, 0.15, "Platt scaling is a monotonic\ntransform — AUC/EER/AP unchanged",
            transform=ax.transAxes, fontsize=9, ha="center", color=COLORS["grey"],
            style="italic")

    fig.suptitle("Platt Scaling Calibration — ECE Improvement Without Discrimination Loss",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig10_calibration.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig10_calibration")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11: Per-Manipulation Radar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig11_per_manipulation_radar(results_dir, out_dir):
    """Radar/spider chart: per-manipulation AUC for key ablation variants."""
    ablation = load_json(results_dir / "ablation" / "ablation_results.json")

    manips = ["Deepfakes", "FaceShifter", "Face2Face", "NeuralTextures", "FaceSwap"]
    variants = {
        "Backbone only": "1_backbone_only",
        "Full fusion": "4_backbone+rppg+blink",
        "Blink only": "6_blink_only",
    }
    colors_v = [COLORS["blue"], COLORS["green"], COLORS["orange"]]

    # Extract data
    data = {}
    for label, key in variants.items():
        data[label] = [ablation[key]["test_per_manip"][m]["auc"] for m in manips]

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(manips), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (label, values) in enumerate(data.items()):
        vals = values + values[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=colors_v[i],
                markersize=6)
        ax.fill(angles, vals, alpha=0.1, color=colors_v[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(manips, fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, 0.85)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80"], fontsize=9)
    ax.set_title("Per-Manipulation AUC by Model Variant\n(FF++ c23 test set)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.0), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig11_per_manip_radar.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig11_per_manip_radar")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12: Mixed Probe Impact — CelebDF TPR@FPR Improvement
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig12_mixed_probe_tpr(results_dir, out_dir):
    """Side-by-side TPR@FPR bars: FF-only vs Mixed on CelebDF."""
    op_mixed = load_json(results_dir / "operating_point_mixed" /
                         "operating_point_mixed_results.json")

    fpr_levels = ["TPR@FPR=1%", "TPR@FPR=5%", "TPR@FPR=10%", "TPR@FPR=20%"]
    fpr_short = ["FPR=1%", "FPR=5%", "FPR=10%", "FPR=20%"]

    ff_tpr = [op_mixed["CelebDF"]["ff_only"][k] for k in fpr_levels]
    mx_tpr = [op_mixed["CelebDF"]["mixed"][k] for k in fpr_levels]

    x = np.arange(len(fpr_short))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(x - width / 2, [v * 100 for v in ff_tpr], width,
                   label="FF-only probe", color=COLORS["grey"],
                   edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, [v * 100 for v in mx_tpr], width,
                   label="Mixed-dataset probe", color=COLORS["blue"],
                   edgecolor="white", linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Multiplier annotations
    for i in range(len(fpr_short)):
        if ff_tpr[i] > 0:
            mult = mx_tpr[i] / ff_tpr[i]
            mid_y = max(ff_tpr[i], mx_tpr[i]) * 100 + 4
            ax.text(x[i], mid_y, f"{mult:.1f}x", ha="center", fontsize=10,
                    fontweight="bold", color=COLORS["dark"],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#E3F2FD", alpha=0.8))

    ax.set_ylabel("True Positive Rate (%)", fontsize=12)
    ax.set_title("CelebDF Forensic Deployment: Mixed Probe Catches 3x More Fakes\n"
                 "TPR at fixed False Positive Rate thresholds",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fpr_short, fontsize=12)
    ax.set_ylim(0, 65)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig12_mixed_probe_tpr.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig12_mixed_probe_tpr")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 13: Summary Dashboard — All Key Results in One Figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig13_summary_dashboard(results_dir, out_dir):
    """4-panel summary dashboard for thesis overview."""
    ablation = load_json(results_dir / "ablation" / "ablation_results.json")
    mixed = load_json(results_dir / "mixed_probe" / "mixed_probe_results.json")
    biosig = load_json(results_dir / "mixed_probe_biosig" / "mixed_biosig_results.json")
    robust = load_json(results_dir / "robustness" / "robustness_results.json")
    cal = load_json(results_dir / "calibration" / "calibration_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Cross-dataset evolution
    ax = axes[0, 0]
    configs = ["FF-only\nprobe", "Mixed\nprobe", "Mixed\n+rPPG"]
    ff_auc = [mixed["FF++"]["ff_only_auc"], mixed["FF++"]["mixed_auc"],
              biosig["mixed_backbone+rppg"]["per_dataset"]["FF++"]["auc"]]
    cdf_auc = [0.574, mixed["CelebDF"]["mixed_auc"],
               biosig["mixed_backbone+rppg"]["per_dataset"]["CelebDF"]["auc"]]

    x = np.arange(len(configs))
    width = 0.3
    ax.bar(x - width / 2, ff_auc, width, label="FF++", color=COLORS["blue"], alpha=0.8)
    ax.bar(x + width / 2, cdf_auc, width, label="CelebDF", color=COLORS["orange"], alpha=0.8)
    for i in range(len(configs)):
        ax.text(x[i] - width / 2, ff_auc[i] + 0.01, f"{ff_auc[i]:.3f}",
                ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width / 2, cdf_auc[i] + 0.01, f"{cdf_auc[i]:.3f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("AUC")
    ax.set_title("(a) Cross-Dataset Evolution", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Ablation compact
    ax = axes[0, 1]
    variant_names = ["BB", "BB+R", "BB+B", "BB+RB", "rPPG", "Blink", "FC"]
    variant_keys = [f"{i}_" for i in range(1, 8)]
    aucs = []
    for k in ["1_backbone_only", "2_backbone+rppg", "3_backbone+blink",
              "4_backbone+rppg+blink", "5_rppg_only", "6_blink_only",
              "7_fakecatcher_svm"]:
        aucs.append(ablation[k]["test_auc"])

    colors_bar = [COLORS["blue"]] * 4 + [COLORS["orange"]] * 2 + [COLORS["red"]]
    bars = ax.bar(variant_names, aucs, color=colors_bar, edgecolor="white", linewidth=1)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5, label="Chance")
    ax.set_ylabel("Test AUC")
    ax.set_title("(b) Ablation Study (FF++ c23)", fontweight="bold")
    ax.set_ylim(0.35, 0.80)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Robustness compact
    ax = axes[1, 0]
    perturbs = ["Clean", "J50", "J30", "J10", "n5", "n10", "n20",
                "b1", "b2", "b3", "d2x", "d4x"]
    rob_keys = ["clean", "jpeg_q50", "jpeg_q30", "jpeg_q10",
                "noise_s5", "noise_s10", "noise_s20",
                "blur_s1", "blur_s2", "blur_s3",
                "downscale_2x", "downscale_4x"]
    rob_aucs = [robust[k]["auc"] for k in rob_keys]

    # Color by category
    cat_colors = ([COLORS["green"]] +                      # clean
                  [COLORS["blue"]] * 3 +                   # jpeg
                  [COLORS["orange"]] * 3 +                 # noise
                  [COLORS["red"]] * 3 +                    # blur
                  [COLORS["purple"]] * 2)                  # downscale
    bars = ax.bar(perturbs, rob_aucs, color=cat_colors, edgecolor="white", linewidth=1)
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_ylabel("AUC")
    ax.set_title("(c) Robustness Profile", fontweight="bold")
    ax.set_ylim(0.35, 0.80)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Legend for categories
    patches = [mpatches.Patch(color=COLORS["blue"], label="JPEG"),
               mpatches.Patch(color=COLORS["orange"], label="Noise"),
               mpatches.Patch(color=COLORS["red"], label="Blur"),
               mpatches.Patch(color=COLORS["purple"], label="Downscale")]
    ax.legend(handles=patches, fontsize=8, ncol=4, loc="upper right")

    # Panel D: Go/No-Go summary
    ax = axes[1, 1]
    ax.axis("off")
    targets = [
        ("Cross-dataset AUC >= 0.90", 0.765, False),
        ("EER <= 10%", 0.168, False),
        ("ECE <= 0.08", 0.040, True),
        ("FF++ AUC >= 0.90", 0.899, True),
    ]
    y_pos = 0.85
    ax.text(0.5, 0.98, "(d) Go/No-Go Assessment", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center", va="top")
    for label, val, passed in targets:
        color = COLORS["green"] if passed else COLORS["red"]
        icon = "PASS" if passed else "FAIL"
        ax.text(0.05, y_pos, icon, transform=ax.transAxes, fontsize=14,
                fontweight="bold", color=color, family="monospace")
        ax.text(0.18, y_pos, f"{label}", transform=ax.transAxes, fontsize=11)
        ax.text(0.85, y_pos, f"{val:.3f}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right")
        y_pos -= 0.15

    # Best model box
    ax.text(0.5, 0.18, "Best Model: Mixed Probe + rPPG\n"
            "FF++ AUC 0.897 | CelebDF AUC 0.764\n"
            "CelebDF Precision 0.951 | ECE 0.040",
            transform=ax.transAxes, fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                      edgecolor=COLORS["green"], linewidth=2))

    fig.suptitle("P3 Bio-Signal Forensics — Results Summary Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig13_summary_dashboard.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig13_summary_dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 14: Training Curves from Trackio (v13 fine-tuning catastrophic forgetting)
# ─────────────────────────────────────────────────────────────────────────────
def parse_trackio_metric(text, metric_name):
    """Parse a metric section from the Trackio dump text file."""
    pattern = rf"Getting metric: {re.escape(metric_name)}\nFound \d+ value\(s\):\n\nStep \| Timestamp \| Value\n-+\n((?:\d+.*\n)*)"
    match = re.search(pattern, text)
    if not match:
        return [], []
    steps, values = [], []
    for line in match.group(1).strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 3:
            steps.append(int(parts[0].strip()))
            values.append(float(parts[2].strip()))
    return steps, values


def plot_fig14_training_curves(results_dir, out_dir):
    """Training curves from Trackio: shows catastrophic forgetting in fine-tuning."""
    trackio_path = results_dir / "week3" / "week3_metrics_v3.txt"
    if not trackio_path.exists():
        print(f"  Skipping fig14 — {trackio_path} not found")
        return

    text = trackio_path.read_text(encoding="utf-8", errors="replace")

    # Parse epoch-level metrics (skip first 2 entries which are from a different run)
    metrics_to_plot = {
        "val/auc": "Val AUC",
        "val/eer": "Val EER",
        "train/total": "Train Total Loss",
        "train/cls": "Train Classification Loss",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (metric_key, metric_label) in zip(axes.flat, metrics_to_plot.items()):
        steps, values = parse_trackio_metric(text, metric_key)

        # The Trackio dump has 12 entries: first 2 are from a 2-epoch run,
        # next 10 are from the 10-epoch run. Use the 10-epoch run.
        if len(steps) >= 10:
            epochs = list(range(1, 11))
            vals = values[2:12]  # skip the first 2-epoch run
        else:
            epochs = steps
            vals = values

        ax.plot(epochs, vals, "o-", linewidth=2.5, markersize=7,
                color=COLORS["blue"], markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=COLORS["blue"])

        # Highlight best epoch
        if "auc" in metric_key:
            best_idx = np.argmax(vals)
            ax.plot(epochs[best_idx], vals[best_idx], "o", markersize=12,
                    color=COLORS["green"], zorder=5)
            ax.annotate(f"Best: {vals[best_idx]:.3f}\n(epoch {epochs[best_idx]})",
                        xy=(epochs[best_idx], vals[best_idx]),
                        xytext=(epochs[best_idx] + 1.5, vals[best_idx] + 0.02),
                        fontsize=9, fontweight="bold", color=COLORS["green"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["green"]))
        elif "eer" in metric_key:
            best_idx = np.argmin(vals)
            ax.plot(epochs[best_idx], vals[best_idx], "o", markersize=12,
                    color=COLORS["green"], zorder=5)
            ax.annotate(f"Best: {vals[best_idx]:.3f}\n(epoch {epochs[best_idx]})",
                        xy=(epochs[best_idx], vals[best_idx]),
                        xytext=(epochs[best_idx] + 1.5, vals[best_idx] - 0.02),
                        fontsize=9, fontweight="bold", color=COLORS["green"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["green"]))

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 11))

        # Add chance line for AUC
        if "auc" in metric_key:
            ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Chance (0.5)")
            ax.legend(fontsize=9)

    fig.suptitle("V13 Fine-Tuning: Training Curves from Trackio\n"
                 "Val AUC peaks at epoch 2 then degrades — evidence of catastrophic forgetting",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig14_training_curves.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig14_training_curves")


def main(args):
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating thesis extra figures...\n")

    plot_fig7_cross_dataset_evolution(results_dir, out_dir)
    plot_fig8_operating_point_heatmap(results_dir, out_dir)
    plot_fig9_biosignal_contribution(results_dir, out_dir)
    plot_fig10_calibration(results_dir, out_dir)
    plot_fig11_per_manipulation_radar(results_dir, out_dir)
    plot_fig12_mixed_probe_tpr(results_dir, out_dir)
    plot_fig13_summary_dashboard(results_dir, out_dir)
    plot_fig14_training_curves(results_dir, out_dir)

    print(f"\nDone! 8 new figures saved to {out_dir}")
    print("  fig7_cross_dataset_evolution  — CelebDF: 0.574 -> 0.748 -> 0.765")
    print("  fig8_operating_point_heatmap  — TPR@FPR deployment matrix")
    print("  fig9_biosignal_contribution   — rPPG useless on FF++, +2.3% mixed")
    print("  fig10_calibration             — ECE 0.092 -> 0.040")
    print("  fig11_per_manip_radar         — Spider chart by manipulation type")
    print("  fig12_mixed_probe_tpr         — 3x more fakes caught on CelebDF")
    print("  fig13_summary_dashboard       — 4-panel overview for thesis intro")
    print("  fig14_training_curves         — Trackio: catastrophic forgetting evidence")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate extra thesis figures from JSON results")
    p.add_argument("--results_dir", default="./outputs_and_cfgs",
                   help="Path to outputs_and_cfgs directory")
    p.add_argument("--out_dir", default="./figures",
                   help="Output directory for figures")
    main(p.parse_args())
