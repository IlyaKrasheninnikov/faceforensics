"""FIG-N3: Per-manipulation AUC with 95% confidence intervals.

Plots per-manipulation AUC for 5 key ablation variants with Hanley-McNeil
asymptotic 95% CIs, computed closed-form from (AUC, n_pos, n_neg).

Variants shown: backbone_only, full fusion (backbone+rPPG+blink),
rppg_only, blink_only, fakecatcher_svm.

Data source: outputs_and_cfgs/ablation/ablation_results.json
"""
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "outputs_and_cfgs" / "ablation" / "ablation_results.json"
OUT  = ROOT / "figures" / "fig_n3_per_manip_ci.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("1_backbone_only",      "Backbone only"),
    ("4_backbone+rppg+blink","Backbone + rPPG + blink"),
    ("5_rppg_only",          "rPPG only"),
    ("6_blink_only",         "Blink only"),
    ("7_fakecatcher_svm",    "FakeCatcher (SVM)"),
]

MANIPS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# Okabe-Ito colorblind-safe palette
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int, z: float = 1.96):
    """Closed-form asymptotic CI for AUC (Hanley & McNeil, 1982).

    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC^2 / (1 + AUC)
    SE^2 = [AUC(1 - AUC) + (n_pos - 1)(Q1 - AUC^2) + (n_neg - 1)(Q2 - AUC^2)] / (n_pos * n_neg)
    """
    a = max(min(auc, 1.0 - 1e-9), 1e-9)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a)
           + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, a - z * se), min(1.0, a + z * se), se


def main():
    with open(JSON) as f:
        data = json.load(f)

    # For FF++ c23: 100 real ("original") + 100 per each of 5 fake manips
    # Each manip AUC is computed on n_pos=100 (fake) vs n_neg=100 (real subset)
    n_pos = n_neg = 100

    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=200)
    n_variants = len(VARIANTS)
    n_manips   = len(MANIPS)
    bar_w      = 0.15
    group_gap  = 1.0

    x_groups = np.arange(n_manips) * group_gap
    offsets  = (np.arange(n_variants) - (n_variants - 1) / 2) * bar_w

    for vi, (key, label) in enumerate(VARIANTS):
        aucs, los, his = [], [], []
        for manip in MANIPS:
            auc = data[key]["test_per_manip"][manip]["auc"]
            lo, hi, _ = hanley_mcneil_ci(auc, n_pos, n_neg)
            aucs.append(auc)
            los.append(auc - lo)
            his.append(hi - auc)
        ax.bar(x_groups + offsets[vi], aucs, width=bar_w,
               yerr=[los, his], capsize=3,
               color=COLORS[vi], edgecolor="black", linewidth=0.5,
               label=label,
               error_kw=dict(lw=0.9, capthick=0.9, ecolor="#333333"))

    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(x_groups[-1] + 0.45, 0.505, "chance",
            fontsize=8, color="#888888", ha="right")

    ax.set_xticks(x_groups)
    ax.set_xticklabels(MANIPS, fontsize=10)
    ax.set_ylabel("Test AUC (FF++ c23, n=100 per manipulation)", fontsize=10)
    ax.set_ylim(0.40, 0.90)
    ax.set_title("Per-manipulation AUC with Hanley-McNeil 95% CIs\n"
                 "Five key ablation variants on FaceForensics++ c23 test set",
                 fontsize=11, pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[fig_n3] wrote {OUT}")


if __name__ == "__main__":
    main()
