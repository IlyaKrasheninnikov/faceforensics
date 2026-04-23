"""FIG-N2: Training variant trajectory.

Plots val AUC for the main-line training variants across the investigation,
showing the adoption of v13 frozen-probe, catastrophic forgetting in v14,
and convergence to the linear-probe strategy.

Data source: outputs_and_cfgs/all_runs_and_outputs_together/all_runs_sorted.csv
Curated to the main-line variants only (not all ~45 attempts).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "outputs_and_cfgs" / "all_runs_and_outputs_together" / "all_runs_sorted.csv"
OUT  = ROOT / "figures" / "fig_n2_variant_trajectory.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Curated main-line variants. Each entry: (csv_run_name_substring, display_label, category)
# Categories: baseline (pre-pivot), transformer (end-to-end physio), physio (continued fine-tune),
#             frozen (v13 frozen backbone + linear probe), mixed (dataset pooling)
MAIN_LINE = [
    ("baseline_v1 (attempt 2)",         "baseline\n(E2E fine-tune)",   "baseline"),
    ("physio_transformer_v3",           "transformer-v3",              "transformer"),
    ("physio_transformer_v7",           "transformer-v7",              "transformer"),
    ("physio_transformer_v8",           "transformer-v8",              "transformer"),
    ("baseline_png_v2",                 "baseline-png-v2",             "baseline"),
    ("physio_rppg_v10 (attempt 1)",     "physio-rppg-v10",             "physio"),
    ("physio_rppg_v11",                 "physio-rppg-v11",             "physio"),
    ("physio_rppg_v12",                 "physio-rppg-v12",             "physio"),
    ("physio_rppg_v13",                 "physio-rppg-v13\n(adopted)",  "adopted"),
    ("physio_rppg_v14",                 "physio-rppg-v14\n(catastrophic)", "catastrophic"),
    ("physio_v15_regularized (full)",   "v15-regularized",             "catastrophic"),
    ("physio_v16_balanced",             "v16-balanced",                "catastrophic"),
    ("physio_v17_resume_v13",           "v17-resume-v13",              "physio"),
    ("physio_v18_gentle_finetune",      "v18-gentle-ft",               "physio"),
    ("physio_v19_v14_plus_aug",         "v19-v14+aug",                 "physio"),
    ("linear_probe_v13 (attempt 4",     "linear-probe\n(final)",       "adopted"),
]

CATEGORY_STYLE = {
    "baseline":     {"color": "#888888", "marker": "o", "label": "Baseline (E2E fine-tune)"},
    "transformer":  {"color": "#4c72b0", "marker": "s", "label": "End-to-end transformer variant"},
    "physio":       {"color": "#55a868", "marker": "D", "label": "Physio multi-task variant"},
    "catastrophic": {"color": "#c44e52", "marker": "X", "label": "Catastrophic forgetting"},
    "adopted":      {"color": "#d48f00", "marker": "*", "label": "Adopted configuration"},
}


def main():
    df = pd.read_csv(CSV)
    df["best_val_auc"] = pd.to_numeric(df["best_val_auc"], errors="coerce")
    labels, aucs, cats = [], [], []
    for substr, label, cat in MAIN_LINE:
        hit = df[df["run_name"].str.contains(substr, regex=False, na=False)]
        hit = hit[hit["best_val_auc"].notna()]
        if hit.empty:
            print(f"[fig_n2] WARN: '{substr}' not found — skipping")
            continue
        labels.append(label)
        aucs.append(float(hit["best_val_auc"].iloc[0]))
        cats.append(cat)

    fig, ax = plt.subplots(figsize=(14.5, 6.6), dpi=200)
    x = list(range(len(labels)))

    # Connecting grey line
    ax.plot(x, aucs, color="#bbbbbb", linewidth=1.0, zorder=1, alpha=0.6)

    # Per-category markers
    plotted_cats = set()
    for xi, auc, cat in zip(x, aucs, cats):
        style = CATEGORY_STYLE[cat]
        ax.scatter(xi, auc,
                   color=style["color"], marker=style["marker"],
                   s=150 if style["marker"] == "*" else 90,
                   edgecolor="black", linewidth=0.6, zorder=3,
                   label=style["label"] if cat not in plotted_cats else None)
        plotted_cats.add(cat)
        ax.text(xi, auc + 0.015, f"{auc:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.axhline(0.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(len(x) - 0.5, 0.51, "chance", fontsize=8, color="#999999", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("FaceForensics++ validation AUC", fontsize=10)
    ax.set_ylim(0.40, 0.85)
    ax.set_title("Training-strategy evaluation: 16 main-line variants\n"
                 "(selection from 45+ runs; full register in Appendix D)",
                 fontsize=11, pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[fig_n2] wrote {OUT}  ({len(labels)} variants)")


if __name__ == "__main__":
    main()
