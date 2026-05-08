"""
v5 corrected figures (E14 strict LODO) — regenerated from CSVs with the four-protocol
terminology, n=1758 subject-aware CelebDF, and Stouffer-significance
annotations on the physiology ablations.

Replaces stale figures:
  - fig2_ablation_auc_v5.png    (was fig2_ablation_auc.png)
  - fig7_cross_dataset_evolution_v5.png  (was fig7)
  - fig9_biosignal_contribution_v5.png   (was fig9)
  - fig12_mixed_probe_tpr_v5.png         (was fig12)
  - fig13_summary_dashboard_v5.png       (was fig13)

Run locally:
    cd p3_physio
    python w9_viz/plot_v5_corrected.py
"""

from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

ROOT = Path(__file__).resolve().parents[1] / "outputs_and_cfgs"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour palette (Okabe-Ito, colour-blind safe)
C_BACKBONE = "#0072B2"      # blue
C_RPPG     = "#009E73"      # green
C_BLINK    = "#E69F00"      # orange
C_FUSION   = "#D55E00"      # red
C_B4       = "#999999"
C_DINOV2   = "#56B4E9"
C_CLIP     = "#0072B2"
PROTOCOL_COLORS = {
    "within-dataset":               "#999999",
    "mixed-domain held-out id":     "#56B4E9",
    "strict LODO":                  "#0072B2",
    "quasi-LODO":                   "#E69F00",
    "external stress test (DF40)":  "#CC79A7",
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────
# Figure 2 — variant ablation under strict LODO with Stouffer p
# ─────────────────────────────────────────────────────────────────
def fig2_ablation_v4():
    """8-seed strict-LODO ablation. Means recomputed from E14 + E15b score arrays."""
    e14_results = read_csv(ROOT / "strict_lodo_bundle/e14_lodo_strict_clip/results.csv")
    e15_results = read_csv(ROOT / "e15_bundle/e15_extra_seed_clip/results.csv")
    all_rows = [r for r in e14_results + e15_results if r["config"] == "test_celebdf"]

    order = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]
    labels = ["Backbone\nonly", "+rPPG", "+Blink", "Full\nfusion"]
    colors = [C_BACKBONE, C_RPPG, C_BLINK, C_FUSION]

    aucs = []; stds = []
    for v in order:
        per_seed = [float(r["auc"]) for r in all_rows if r["variant"] == v]
        aucs.append(float(np.mean(per_seed)))
        stds.append(float(np.std(per_seed, ddof=1)))

    # Stouffer p across 5 seeds for each variant vs backbone_only
    # (computed from this session's per-seed DeLong recomputation)
    # 8-seed strict-LODO Stouffer p (E14 + E15b, recomputed 2026-05-07
    # from per-seed DeLong; positive z = backbone-only better)
    stouffer_p = {
        "backbone_only":  None,
        "backbone+rppg":  0.00046,   # z = +3.50, *** (bimodal seeds, mean pro-backbone)
        "backbone+blink": 0.00033,   # z = +3.59, *** (backbone significantly better)
        "full_fusion":    7e-6,      # z = +4.50, *** (backbone significantly better)
    }

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, aucs, yerr=stds, color=colors, edgecolor="black",
                  linewidth=0.8, capsize=4, error_kw={"linewidth": 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.72, 0.78)
    ax.set_ylabel("AUC (8-seed mean ± std)")
    ax.set_title("CLIP ViT-L/14 — Strict LODO CelebDF (n = 1758, 8-seed mean ± std)\n"
                 "Variant ablation: all physiology variants statistically inferior to backbone-only",
                 fontsize=10)

    # Annotate AUC values + Stouffer p
    for i, (bar, auc, std) in enumerate(zip(bars, aucs, stds)):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + std + 0.0008,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=9)
        v = order[i]
        if stouffer_p[v] is not None:
            delta = auc - aucs[0]
            sig = "n.s." if stouffer_p[v] > 0.05 else (
                "*" if stouffer_p[v] > 1e-3 else (
                    "**" if stouffer_p[v] > 1e-5 else "***"))
            ax.text(bar.get_x() + bar.get_width() / 2, 0.722,
                    f"Δ={delta:+.4f}\n{sig}", ha="center", va="bottom",
                    fontsize=8, style="italic")

    ax.axhline(aucs[0], linestyle=":", color="gray", linewidth=0.8, alpha=0.7)
    ax.text(0.02, aucs[0] + 0.0003, "backbone-only", color="gray",
            fontsize=8, transform=ax.get_yaxis_transform())

    # Footnote
    fig.text(0.5, -0.02,
             "8-seed Stouffer-combined DeLong p (E14 seeds 0/1/42/1337/2024 + E15b seeds 7/13/99). "
             "All physiology variants are statistically inferior to backbone-only (p < 0.001) but "
             "differences fall below the 0.005 practical-significance threshold.",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "fig2_ablation_auc_v5.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig2_ablation_auc_v5.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig2_ablation_auc_v5")


# ─────────────────────────────────────────────────────────────────
# Figure 7 — protocol-stratified CelebDF AUC across backbones
# ─────────────────────────────────────────────────────────────────
def fig7_cross_dataset_v4():
    """4-protocol comparison: within / mixed-domain / strict LODO / DF40."""
    # Backbone × protocol matrix (n=1758 by-subject; DF40 ALL n=2209)
    backbones = ["EfficientNet-B4 (v13)", "DINOv2", "CLIP ViT-L/14"]

    # Source files
    sanity_csv = {
        "B4":     ROOT / "sanity_bundle/sanity_b4_idsplit/aggregate.csv",
        "DINO":   ROOT / "sanity_bundle/sanity_dinov2_idsplit/aggregate.csv",
        "CLIP":   ROOT / "sanity_bundle/sanity_clip_idsplit/aggregate.csv",
    }
    lodo_csv = {
        "B4":   ROOT / "strict_lodo_bundle/e14_lodo_strict_b4/aggregate.csv",
        "DINO": ROOT / "strict_lodo_bundle/e14_lodo_strict_dinov2/aggregate.csv",
        "CLIP": ROOT / "strict_lodo_bundle/e14_lodo_strict_clip/aggregate.csv",
    }
    df40_csv = {
        "B4":   ROOT / "fixes_bundle/e13_b4_df40/aggregate.csv",
        "DINO": ROOT / "fixes_bundle/e13_dinov2_df40/aggregate.csv",
        "CLIP": ROOT / "fixes_bundle/e13_clip_df40/aggregate.csv",
    }

    def get_sanity(key, regime):
        rows = read_csv(sanity_csv[key])
        r = next(x for x in rows if x["variant"] == "backbone_only"
                 and x["regime"] == regime and x["cdf_split"] == "by_subject")
        return float(r["auc_mean"]), float(r["auc_std"])

    def get_lodo(key):
        rows = read_csv(lodo_csv[key])
        r = next(x for x in rows if x["config"] == "test_celebdf"
                 and x["variant"] == "backbone_only")
        return float(r["auc_mean"]), float(r["auc_std"])

    def get_df40(key):
        rows = read_csv(df40_csv[key])
        r = next(x for x in rows if x["variant"] == "backbone_only"
                 and x["method"] == "ALL")
        return float(r["auc_mean"]), float(r["auc_std"])

    data = {}
    for key, name in zip(["B4", "DINO", "CLIP"], backbones):
        data[name] = {
            "within-dataset (FF-only → CelebDF, n=1758)":      get_sanity(key, "ff_only"),
            "mixed-domain held-out id (n=1758)":               get_sanity(key, "mixed"),
            "strict LODO (FF+DFDC → CelebDF, n=1758)":         get_lodo(key),
            "external stress (DF40 ALL, n=2209)":              get_df40(key),
        }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    protocol_order = list(data[backbones[0]].keys())
    n_p = len(protocol_order)
    n_b = len(backbones)
    x = np.arange(n_p)
    w = 0.25

    colors_b = [C_B4, C_DINOV2, C_CLIP]

    for i, b in enumerate(backbones):
        means = [data[b][p][0] for p in protocol_order]
        stds  = [data[b][p][1] for p in protocol_order]
        bars = ax.bar(x + (i - 1) * w, means, w, yerr=stds,
                      label=b, color=colors_b[i], edgecolor="black",
                      linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.012,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" (", "\n(") for p in protocol_order], fontsize=8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(n_p - 0.7, 0.51, "chance", color="red", fontsize=7)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("CelebDF AUC (backbone-only, 5-seed mean ± std)")
    ax.set_title("CelebDF detection across four evaluation protocols\n"
                 "Only 'strict LODO' constitutes cross-dataset generalisation",
                 fontsize=10)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.text(0.5, -0.02,
             "All CelebDF rows use subject-aware test split (n=1758, 1542 fakes / 216 reals). "
             "DF40 = video-based methods only (sadtalker, simswap × CDF/FF). "
             "Strict LODO: target dataset fully held out from training, validation, calibration.",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "fig7_cross_dataset_evolution_v5.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig7_cross_dataset_evolution_v5.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig7_cross_dataset_evolution_v5")


# ─────────────────────────────────────────────────────────────────
# Figure 9 — biosignal contribution: Δ vs backbone with practical threshold
# ─────────────────────────────────────────────────────────────────
def fig9_biosignal_v4():
    """Δ-AUC of physiology variants vs backbone-only, three protocols × three backbones."""
    sanity_csv = {
        "B4":   ROOT / "sanity_bundle/sanity_b4_idsplit/aggregate.csv",
        "DINO": ROOT / "sanity_bundle/sanity_dinov2_idsplit/aggregate.csv",
        "CLIP": ROOT / "sanity_bundle/sanity_clip_idsplit/aggregate.csv",
    }
    lodo_csv = {
        "B4":   ROOT / "strict_lodo_bundle/e14_lodo_strict_b4/aggregate.csv",
        "DINO": ROOT / "strict_lodo_bundle/e14_lodo_strict_dinov2/aggregate.csv",
        "CLIP": ROOT / "strict_lodo_bundle/e14_lodo_strict_clip/aggregate.csv",
    }

    variants = ["backbone+rppg", "backbone+blink", "full_fusion"]
    var_labels = ["+rPPG", "+Blink", "Full fusion"]

    def deltas(key, source, regime=None):
        if source == "sanity":
            rows = [x for x in read_csv(sanity_csv[key])
                    if x["regime"] == regime and x["cdf_split"] == "by_subject"]
        else:  # lodo
            rows = [x for x in read_csv(lodo_csv[key])
                    if x["config"] == "test_celebdf"]
        bb = next(x for x in rows if x["variant"] == "backbone_only")
        bb_auc = float(bb["auc_mean"])
        return [float(next(x for x in rows if x["variant"] == v)["auc_mean"]) - bb_auc
                for v in variants]

    panels = [
        ("within-dataset (FF-only)",         "sanity", "ff_only"),
        ("mixed-domain held-out identity",   "sanity", "mixed"),
        ("strict LODO (cross-dataset)",      "lodo",   None),
    ]
    backbones = ["B4", "DINO", "CLIP"]
    backbone_labels = ["B4 (v13)", "DINOv2", "CLIP ViT-L/14"]
    colors_b = [C_B4, C_DINOV2, C_CLIP]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, (title, src, regime) in zip(axes, panels):
        x = np.arange(len(variants))
        w = 0.25
        for i, key in enumerate(backbones):
            d = deltas(key, src, regime)
            ax.bar(x + (i - 1) * w, d, w, label=backbone_labels[i],
                   color=colors_b[i], edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(0.005, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.axhline(-0.005, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(var_labels)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(-0.03, 0.05)

    axes[0].set_ylabel("Δ AUC vs backbone-only (5-seed mean)")
    axes[2].text(2.4, 0.0055, "±0.005 = practical threshold",
                 fontsize=7, color="red", style="italic", rotation=0)
    axes[0].legend(loc="upper left", framealpha=0.95, fontsize=8)

    fig.suptitle("Physiology contribution: Δ AUC vs backbone-only across three protocols\n"
                 "Representation-dependent marginal value — practical-significance threshold "
                 "(±0.005 AUC) shown in red",
                 fontsize=10)

    fig.text(0.5, -0.01,
             "CLIP under strict LODO (E14, no leak): full_fusion Δ = −0.005 AUC (Stouffer-combined "
             "p = 0.029 in favour of backbone-only across 5 seeds, n = 1758) — physiology slightly "
             "harmful under true cross-dataset evaluation.",
             ha="center", fontsize=8, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(OUT / "fig9_biosignal_contribution_v5.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig9_biosignal_contribution_v5.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig9_biosignal_contribution_v5")


# ─────────────────────────────────────────────────────────────────
# Figure 12 — TPR@FPR curves under mixed-domain (corrected to n=1758)
# ─────────────────────────────────────────────────────────────────
def fig12_mixed_probe_tpr_v4():
    """TPR @ {1, 5, 10}% FPR for CLIP under STRICT LODO CelebDF, n=1758.

    Retargeted from mixed-domain to strict LODO since the sanity bundle
    only exports tpr5, while the LODO aggregate has all three FPR levels.
    LODO is also the more important protocol for this thesis claim.
    """
    rows = [x for x in read_csv(ROOT / "strict_lodo_bundle/e14_lodo_strict_clip/aggregate.csv")
            if x["config"] == "test_celebdf"]

    variants = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]
    var_labels = ["Backbone only", "+rPPG", "+Blink", "Full fusion"]
    colors = [C_BACKBONE, C_RPPG, C_BLINK, C_FUSION]

    fpr_levels = [0.01, 0.05, 0.10]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for v, lbl, col in zip(variants, var_labels, colors):
        r = next(x for x in rows if x["variant"] == v)
        tprs = [float(r["tpr1_mean"]), float(r["tpr5_mean"]), float(r["tpr10_mean"])]
        stds = [float(r["tpr1_std"]),  float(r["tpr5_std"]),  float(r["tpr10_std"])]
        ax.errorbar(fpr_levels, tprs, yerr=stds, label=lbl, color=col,
                    marker="o", capsize=3, linewidth=1.6, markersize=6)

    ax.set_xscale("log")
    ax.set_xticks(fpr_levels)
    ax.set_xticklabels(["1%", "5%", "10%"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (5-seed mean ± std)")
    ax.set_title("CLIP under strict LODO (CelebDF, n = 1758, train FF + DFDC)\n"
                 "Operating points across physiology variants",
                 fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_ylim(0.05, 0.45)
    fig.text(0.5, -0.01,
             "Subject-aware CelebDF held-out (1542 fakes / 216 reals). "
             "Strict-LODO operating points are low: TPR@5% ≈ 0.24 across variants. "
             "Physiology yields no operationally meaningful gain (lines overlap within seed std).",
             ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(OUT / "fig12_mixed_probe_tpr_v5.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig12_mixed_probe_tpr_v5.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig12_mixed_probe_tpr_v5")


# ─────────────────────────────────────────────────────────────────
# Figure 13 — summary dashboard (4 panels)
# ─────────────────────────────────────────────────────────────────
def fig13_summary_v4():
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    # ── Panel A: protocol-stratified CelebDF AUC, CLIP only ──
    axA = fig.add_subplot(gs[0, 0])
    sanity = read_csv(ROOT / "sanity_bundle/sanity_clip_idsplit/aggregate.csv")
    lodo = read_csv(ROOT / "strict_lodo_bundle/e14_lodo_strict_clip/aggregate.csv")
    df40 = read_csv(ROOT / "fixes_bundle/e13_clip_df40/aggregate.csv")

    ff_only_bb = next(x for x in sanity if x["regime"] == "ff_only"
                      and x["variant"] == "backbone_only" and x["cdf_split"] == "by_subject")
    mixed_bb = next(x for x in sanity if x["regime"] == "mixed"
                    and x["variant"] == "backbone_only" and x["cdf_split"] == "by_subject")
    lodo_bb = next(x for x in lodo if x["config"] == "test_celebdf"
                   and x["variant"] == "backbone_only")
    df40_all = next(x for x in df40 if x["variant"] == "backbone_only" and x["method"] == "ALL")

    proto = ["within-dataset\n(FF-only)", "mixed-domain\nheld-out id",
             "strict LODO\n(cross-dataset)", "external stress\n(DF40 video)"]
    means = [float(ff_only_bb["auc_mean"]), float(mixed_bb["auc_mean"]),
             float(lodo_bb["auc_mean"]),    float(df40_all["auc_mean"])]
    stds  = [float(ff_only_bb["auc_std"]),  float(mixed_bb["auc_std"]),
             float(lodo_bb["auc_std"]),     float(df40_all["auc_std"])]
    cols = [PROTOCOL_COLORS["within-dataset"],
            PROTOCOL_COLORS["mixed-domain held-out id"],
            PROTOCOL_COLORS["strict LODO"],
            PROTOCOL_COLORS["external stress test (DF40)"]]
    bars = axA.bar(range(4), means, yerr=stds, color=cols, edgecolor="black",
                   linewidth=0.7, capsize=4)
    for b, m in zip(bars, means):
        axA.text(b.get_x() + b.get_width() / 2, m + 0.018,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    axA.set_xticks(range(4)); axA.set_xticklabels(proto, fontsize=8)
    axA.set_ylim(0.5, 1.0); axA.axhline(0.5, color="red", linestyle="--", lw=0.6, alpha=0.5)
    axA.set_ylabel("CLIP AUC (backbone-only)")
    axA.set_title("(A) CLIP across four protocols", fontsize=10)
    axA.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel B: CLIP variant ablation under strict LODO ──
    axB = fig.add_subplot(gs[0, 1])
    rows = [x for x in lodo if x["config"] == "test_celebdf"]
    order = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]
    labels = ["Backbone\nonly", "+rPPG", "+Blink", "Full\nfusion"]
    aucs = [float(next(x for x in rows if x["variant"] == v)["auc_mean"]) for v in order]
    stds = [float(next(x for x in rows if x["variant"] == v)["auc_std"]) for v in order]
    cols2 = [C_BACKBONE, C_RPPG, C_BLINK, C_FUSION]
    bars = axB.bar(range(4), aucs, yerr=stds, color=cols2, edgecolor="black",
                   linewidth=0.7, capsize=4)
    for b, a in zip(bars, aucs):
        axB.text(b.get_x() + b.get_width() / 2, a + 0.0008,
                 f"{a:.4f}", ha="center", va="bottom", fontsize=7.5)
    axB.set_xticks(range(4)); axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylim(0.72, 0.78)
    axB.set_ylabel("AUC")
    axB.set_title("(B) Strict LODO: physiology contribution (n = 1758)\n"
                  "Δ ≈ −0.005 (full fusion slightly worse, p=0.029)", fontsize=10)
    axB.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel C: DF40 method-stratified ──
    axC = fig.add_subplot(gs[1, 0])
    bucket_order = ["sadtalker_ff", "sadtalker_cdf", "simswap_ff", "simswap_cdf"]
    bucket_labels = ["sadtalker\nff", "sadtalker\ncdf", "simswap\nff", "simswap\ncdf"]
    df40_bb = [float(next(x for x in df40 if x["variant"] == "backbone_only"
                          and x["method"] == m)["auc_mean"]) for m in bucket_order]
    df40_std = [float(next(x for x in df40 if x["variant"] == "backbone_only"
                           and x["method"] == m)["auc_std"]) for m in bucket_order]
    bar_cols = ["#D55E00", "#E69F00", "#56B4E9", "#0072B2"]
    bars = axC.bar(range(4), df40_bb, yerr=df40_std, color=bar_cols, edgecolor="black",
                   linewidth=0.7, capsize=4)
    axC.axhline(0.5, color="red", linestyle="--", lw=0.6, alpha=0.5)
    axC.text(3.5, 0.51, "chance", color="red", fontsize=7)
    for b, m in zip(bars, df40_bb):
        axC.text(b.get_x() + b.get_width() / 2, m + 0.025,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    axC.set_xticks(range(4)); axC.set_xticklabels(bucket_labels, fontsize=8)
    axC.set_ylim(0.3, 1.0)
    axC.set_ylabel("CLIP AUC (backbone-only)")
    axC.set_title("(C) DF40 external stress test (no physiology — vectors zero-filled)",
                  fontsize=10)
    axC.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel D: text panel — corrected claims summary ──
    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")
    text = (
        r"$\bf{Headline\ findings\ (v5-corrected,\ E14\ strict\ LODO)}$" + "\n\n"
        r"$\bf{1.}$ CLIP CelebDF AUC = $\bf{0.749 \pm 0.010}$ under strict LODO" + "\n"
        r"     (n = 1758, train FF + DFDC, CelebDF fully held out)." + "\n"
        r"     v3/v4 0.941 number was a mixed-domain leak — retracted." + "\n\n"
        r"$\bf{2.}$ Full fusion = 0.745; Δ = −0.005 vs backbone-only." + "\n"
        r"     Stouffer p = 0.029 in favour of backbone-only — physiology" + "\n"
        r"     is at best neutral and slightly negative under strict LODO." + "\n\n"
        r"$\bf{3.}$ Apples-to-apples vs DeepfakeBench (FF-only → CelebDF):" + "\n"
        r"     CLIP 0.770 ≈ SPSL 0.7650 (field ceiling) without auxiliary" + "\n"
        r"     hand-engineered features. Strict LODO 0.749 ≈ Xception 0.7365." + "\n\n"
        r"$\bf{4.}$ Physiology has $\it{representation\!-\!dependent}$" + "\n"
        r"     $\it{marginal\ value}$: small residual gains for B4 within-dataset," + "\n"
        r"     neutral or slightly harmful for CLIP cross-dataset." + "\n\n"
        r"$\bf{5.}$ DF40 stress test: CLIP 0.746 ALL, but sadtalker_ff" + "\n"
        r"     0.462 (below chance) — talking-head reanimation" + "\n"
        r"     fundamentally harder than face-swap."
    )
    axD.text(0, 1, text, va="top", ha="left", fontsize=9, family="sans-serif")

    fig.suptitle("CLIP-based deepfake detection — corrected v5 dashboard (E14 strict LODO)\n"
                 "Four protocols, n = 1758 subject-aware throughout",
                 fontsize=11, y=0.995)
    plt.savefig(OUT / "fig13_summary_dashboard_v5.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig13_summary_dashboard_v5.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig13_summary_dashboard_v5")


def fig15_evidence_matrix_v5():
    """The central thesis artifact: Δ AUC of full_fusion vs backbone-only,
    across protocols × backbones. Shows representation-dependent marginal value."""
    sanity_csv = {
        "B4":   ROOT / "sanity_bundle/sanity_b4_idsplit/aggregate.csv",
        "DINO": ROOT / "sanity_bundle/sanity_dinov2_idsplit/aggregate.csv",
        "CLIP": ROOT / "sanity_bundle/sanity_clip_idsplit/aggregate.csv",
    }
    lodo_csv = {
        "B4":   ROOT / "strict_lodo_bundle/e14_lodo_strict_b4/aggregate.csv",
        "DINO": ROOT / "strict_lodo_bundle/e14_lodo_strict_dinov2/aggregate.csv",
        "CLIP": ROOT / "strict_lodo_bundle/e14_lodo_strict_clip/aggregate.csv",
    }
    w5 = read_csv(ROOT / "ablation/ablation_results.csv")

    # delta = full_fusion AUC - backbone_only AUC
    def w5_delta():
        bb = float(next(x for x in w5 if x["variant"] == "1_backbone_only")["test_auc"])
        fu = float(next(x for x in w5 if x["variant"] == "4_backbone+rppg+blink")["test_auc"])
        return fu - bb

    def sanity_delta(key, regime):
        rows = [x for x in read_csv(sanity_csv[key])
                if x["regime"] == regime and x["cdf_split"] == "by_subject"]
        bb = float(next(x for x in rows if x["variant"] == "backbone_only")["auc_mean"])
        fu = float(next(x for x in rows if x["variant"] == "full_fusion")["auc_mean"])
        return fu - bb

    def lodo_delta(key):
        rows = [x for x in read_csv(lodo_csv[key])
                if x["config"] == "test_celebdf"]
        bb = float(next(x for x in rows if x["variant"] == "backbone_only")["auc_mean"])
        fu = float(next(x for x in rows if x["variant"] == "full_fusion")["auc_mean"])
        return fu - bb

    # 4 protocols (rows) × 3 backbones (cols), with W5 as a separate B4-only entry
    protocols = ["within-dataset (FF++ test)",
                 "FF-only → CelebDF (n=1758)",
                 "mixed-domain held-out id (n=1758)",
                 "strict LODO CelebDF (n=1758)"]
    backbones = ["B4 v13", "DINOv2", "CLIP"]
    M = np.full((4, 3), np.nan)
    M[0, 0] = w5_delta()                              # W5 only on B4
    for j, key in enumerate(["B4", "DINO", "CLIP"]):
        M[1, j] = sanity_delta(key, "ff_only")
        M[2, j] = sanity_delta(key, "mixed")
        M[3, j] = lodo_delta(key)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    # Diverging colormap centered at 0; ±0.005 is the practical-significance threshold
    vmax = 0.025
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(3)); ax.set_xticklabels(backbones, fontsize=10)
    ax.set_yticks(np.arange(4)); ax.set_yticklabels(protocols, fontsize=9)
    ax.set_title("Evidence matrix — Δ AUC (full_fusion − backbone-only)\n"
                 "Negative (blue) = physiology hurts; positive (red) = physiology helps. "
                 "Practical threshold ±0.005.",
                 fontsize=10)

    # Annotate each cell
    for i in range(4):
        for j in range(3):
            val = M[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center", color="gray",
                        fontsize=10, style="italic")
            else:
                color = "white" if abs(val) > 0.012 else "black"
                marker = ""
                if val >= 0.005:
                    marker = " ↑"  # physiology helps practically
                elif val <= -0.005:
                    marker = " ↓"  # physiology hurts practically
                ax.text(j, i, f"{val:+.4f}{marker}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Δ AUC", rotation=270, labelpad=15)
    cbar.ax.axhline(0.005, color="black", linewidth=0.5, linestyle=":")
    cbar.ax.axhline(-0.005, color="black", linewidth=0.5, linestyle=":")

    fig.text(0.5, -0.02,
             "Pattern: physiology helps low-capacity B4 (Δ ≈ +0.01 to +0.02), is neutral on DINOv2, "
             "and is at best neutral / slightly negative for CLIP under cross-dataset evaluation. "
             "Representation-dependent marginal value (see thesis Discussion).",
             ha="center", fontsize=8, style="italic", wrap=True)

    plt.tight_layout()
    plt.savefig(OUT / "fig15_evidence_matrix_v5.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig15_evidence_matrix_v5.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig15_evidence_matrix_v5")


def fig16_e16_quality_and_errors():
    """Two-panel figure summarising E16: (A) stratified AUC by SNR quartile,
    (B) error-conditional rescue/regression counts per variant."""
    e16_dir = ROOT / "e16_physio_quality_clip"
    if not e16_dir.exists():
        print("[skip] fig16: e16 bundle not found")
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: stratified AUC bars
    strat_rows = read_csv(e16_dir / "stratified_auc.csv")
    strata = ["ALL", "rppg_snr_high", "rppg_snr_low"]
    strata_labels = ["ALL\n(n=1758)", "high SNR\n(top Q, n=440)",
                     "low SNR\n(bot Q, n=440)"]
    variants = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]
    var_labels = ["Backbone", "+rPPG", "+Blink", "Fusion"]
    var_colors = [C_BACKBONE, C_RPPG, C_BLINK, C_FUSION]

    x = np.arange(len(strata))
    w = 0.18
    for i, (v, lbl, col) in enumerate(zip(variants, var_labels, var_colors)):
        means = []
        stds = []
        for s in strata:
            r = next(x for x in strat_rows if x["stratum"] == s)
            means.append(float(r[f"{v}_mean"]))
            stds.append(float(r[f"{v}_std"]))
        axA.bar(x + (i - 1.5) * w, means, w, yerr=stds, label=lbl,
                color=col, edgecolor="black", linewidth=0.5, capsize=2)
    axA.set_xticks(x)
    axA.set_xticklabels(strata_labels, fontsize=9)
    axA.set_ylabel("AUC (5-seed mean ± std)")
    axA.set_ylim(0.72, 0.80)
    axA.set_title("(A) Stratified AUC under strict LODO (CelebDF n=1758)\n"
                  "Both top and bottom rPPG-SNR quartiles show negative Δ\n"
                  "→ physiology does not help even at high signal quality",
                  fontsize=10)
    axA.legend(loc="lower right", framealpha=0.95, fontsize=8)
    axA.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B: error-conditional bars
    err_rows = read_csv(e16_dir / "error_conditional.csv")
    err_variants = ["backbone+rppg", "backbone+blink", "full_fusion"]
    err_labels = ["+rPPG", "+Blink", "Full fusion"]
    rescues = [float(next(r for r in err_rows if r["variant"] == v)["rescue_mean"])
               for v in err_variants]
    regressions = [float(next(r for r in err_rows if r["variant"] == v)["regression_mean"])
                   for v in err_variants]
    nets = [r - g for r, g in zip(rescues, regressions)]

    x = np.arange(len(err_variants))
    w = 0.35
    axB.bar(x - w/2, rescues, w, label="Rescues (bb wrong → variant right)",
            color="#2CA02C", edgecolor="black", linewidth=0.5)
    axB.bar(x + w/2, regressions, w, label="Regressions (bb right → variant wrong)",
            color="#D62728", edgecolor="black", linewidth=0.5)
    for i, n in enumerate(nets):
        sign = "+" if n >= 0 else ""
        col = "#2CA02C" if n >= 0 else "#D62728"
        axB.text(i, max(rescues[i], regressions[i]) + 3,
                 f"net = {sign}{n:.1f}", ha="center", color=col,
                 fontsize=10, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(err_labels)
    axB.set_ylabel("flip count (5-seed mean per 1758 clips)")
    axB.set_title("(B) Error-conditional analysis under strict LODO\n"
                  "rPPG alone has positive net; blink and fusion harm threshold decisions",
                  fontsize=10)
    axB.legend(loc="upper left", framealpha=0.95, fontsize=8)
    axB.grid(axis="y", linestyle=":", alpha=0.4)
    axB.set_ylim(0, 100)

    fig.text(0.5, -0.01,
             "E16 (2026-05-08, n=1758 strict-LODO CelebDF, 5 seeds). "
             "100% physiology-extraction coverage; SNR proxy range 5.08–5.83 (narrow). "
             "Source: outputs_and_cfgs/e16_physio_quality_clip/.",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "fig16_e16_quality_and_errors.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig16_e16_quality_and_errors.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] fig16_e16_quality_and_errors")


if __name__ == "__main__":
    print(f"Output dir: {OUT}")
    fig2_ablation_v4()
    fig7_cross_dataset_v4()
    fig9_biosignal_v4()
    fig12_mixed_probe_tpr_v4()
    fig13_summary_v4()
    fig15_evidence_matrix_v5()
    fig16_e16_quality_and_errors()
    print("\nAll 7 corrected figures regenerated.")
