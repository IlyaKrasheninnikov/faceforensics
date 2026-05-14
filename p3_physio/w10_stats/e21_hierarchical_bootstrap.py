"""
E21 - Hierarchical paired (subject + seed) bootstrap + threshold CIs +
       cross-backbone (B4 vs CLIP) representation-dependent comparison.

Addresses professor v9 feedback (4 of 5 points; point 3 is a figure
fix done separately, point 5 is a no-op confirmation):

1. **Hierarchical paired subject+seed bootstrap on Delta AUC.** v9's
   E20 averaged CI endpoints across 5 seeds — that's not a proper
   bootstrap. v9 §1 should be replaced with a hierarchical bootstrap:
   each iteration resamples BOTH subjects (with replacement) AND a
   seed (with replacement), then computes the paired delta AUC. The
   single resulting CI captures both seed variance and subject
   variance jointly.

2. **Confidence intervals on the threshold rescue/regression table.**
   Currently rescue/regression counts are mean across 5 seeds with
   no CI. Add per-stratum subject-cluster bootstrap CIs to the
   threshold-level rescue/regression counts. Mark the FPR thresholds
   as "diagnostic-only, not source-validation-calibrated".

3. (figure fix is in plot_v5_corrected.py; not in this script)

4. **B4 cross-backbone delta analysis.** Run the same hierarchical
   paired subject-cluster bootstrap on Delta AUC for EfficientNet-B4
   under strict LODO. The "representation-dependent marginal value"
   claim needs B4 to show statistically defensible positive Delta in
   contrast to CLIP's neutral-to-negative Delta. If B4's Delta CI
   excludes 0 on the positive side, the cross-backbone claim is
   statistically supported.

All three analyses run locally from the existing E14 strict-LODO score
arrays for B4 and CLIP, plus the E16 quality CSV.

Outputs (in --out_dir):
  e21_hierarchical_paired_clip.csv  - hierarchical paired delta AUC CI per variant x stratum (CLIP)
  e21_hierarchical_paired_b4.csv    - same for B4
  e21_threshold_curves_with_ci.csv  - threshold rescue/regression with subject-cluster CI
  e21_summary.json
  e21_findings.md

Usage:
    python p3_physio/w10_stats/e21_hierarchical_bootstrap.py \\
        --quality_csv      p3_physio/outputs_and_cfgs/e16_physio_quality_clip/quality_metrics.csv \\
        --clip_scores_dir  p3_physio/outputs_and_cfgs/strict_lodo_bundle/e14_lodo_strict_clip/scores \\
        --b4_scores_dir    p3_physio/outputs_and_cfgs/strict_lodo_bundle/e14_lodo_strict_b4/scores \\
        --out_dir          p3_physio/outputs_and_cfgs/e21_v10_bundle
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = [0, 1, 42, 1337, 2024]
VARIANTS = ["backbone+rppg", "backbone+blink", "full_fusion"]
FPR_LEVELS = [0.01, 0.05, 0.10]
N_BOOT = 2000
ALPHA = 0.05


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return 0.5
    idx = np.argsort(-scores)
    y = labels[idx]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    tp = np.cumsum(y) / n1
    fp = np.cumsum(1 - y) / n0
    return float(np.trapezoid(tp, fp)) if hasattr(np, "trapezoid") else float(np.trapz(tp, fp))


def fpr_threshold(labels, scores, target_fpr):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    neg = scores[labels == 0]
    if len(neg) == 0:
        return float(scores.max())
    return float(np.quantile(neg, 1.0 - target_fpr))


def hierarchical_paired_bootstrap_delta_auc(labels, scores_per_seed_bb,
                                            scores_per_seed_v, subjects,
                                            n_boot=N_BOOT, alpha=ALPHA, seed=0,
                                            mask=None):
    """Hierarchical paired bootstrap: jointly resample subjects and seeds.

    Each bootstrap iteration:
      1. Sample 1 seed uniformly from the available seeds (with replacement
         across iterations).
      2. Sample n_subjects unique subjects (with replacement) from the
         in-mask subject pool.
      3. Concatenate all clips of each chosen subject.
      4. Compute paired delta AUC = AUC(v) - AUC(bb) on the same resampled
         clips, using the chosen seed's scores for both v and bb.

    This combines seed variance and subject variance into a single CI.

    Returns (lo, hi, mean, std, n_subj_in_mask).
    """
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    keep_subjects = np.unique(np.asarray(subjects)[mask])
    n_subj = len(keep_subjects)
    if n_subj < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(n_subj)

    subjects_arr = np.asarray(subjects)
    subj_to_clips = {s: np.where((subjects_arr == s) & mask)[0]
                     for s in keep_subjects}

    seeds_avail = list(scores_per_seed_bb.keys())
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    valid_count = 0
    for b in range(n_boot):
        chosen_seed = seeds_avail[rng.integers(0, len(seeds_avail))]
        chosen_subj = rng.choice(keep_subjects, size=n_subj, replace=True)
        idx_list = [subj_to_clips[s] for s in chosen_subj]
        idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
        if len(idx) == 0 or len(np.unique(labels[idx])) < 2:
            deltas[b] = np.nan
            continue
        sa = scores_per_seed_bb[chosen_seed][idx]
        sb = scores_per_seed_v[chosen_seed][idx]
        deltas[b] = roc_auc(labels[idx], sb) - roc_auc(labels[idx], sa)
        valid_count += 1

    valid = deltas[~np.isnan(deltas)]
    if len(valid) < 50:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(n_subj)
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return lo, hi, float(valid.mean()), float(valid.std(ddof=1)), int(n_subj)


def threshold_flip_counts_per_seed(labels, scores_bb, scores_v, target_fpr,
                                   stratum_mask=None):
    """For one seed: set thresholds at FPR=target on each probe (independently),
    return rescue and regression counts on stratum_mask."""
    if stratum_mask is None:
        stratum_mask = np.ones(len(labels), dtype=bool)
    thr_bb = fpr_threshold(labels, scores_bb, target_fpr)
    thr_v  = fpr_threshold(labels, scores_v, target_fpr)
    pred_bb = (scores_bb >= thr_bb).astype(int)
    pred_v  = (scores_v  >= thr_v).astype(int)
    correct_bb = (pred_bb == labels).astype(int)
    correct_v  = (pred_v  == labels).astype(int)
    rescue = (correct_bb == 0) & (correct_v == 1) & stratum_mask
    regr   = (correct_bb == 1) & (correct_v == 0) & stratum_mask
    return int(rescue.sum()), int(regr.sum())


def hierarchical_subject_bootstrap_threshold_flips(labels,
                                                   scores_per_seed_bb,
                                                   scores_per_seed_v,
                                                   subjects, target_fpr,
                                                   n_boot=N_BOOT, alpha=ALPHA,
                                                   seed=0, mask=None):
    """Bootstrap CI for (rescue, regression, net) flip counts at fixed FPR.

    Each bootstrap iteration:
      1. Sample 1 seed.
      2. Sample n_subjects with replacement.
      3. Concatenate clips of chosen subjects (within mask).
      4. Apply per-probe per-seed threshold (set at FPR=target on the FULL
         test set per seed, so threshold is fixed across bootstrap
         resampling — diagnostic interpretation).
      5. Count rescues / regressions on the resampled stratum-clips.

    Reports mean +/- bootstrap CI for rescue, regression, net.
    """
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    keep_subjects = np.unique(np.asarray(subjects)[mask])
    n_subj = len(keep_subjects)
    if n_subj < 2:
        return None

    subjects_arr = np.asarray(subjects)
    subj_to_clips = {s: np.where((subjects_arr == s) & mask)[0]
                     for s in keep_subjects}

    # Pre-compute per-seed thresholds on the full set
    seeds_avail = list(scores_per_seed_bb.keys())
    thr_bb_per_seed = {s: fpr_threshold(labels, scores_per_seed_bb[s], target_fpr)
                       for s in seeds_avail}
    thr_v_per_seed  = {s: fpr_threshold(labels, scores_per_seed_v[s], target_fpr)
                       for s in seeds_avail}

    rng = np.random.default_rng(seed)
    rescue_counts = np.empty(n_boot)
    regr_counts   = np.empty(n_boot)
    for b in range(n_boot):
        chosen_seed = seeds_avail[rng.integers(0, len(seeds_avail))]
        chosen_subj = rng.choice(keep_subjects, size=n_subj, replace=True)
        idx_list = [subj_to_clips[s] for s in chosen_subj]
        idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
        if len(idx) == 0:
            rescue_counts[b] = np.nan; regr_counts[b] = np.nan
            continue
        bb_sc = scores_per_seed_bb[chosen_seed][idx]
        v_sc  = scores_per_seed_v[chosen_seed][idx]
        y = labels[idx]
        thr_bb = thr_bb_per_seed[chosen_seed]
        thr_v  = thr_v_per_seed[chosen_seed]
        pred_bb = (bb_sc >= thr_bb).astype(int)
        pred_v  = (v_sc  >= thr_v).astype(int)
        correct_bb = (pred_bb == y).astype(int)
        correct_v  = (pred_v  == y).astype(int)
        rescue_counts[b] = int(((correct_bb == 0) & (correct_v == 1)).sum())
        regr_counts[b]   = int(((correct_bb == 1) & (correct_v == 0)).sum())

    valid_mask = (~np.isnan(rescue_counts)) & (~np.isnan(regr_counts))
    rescue_counts = rescue_counts[valid_mask]
    regr_counts = regr_counts[valid_mask]
    net_counts = rescue_counts - regr_counts

    return {
        "rescue_mean": float(rescue_counts.mean()),
        "rescue_ci_lo": float(np.quantile(rescue_counts, alpha / 2)),
        "rescue_ci_hi": float(np.quantile(rescue_counts, 1 - alpha / 2)),
        "regression_mean": float(regr_counts.mean()),
        "regression_ci_lo": float(np.quantile(regr_counts, alpha / 2)),
        "regression_ci_hi": float(np.quantile(regr_counts, 1 - alpha / 2)),
        "net_mean": float(net_counts.mean()),
        "net_ci_lo": float(np.quantile(net_counts, alpha / 2)),
        "net_ci_hi": float(np.quantile(net_counts, 1 - alpha / 2)),
    }


def run_hierarchical_paired_table(label, scores_dir, labels, subjects,
                                  rppg_snr, blink_int, out_csv):
    """Compute hierarchical paired bootstrap delta AUC CIs for one backbone."""
    scores_per_seed_variant = {}
    for s in SEEDS:
        d_bb = np.load(str(Path(scores_dir) / f"test_celebdf_s{s}_backbone_only.npz"))
        if not np.array_equal(d_bb["labels"].astype(int), labels):
            raise RuntimeError(f"[{label}] label mismatch seed {s} backbone_only")
        scores_per_seed_variant[(s, "backbone_only")] = d_bb["scores"]
        for v in VARIANTS:
            d_v = np.load(str(Path(scores_dir) / f"test_celebdf_s{s}_{v}.npz"))
            scores_per_seed_variant[(s, v)] = d_v["scores"]

    snr_q3 = np.quantile(rppg_snr, 0.75)
    snr_q1 = np.quantile(rppg_snr, 0.25)
    blink_q3 = np.quantile(blink_int, 0.75)
    blink_q1 = np.quantile(blink_int, 0.25)
    n = len(labels)
    # Note: real-only / fake-only strata are excluded from paired AUC tests
    # because AUC is undefined on a single-class subset. They appear only in
    # the threshold-rescue analysis below.
    strata = {
        "ALL":               np.ones(n, dtype=bool),
        "rppg_snr_high_Q":   rppg_snr >= snr_q3,
        "rppg_snr_low_Q":    rppg_snr <= snr_q1,
        "blink_int_high_Q":  blink_int >= blink_q3,
        "blink_int_low_Q":   blink_int <= blink_q1,
    }

    print(f"\n[{label}] Hierarchical paired (subject+seed) bootstrap on delta AUC")
    print(f"{'variant':<18s} {'stratum':<20s} {'n_subj':>7s} | "
          f"{'mean d':>9s} {'CI lo':>9s} {'CI hi':>9s} | excludes 0?")
    print("-" * 95)
    rows = []
    for variant in VARIANTS:
        bb_per_seed = {s: scores_per_seed_variant[(s, "backbone_only")]
                       for s in SEEDS}
        v_per_seed  = {s: scores_per_seed_variant[(s, variant)] for s in SEEDS}
        for stratum_name, mask in strata.items():
            lo, hi, mn, sd, n_subj = hierarchical_paired_bootstrap_delta_auc(
                labels, bb_per_seed, v_per_seed, subjects,
                n_boot=N_BOOT, alpha=ALPHA, seed=hash((label, variant, stratum_name)) & 0xFFFFFFFF,
                mask=mask)
            excludes = (lo > 0) or (hi < 0)
            rows.append({
                "backbone": label, "variant": variant, "stratum": stratum_name,
                "n_subjects": n_subj, "mean_delta_auc": mn,
                "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": excludes,
            })
            ex = "YES" if excludes else "no"
            print(f"{variant:<18s} {stratum_name:<20s} {n_subj:>7d} | "
                  f"{mn:>+9.4f} {lo:>+9.4f} {hi:>+9.4f} | {ex}")
        print()

    head = ["backbone", "variant", "stratum", "n_subjects",
            "mean_delta_auc", "ci_lo", "ci_hi", "ci_excludes_zero"]
    with open(out_csv, "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float)
                             else str(r[k]) for k in head) + "\n")
    print(f"[{label}] wrote {out_csv}")
    return rows, scores_per_seed_variant


def run_threshold_curves_with_ci(label, scores_per_seed_variant, labels,
                                 subjects, rppg_snr, blink_int, out_csv):
    """Threshold-level rescue/regression with subject-cluster bootstrap CI."""
    snr_q3 = np.quantile(rppg_snr, 0.75)
    blink_q3 = np.quantile(blink_int, 0.75)
    blink_q1 = np.quantile(blink_int, 0.25)
    n = len(labels)
    strata = {
        "ALL":               np.ones(n, dtype=bool),
        "rppg_snr_high_Q":   rppg_snr >= snr_q3,
        "blink_int_high_Q":  blink_int >= blink_q3,
        "blink_int_low_Q":   blink_int <= blink_q1,
        "real":              labels == 0,
        "fake":              labels == 1,
    }

    print(f"\n[{label}] Threshold-level rescue/regression with bootstrap CIs")
    print(f"{'variant':<18s} {'stratum':<20s} {'FPR':>5s} | "
          f"{'rescue (CI)':>22s} {'regr (CI)':>22s} {'net (CI)':>22s}")
    print("-" * 105)
    rows = []
    for variant in VARIANTS:
        bb_per_seed = {s: scores_per_seed_variant[(s, "backbone_only")] for s in SEEDS}
        v_per_seed  = {s: scores_per_seed_variant[(s, variant)] for s in SEEDS}
        for stratum_name, mask in strata.items():
            for fpr_target in FPR_LEVELS:
                result = hierarchical_subject_bootstrap_threshold_flips(
                    labels, bb_per_seed, v_per_seed, subjects, fpr_target,
                    n_boot=N_BOOT, alpha=ALPHA,
                    seed=hash((label, variant, stratum_name, fpr_target)) & 0xFFFFFFFF,
                    mask=mask)
                if result is None:
                    continue
                row = {"backbone": label, "variant": variant,
                       "stratum": stratum_name, "fpr_target": fpr_target,
                       **result}
                rows.append(row)
                rescue_str = f"{result['rescue_mean']:.1f} [{result['rescue_ci_lo']:.0f},{result['rescue_ci_hi']:.0f}]"
                regr_str = f"{result['regression_mean']:.1f} [{result['regression_ci_lo']:.0f},{result['regression_ci_hi']:.0f}]"
                net_str = f"{result['net_mean']:+.1f} [{result['net_ci_lo']:+.0f},{result['net_ci_hi']:+.0f}]"
                print(f"{variant:<18s} {stratum_name:<20s} "
                      f"{fpr_target*100:>4.0f}% | "
                      f"{rescue_str:>22s} {regr_str:>22s} {net_str:>22s}")
        print()

    head = ["backbone", "variant", "stratum", "fpr_target",
            "rescue_mean", "rescue_ci_lo", "rescue_ci_hi",
            "regression_mean", "regression_ci_lo", "regression_ci_hi",
            "net_mean", "net_ci_lo", "net_ci_hi"]
    with open(out_csv, "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float)
                             else str(r[k]) for k in head) + "\n")
    print(f"[{label}] wrote {out_csv}")
    return rows


def main(args):
    qrows = []
    with open(args.quality_csv) as f:
        for r in csv.DictReader(f):
            qrows.append({
                "clip_idx": int(r["clip_idx"]),
                "subject_id": r["subject_id"],
                "label": int(float(r["label"])),
                "rppg_snr": float(r["rppg_snr"]),
                "blink_intensity": float(r["blink_intensity"]),
            })
    n = len(qrows)
    subjects = np.array([r["subject_id"] for r in qrows])
    labels = np.array([r["label"] for r in qrows])
    rppg_snr = np.array([r["rppg_snr"] for r in qrows])
    blink_int = np.array([r["blink_intensity"] for r in qrows])
    print(f"[e21] n_clips={n}  n_subjects={len(np.unique(subjects))}")
    print(f"[e21] N_BOOT={N_BOOT}  alpha={ALPHA}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 1+4: Hierarchical paired bootstrap on delta AUC for both backbones
    clip_paired_rows, clip_scores = run_hierarchical_paired_table(
        "CLIP", args.clip_scores_dir, labels, subjects, rppg_snr, blink_int,
        out_dir / "e21_hierarchical_paired_clip.csv")

    b4_paired_rows, b4_scores = run_hierarchical_paired_table(
        "B4", args.b4_scores_dir, labels, subjects, rppg_snr, blink_int,
        out_dir / "e21_hierarchical_paired_b4.csv")

    # Part 2: Threshold-level rescue/regression with bootstrap CIs (CLIP + B4)
    threshold_rows = run_threshold_curves_with_ci(
        "CLIP", clip_scores, labels, subjects, rppg_snr, blink_int,
        out_dir / "e21_threshold_curves_with_ci.csv")
    threshold_rows_b4 = run_threshold_curves_with_ci(
        "B4", b4_scores, labels, subjects, rppg_snr, blink_int,
        out_dir / "e21_threshold_curves_with_ci_b4.csv")

    # Summary JSON
    with open(out_dir / "e21_summary.json", "w") as f:
        json.dump({
            "n_clips": int(n),
            "n_subjects": int(len(np.unique(subjects))),
            "n_boot": N_BOOT, "alpha": ALPHA,
            "seeds": SEEDS, "variants": VARIANTS, "fpr_levels": FPR_LEVELS,
            "clip_paired": clip_paired_rows,
            "b4_paired": b4_paired_rows,
            "clip_threshold_curves": threshold_rows,
            "b4_threshold_curves": threshold_rows_b4,
        }, f, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating)
                        else int(o) if isinstance(o, np.integer)
                        else bool(o) if isinstance(o, np.bool_)
                        else str(o))
    print(f"\n[e21] wrote {out_dir/'e21_summary.json'}")

    # Markdown findings
    with open(out_dir / "e21_findings.md", "w", encoding="utf-8") as f:
        f.write("# E21 - Hierarchical paired bootstrap + threshold CIs + B4 vs CLIP\n\n")
        f.write(f"**Date:** 2026-05-10  \n")
        f.write(f"**N_BOOT:** {N_BOOT}, alpha = {ALPHA}\n\n")

        f.write("## 1. CLIP - hierarchical paired (subject+seed) bootstrap on delta AUC\n\n")
        f.write("Single CI per (variant, stratum) computed jointly over subject "
                "and seed resampling (NOT averaged across seeds).\n\n")
        f.write("| Variant | Stratum | n_subj | mean delta | 95% CI | excludes 0? |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in clip_paired_rows:
            ex = "**YES**" if r["ci_excludes_zero"] else "no"
            f.write(f"| {r['variant']} | {r['stratum']} | {r['n_subjects']} | "
                    f"{r['mean_delta_auc']:+.4f} | "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {ex} |\n")

        f.write("\n## 2. B4 - hierarchical paired bootstrap on delta AUC\n\n")
        f.write("Same procedure for EfficientNet-B4 v13 strict-LODO scores. "
                "If the B4 deltas show CIs that exclude zero on the positive "
                "side, the representation-dependent claim is statistically "
                "supported.\n\n")
        f.write("| Variant | Stratum | n_subj | mean delta | 95% CI | excludes 0? |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in b4_paired_rows:
            ex = "**YES**" if r["ci_excludes_zero"] else "no"
            f.write(f"| {r['variant']} | {r['stratum']} | {r['n_subjects']} | "
                    f"{r['mean_delta_auc']:+.4f} | "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {ex} |\n")

        f.write("\n## 3. CLIP - threshold-level rescue/regression with subject-cluster CIs\n\n")
        f.write("**Diagnostic interpretation only.** The FPR thresholds are set "
                "directly on the test partition for each probe per seed; this is "
                "an oracle calibration, not a deployment-validatable threshold. "
                "Recalibration on a source-domain validation set would shift the "
                "thresholds.\n\n")
        f.write("| Variant | Stratum | FPR | rescue [CI] | regression [CI] | net [CI] |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in threshold_rows:
            f.write(f"| {r['variant']} | {r['stratum']} | "
                    f"{r['fpr_target']*100:.0f}% | "
                    f"{r['rescue_mean']:.1f} [{r['rescue_ci_lo']:.1f}, {r['rescue_ci_hi']:.1f}] | "
                    f"{r['regression_mean']:.1f} [{r['regression_ci_lo']:.1f}, {r['regression_ci_hi']:.1f}] | "
                    f"{r['net_mean']:+.1f} [{r['net_ci_lo']:+.1f}, {r['net_ci_hi']:+.1f}] |\n")

        f.write("\n## 4. Cross-backbone comparison\n\n")
        # Compare B4 ALL delta vs CLIP ALL delta for each variant
        f.write("Direct B4-vs-CLIP comparison on the ALL stratum:\n\n")
        f.write("| Variant | B4 mean delta [CI] | CLIP mean delta [CI] | B4-CLIP gap |\n")
        f.write("|---|---|---|---|\n")
        for variant in VARIANTS:
            b4_all = next(r for r in b4_paired_rows
                          if r["variant"] == variant and r["stratum"] == "ALL")
            cl_all = next(r for r in clip_paired_rows
                          if r["variant"] == variant and r["stratum"] == "ALL")
            gap = b4_all["mean_delta_auc"] - cl_all["mean_delta_auc"]
            f.write(f"| {variant} | "
                    f"{b4_all['mean_delta_auc']:+.4f} [{b4_all['ci_lo']:+.4f}, {b4_all['ci_hi']:+.4f}] | "
                    f"{cl_all['mean_delta_auc']:+.4f} [{cl_all['ci_lo']:+.4f}, {cl_all['ci_hi']:+.4f}] | "
                    f"{gap:+.4f} |\n")

    print(f"[e21] wrote {out_dir/'e21_findings.md'}")
    print(f"\n[e21] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality_csv", required=True)
    ap.add_argument("--clip_scores_dir", required=True)
    ap.add_argument("--b4_scores_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
