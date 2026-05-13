"""
E20 - Three decisive analyses for the v8 -> v9 iteration.

Addresses professor v8 feedback (three points):

1. **Paired subject-cluster bootstrap for deltas.** Extend E18 to give
   uncertainty for the *differences* delta AUC = variant - backbone-only,
   not just for absolute AUC. The paired bootstrap pairs each backbone-
   only score with each variant score on the same resampled subjects,
   so it cancels the between-subject variance that dominated E18's
   absolute CIs.

2. **Reliability-conditioned physiology test.** Test whether physiology
   helps only on reliable-signal clips. Using E16 quality metrics
   (rppg_snr proxy, blink_intensity), evaluate delta AUC on the top
   quartile of "reliable" clips per metric, with paired subject-
   cluster CIs. If quality-gated rPPG helps (positive delta with CI not
   straddling 0), that is a deployment-positive result.

3. **Threshold-level rescue/regression curves.** Extend the E16/E17
   rescue/regression analysis to thresholds set at FPR in {1%, 5%, 10%}
   instead of Youden. AUC hides the operating-point behaviour;
   this exposes how rescue/regression behaviour changes with
   operating point.

All three analyses run locally from existing artefacts:
  - E14 strict-LODO score arrays
  - E16 quality_metrics.csv (per-clip subject_id, rppg_snr, blink_intensity)

Outputs (in --out_dir):
  e20_paired_cis.csv          - per-seed-aggregated paired delta AUC CI per variant x stratum
  e20_threshold_curves.csv    - per-(threshold, variant, stratum): rescue/regression counts
  e20_summary.json            - bundle
  e20_findings.md             - markdown summary

Usage:
    python p3_physio/w10_stats/e20_paired_bootstrap_reliability_curves.py \\
        --quality_csv  p3_physio/outputs_and_cfgs/e16_physio_quality_clip/quality_metrics.csv \\
        --scores_dir   p3_physio/outputs_and_cfgs/strict_lodo_bundle/e14_lodo_strict_clip/scores \\
        --out_dir      p3_physio/outputs_and_cfgs/e20_v9_bundle
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SEEDS = [0, 1, 42, 1337, 2024]
VARIANTS = ["backbone+rppg", "backbone+blink", "full_fusion"]
FPR_LEVELS = [0.01, 0.05, 0.10]
N_BOOT = 2000  # 2000 bootstrap iterations
ALPHA = 0.05


def roc_auc(labels, scores):
    """Trapezoidal AUC. Returns 0.5 if degenerate."""
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
    """Threshold that gives the largest FPR <= target_fpr on (labels, scores)."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    neg = scores[labels == 0]
    if len(neg) == 0:
        return float(scores.max())
    return float(np.quantile(neg, 1.0 - target_fpr))


def paired_subject_bootstrap_delta_auc(labels, scores_bb, scores_v, subjects,
                                       n_boot=N_BOOT, alpha=ALPHA, seed=0,
                                       mask=None):
    """Paired subject-cluster bootstrap CI for delta AUC = AUC(v) - AUC(bb).

    For each bootstrap iteration:
      1. Resample unique subjects (within mask) with replacement.
      2. Concatenate all clips of each chosen subject.
      3. Compute AUC for bb and v on the SAME resampled clips (paired).
      4. Take delta.

    Returns (delta_lo, delta_hi, delta_mean, delta_std, n_subj_in_mask).
    """
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    keep_subjects = np.unique(np.asarray(subjects)[mask])
    n_subj = len(keep_subjects)
    if n_subj < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(n_subj)

    # Pre-build subject -> clip indices (within mask)
    subj_to_clips = {}
    subjects_arr = np.asarray(subjects)
    for s in keep_subjects:
        idx = np.where((subjects_arr == s) & mask)[0]
        subj_to_clips[s] = idx

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    skipped = 0
    for b in range(n_boot):
        chosen = rng.choice(keep_subjects, size=n_subj, replace=True)
        idx_list = [subj_to_clips[s] for s in chosen]
        idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
        if len(idx) == 0 or len(np.unique(labels[idx])) < 2:
            deltas[b] = np.nan
            skipped += 1
            continue
        a_bb = roc_auc(labels[idx], scores_bb[idx])
        a_v  = roc_auc(labels[idx], scores_v[idx])
        deltas[b] = a_v - a_bb

    valid = deltas[~np.isnan(deltas)]
    if len(valid) < 50:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(n_subj)
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return lo, hi, float(valid.mean()), float(valid.std(ddof=1)), int(n_subj)


def per_seed_flip_counts(labels, scores_bb, scores_v, threshold_fn,
                         stratum_mask=None):
    """For each seed, set thresholds independently per probe via threshold_fn,
    then count rescues (bb wrong -> v correct) and regressions on stratum_mask.
    """
    if stratum_mask is None:
        stratum_mask = np.ones(len(labels), dtype=bool)
    thr_bb = threshold_fn(labels, scores_bb)
    thr_v  = threshold_fn(labels, scores_v)
    pred_bb = (scores_bb >= thr_bb).astype(int)
    pred_v  = (scores_v  >= thr_v).astype(int)
    correct_bb = (pred_bb == labels).astype(int)
    correct_v  = (pred_v  == labels).astype(int)
    rescue = (correct_bb == 0) & (correct_v == 1) & stratum_mask
    regr   = (correct_bb == 1) & (correct_v == 0) & stratum_mask
    return int(rescue.sum()), int(regr.sum())


def main(args):
    # Load quality CSV
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
    print(f"[e20] n_clips={n}  n_subjects={len(np.unique(subjects))}")

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all score arrays, verify ordering
    scores_per_seed_variant = {}
    for s in SEEDS:
        d_bb = np.load(str(scores_dir / f"test_celebdf_s{s}_backbone_only.npz"))
        assert np.array_equal(d_bb["labels"].astype(int), labels), \
            f"Label mismatch seed {s} backbone_only"
        scores_per_seed_variant[(s, "backbone_only")] = d_bb["scores"]
        for v in VARIANTS:
            d_v = np.load(str(scores_dir / f"test_celebdf_s{s}_{v}.npz"))
            assert np.array_equal(d_v["labels"].astype(int), labels), \
                f"Label mismatch seed {s} {v}"
            scores_per_seed_variant[(s, v)] = d_v["scores"]
    print(f"[e20] loaded scores for {len(SEEDS)} seeds x 4 variants")

    # ---------------------------------------------------------
    # PART 1 - Paired subject-cluster bootstrap on delta AUC
    # ---------------------------------------------------------
    print(f"\n[e20] PART 1 - Paired subject-cluster bootstrap on delta AUC")
    snr_q3 = np.quantile(rppg_snr, 0.75)
    snr_q1 = np.quantile(rppg_snr, 0.25)
    blink_q3 = np.quantile(blink_int, 0.75)
    blink_q1 = np.quantile(blink_int, 0.25)

    strata_for_paired = {
        "ALL":                np.ones(n, dtype=bool),
        "rppg_snr_high_Q":    rppg_snr >= snr_q3,
        "rppg_snr_low_Q":     rppg_snr <= snr_q1,
        "blink_int_high_Q":   blink_int >= blink_q3,
        "blink_int_low_Q":    blink_int <= blink_q1,
    }
    paired_rows = []
    print(f"{'variant':<18s} {'stratum':<20s} {'n_subj':>7s} | "
          f"{'mean delta':>9s} {'CI lo':>9s} {'CI hi':>9s} | "
          f"{'CI excludes 0?':>15s}")
    print("-" * 95)

    for variant in VARIANTS:
        for stratum_name, mask in strata_for_paired.items():
            # Aggregate per-seed deltas first, then bootstrap the aggregate
            # OR: bootstrap each seed and average per-seed CIs.
            # We do the latter - closer to E18's reportable shape - then
            # also report the seed-averaged paired-bootstrap CI as the
            # primary number.
            seed_deltas = []
            seed_los = []
            seed_his = []
            n_subj_used = None
            for s in SEEDS:
                bb = scores_per_seed_variant[(s, "backbone_only")]
                v  = scores_per_seed_variant[(s, variant)]
                lo, hi, mn, sd, n_subj_used = paired_subject_bootstrap_delta_auc(
                    labels, bb, v, subjects, n_boot=N_BOOT, alpha=ALPHA,
                    seed=s * 7919, mask=mask)
                seed_deltas.append(mn)
                seed_los.append(lo)
                seed_his.append(hi)
            seed_mean_delta = float(np.mean(seed_deltas))
            mean_lo = float(np.mean(seed_los))
            mean_hi = float(np.mean(seed_his))
            excludes_zero = (mean_lo > 0) or (mean_hi < 0)
            paired_rows.append({
                "variant": variant, "stratum": stratum_name,
                "n_subjects": n_subj_used,
                "mean_delta_auc": seed_mean_delta,
                "ci_lo": mean_lo, "ci_hi": mean_hi,
                "ci_excludes_zero": excludes_zero,
            })
            ex = "YES" if excludes_zero else "no"
            print(f"{variant:<18s} {stratum_name:<20s} {n_subj_used:>7d} | "
                  f"{seed_mean_delta:>+9.4f} {mean_lo:>+9.4f} {mean_hi:>+9.4f} | "
                  f"{ex:>15s}")
        print()

    with open(out_dir / "e20_paired_cis.csv", "w") as f:
        f.write("variant,stratum,n_subjects,mean_delta_auc,ci_lo,ci_hi,ci_excludes_zero\n")
        for r in paired_rows:
            f.write(f"{r['variant']},{r['stratum']},{r['n_subjects']},"
                    f"{r['mean_delta_auc']:.6f},{r['ci_lo']:.6f},"
                    f"{r['ci_hi']:.6f},{r['ci_excludes_zero']}\n")
    print(f"[e20] wrote {out_dir/'e20_paired_cis.csv'}")

    # ---------------------------------------------------------
    # PART 2 - Reliability-conditioned physiology test
    # (already covered in PART 1's rppg_snr_high_Q and blink_int_high_Q rows;
    #  add an explicit dedicated section in the markdown)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # PART 3 - Threshold-level rescue/regression curves
    # ---------------------------------------------------------
    print(f"\n[e20] PART 3 - Threshold-level rescue/regression at FPR thresholds")
    threshold_rows = []
    strata_for_threshold = {
        "ALL":                np.ones(n, dtype=bool),
        "rppg_snr_high_Q":    rppg_snr >= snr_q3,
        "blink_int_high_Q":   blink_int >= blink_q3,
        "blink_int_low_Q":    blink_int <= blink_q1,
        "real":               labels == 0,
        "fake":               labels == 1,
    }
    print(f"{'variant':<18s} {'stratum':<20s} {'FPR':>5s} | "
          f"{'rescue':>8s} {'regr':>8s} {'net':>8s} {'rescue%':>9s}")
    print("-" * 80)

    for variant in VARIANTS:
        for stratum_name, mask in strata_for_threshold.items():
            for fpr_target in FPR_LEVELS:
                rescues = []
                regressions = []
                for s in SEEDS:
                    bb = scores_per_seed_variant[(s, "backbone_only")]
                    v  = scores_per_seed_variant[(s, variant)]
                    threshold_fn = lambda y, sc, fpr=fpr_target: fpr_threshold(y, sc, fpr)
                    rescue_n, regr_n = per_seed_flip_counts(
                        labels, bb, v, threshold_fn, stratum_mask=mask)
                    rescues.append(rescue_n)
                    regressions.append(regr_n)
                rescue_mean = float(np.mean(rescues))
                regression_mean = float(np.mean(regressions))
                net = rescue_mean - regression_mean
                denom = rescue_mean + regression_mean
                rescue_pct = (100.0 * rescue_mean / denom) if denom > 0 else float("nan")
                threshold_rows.append({
                    "variant": variant, "stratum": stratum_name,
                    "fpr_target": fpr_target,
                    "rescue_mean": rescue_mean,
                    "regression_mean": regression_mean,
                    "net_mean": net,
                    "rescue_pct": rescue_pct,
                })
                print(f"{variant:<18s} {stratum_name:<20s} "
                      f"{fpr_target*100:>4.0f}% | "
                      f"{rescue_mean:>8.1f} {regression_mean:>8.1f} "
                      f"{net:>+8.1f} {rescue_pct:>8.1f}%")
        print()

    with open(out_dir / "e20_threshold_curves.csv", "w") as f:
        f.write("variant,stratum,fpr_target,rescue_mean,regression_mean,"
                "net_mean,rescue_pct\n")
        for r in threshold_rows:
            f.write(f"{r['variant']},{r['stratum']},{r['fpr_target']},"
                    f"{r['rescue_mean']:.2f},{r['regression_mean']:.2f},"
                    f"{r['net_mean']:.2f},{r['rescue_pct']:.2f}\n")
    print(f"[e20] wrote {out_dir/'e20_threshold_curves.csv'}")

    # ---------------------------------------------------------
    # Summary JSON + markdown
    # ---------------------------------------------------------
    with open(out_dir / "e20_summary.json", "w") as f:
        json.dump({
            "n_clips": int(n),
            "n_subjects": int(len(np.unique(subjects))),
            "n_boot": N_BOOT,
            "alpha": ALPHA,
            "seeds": SEEDS,
            "variants": VARIANTS,
            "fpr_levels": FPR_LEVELS,
            "paired_cis": paired_rows,
            "threshold_curves": threshold_rows,
        }, f, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating)
                        else int(o) if isinstance(o, np.integer)
                        else bool(o) if isinstance(o, np.bool_)
                        else str(o))
    print(f"[e20] wrote {out_dir/'e20_summary.json'}")

    # Markdown summary
    with open(out_dir / "e20_findings.md", "w", encoding="utf-8") as f:
        f.write("# E20 - Paired bootstrap + reliability test + threshold curves\n\n")
        f.write(f"**Date:** 2026-05-10  \n")
        f.write(f"**Bootstrap iterations:** {N_BOOT}; alpha = {ALPHA}\n\n")
        f.write("## 1. Paired subject-cluster bootstrap on delta AUC\n\n")
        f.write("CIs that exclude zero indicate statistically reliable "
                "differences at the subject-cluster level (not just clip-"
                "level). 5-seed mean +/- paired-bootstrap CI.\n\n")
        for variant in VARIANTS:
            f.write(f"### {variant} vs backbone-only\n\n")
            f.write("| Stratum | n_subj | mean delta AUC | 95% CI | excludes 0? |\n")
            f.write("|---|---|---|---|---|\n")
            for r in paired_rows:
                if r["variant"] != variant:
                    continue
                ex = "**YES**" if r["ci_excludes_zero"] else "no"
                f.write(f"| {r['stratum']} | {r['n_subjects']} | "
                        f"{r['mean_delta_auc']:+.4f} | "
                        f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {ex} |\n")
            f.write("\n")
        f.write("\n## 2. Reliability-conditioned physiology test\n\n")
        f.write("From part 1 (rppg_snr_high_Q stratum, n=440 / "
                f"{int(np.sum(strata_for_paired['rppg_snr_high_Q']))} clips):\n\n")
        for variant in VARIANTS:
            r = next((x for x in paired_rows
                      if x["variant"] == variant and
                         x["stratum"] == "rppg_snr_high_Q"), None)
            if r:
                ex = "**YES**" if r["ci_excludes_zero"] else "no"
                f.write(f"- {variant} on high-SNR rPPG quartile: "
                        f"delta = {r['mean_delta_auc']:+.4f}, "
                        f"CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                        f"excludes 0: {ex}\n")
        f.write("\nFor the reliability hypothesis to be confirmed, the "
                "high-SNR row for +rPPG (or any variant) would need to "
                "show a positive delta with a CI that excludes zero. If no "
                "variant achieves this, the redundancy argument is "
                "strengthened.\n")
        f.write("\n## 3. Threshold-level rescue/regression curves\n\n")
        f.write("Per-seed mean rescue/regression counts at FPR in {1, 5, 10}%.\n\n")
        for variant in VARIANTS:
            f.write(f"### {variant}\n\n")
            f.write("| Stratum | FPR=1% | FPR=5% | FPR=10% |\n")
            f.write("|---|---|---|---|\n")
            for stratum_name in strata_for_threshold.keys():
                row_cells = []
                for fpr in FPR_LEVELS:
                    r = next((x for x in threshold_rows
                              if x["variant"] == variant and
                                 x["stratum"] == stratum_name and
                                 abs(x["fpr_target"] - fpr) < 1e-9), None)
                    if r:
                        row_cells.append(
                            f"net={r['net_mean']:+.1f} (rescue% {r['rescue_pct']:.0f})"
                            if not np.isnan(r["rescue_pct"]) else
                            f"net={r['net_mean']:+.1f}")
                    else:
                        row_cells.append("n/a")
                f.write(f"| {stratum_name} | {row_cells[0]} | "
                        f"{row_cells[1]} | {row_cells[2]} |\n")
            f.write("\n")
    print(f"[e20] wrote {out_dir/'e20_findings.md'}")
    print(f"\n[e20] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality_csv", required=True)
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
