"""
E18 — Subject-cluster bootstrap for the strict-LODO CelebDF headline.

Addresses professor v7 review point 3:
  "Add subject-cluster bootstrap for the CelebDF headline CIs.
   If this is not done, label the current CIs as clip-level and
   reduce the strength of the statistical claims."

The 1758 strict-LODO CelebDF test clips come from 72 unique subjects.
Clip-level bootstrap (the standard approach) treats every clip as
independent — but two clips of the same subject share identity-level
nuisance variation. Subject-cluster bootstrap resamples whole
subjects (with replacement), which gives a CI that reflects
between-subject variance rather than within-subject correlation.

For the strict-LODO CelebDF n=1758 partition (72 subjects, 7:1 fake/
real imbalance), the subject-cluster CI is expected to be wider than
the clip-level CI because the effective sample size is n_subjects
(72), not n_clips (1758).

Outputs (in --out_dir):
  e18_bootstrap_cis.csv   - clip-level vs subject-cluster CIs per (seed × variant)
  e18_summary.json        - same plus 5-seed aggregates
  e18_findings.md         - markdown summary

Usage:
    python p3_physio/w10_stats/e18_subject_cluster_bootstrap.py \\
        --quality_csv  p3_physio/outputs_and_cfgs/e16_physio_quality_clip/quality_metrics.csv \\
        --scores_dir   p3_physio/outputs_and_cfgs/strict_lodo_bundle/e14_lodo_strict_clip/scores \\
        --out_dir      p3_physio/outputs_and_cfgs/e18_subject_bootstrap
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SEEDS = [0, 1, 42, 1337, 2024]
VARIANTS = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]


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
    return float(np.trapezoid(tp, fp)) if hasattr(np, 'trapezoid') else float(np.trapz(tp, fp))


def clip_bootstrap_ci(labels, scores, n_boot=1000, alpha=0.05, seed=0):
    """Standard clip-level bootstrap: resample clip indices with replacement."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aucs[b] = roc_auc(labels[idx], scores[idx])
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return lo, hi, float(aucs.mean()), float(aucs.std(ddof=1))


def subject_cluster_bootstrap_ci(labels, scores, subject_ids,
                                 n_boot=1000, alpha=0.05, seed=0):
    """Subject-cluster bootstrap: resample whole subjects with replacement.

    For each bootstrap iteration:
    1. Sample n_subjects subjects (with replacement) from the unique pool.
    2. Concatenate ALL clips of each sampled subject.
    3. Compute AUC on the concatenated set.
    """
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subject_ids)
    unique_subjects = np.unique(subjects)
    n_subj = len(unique_subjects)
    aucs = np.empty(n_boot)
    # Pre-build subject -> clip-index mapping
    subj_to_clips = {s: np.where(subjects == s)[0] for s in unique_subjects}
    for b in range(n_boot):
        chosen = rng.choice(unique_subjects, size=n_subj, replace=True)
        idx_list = [subj_to_clips[s] for s in chosen]
        idx = np.concatenate(idx_list)
        aucs[b] = roc_auc(labels[idx], scores[idx])
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return lo, hi, float(aucs.mean()), float(aucs.std(ddof=1))


def main(args):
    # Load quality CSV (gives clip_idx, subject_id, label per test clip,
    # in the order used by E14 strict-LODO score files)
    quality = []
    with open(args.quality_csv) as f:
        for r in csv.DictReader(f):
            quality.append({
                "clip_idx": int(r["clip_idx"]),
                "subject_id": r["subject_id"],
                "label": int(float(r["label"])),
            })
    n = len(quality)
    subjects = np.array([r["subject_id"] for r in quality])
    labels = np.array([r["label"] for r in quality])
    unique_subjects = np.unique(subjects)
    print(f"[e18] n_clips={n}  n_subjects={len(unique_subjects)}")
    fake_subjects = np.unique(subjects[labels == 1])
    real_subjects = np.unique(subjects[labels == 0])
    print(f"[e18] subjects: {len(fake_subjects)} with fake clips, "
          f"{len(real_subjects)} with real clips, "
          f"{len(np.intersect1d(fake_subjects, real_subjects))} with both")

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per (seed × variant): compute clip-level vs subject-cluster CIs
    rows = []
    print(f"\n[e18] Per-seed, per-variant CIs (1000-bootstrap, 95% CI):")
    print(f"{'seed':>5s} {'variant':<18s} | {'AUC':>8s} | "
          f"{'clip CI':>20s} {'clip width':>10s} | "
          f"{'subj CI':>20s} {'subj width':>10s} {'inflation':>10s}")
    for seed in SEEDS:
        for variant in VARIANTS:
            p = scores_dir / f"test_celebdf_s{seed}_{variant}.npz"
            d = np.load(str(p))
            scores = d["scores"]
            labels_check = d["labels"].astype(int)
            assert np.array_equal(labels_check, labels), \
                f"Label order mismatch on seed {seed} variant {variant}"
            auc = roc_auc(labels, scores)
            clip_lo, clip_hi, _, clip_std = clip_bootstrap_ci(
                labels, scores, n_boot=1000, seed=seed * 1000)
            subj_lo, subj_hi, _, subj_std = subject_cluster_bootstrap_ci(
                labels, scores, subjects, n_boot=1000, seed=seed * 1000)
            clip_width = clip_hi - clip_lo
            subj_width = subj_hi - subj_lo
            inflation = subj_width / clip_width if clip_width > 0 else float("nan")
            rows.append({
                "seed": seed, "variant": variant, "auc": auc,
                "clip_ci_lo": clip_lo, "clip_ci_hi": clip_hi,
                "clip_ci_width": clip_width, "clip_boot_std": clip_std,
                "subj_ci_lo": subj_lo, "subj_ci_hi": subj_hi,
                "subj_ci_width": subj_width, "subj_boot_std": subj_std,
                "inflation_ratio": inflation,
            })
            print(f"{seed:>5d} {variant:<18s} | {auc:>8.4f} | "
                  f"[{clip_lo:.4f}, {clip_hi:.4f}] {clip_width:>10.4f} | "
                  f"[{subj_lo:.4f}, {subj_hi:.4f}] {subj_width:>10.4f} "
                  f"{inflation:>9.2f}x")
        print()

    # 5-seed aggregate per variant
    print(f"\n[e18] 5-seed aggregate (clip vs subject CI widths):")
    print(f"{'variant':<18s} | {'mean AUC':>10s} | "
          f"{'mean clip width':>16s} {'mean subj width':>17s} {'mean inflation':>15s}")
    agg = []
    for variant in VARIANTS:
        bucket = [r for r in rows if r["variant"] == variant]
        mean_auc = float(np.mean([b["auc"] for b in bucket]))
        mean_clip_w = float(np.mean([b["clip_ci_width"] for b in bucket]))
        mean_subj_w = float(np.mean([b["subj_ci_width"] for b in bucket]))
        mean_infl = float(np.mean([b["inflation_ratio"] for b in bucket]))
        # Use seed 0 CIs as the reportable headline CI for the variant
        seed0 = next(b for b in bucket if b["seed"] == 0)
        agg.append({
            "variant": variant,
            "mean_auc": mean_auc,
            "mean_clip_ci_width": mean_clip_w,
            "mean_subj_ci_width": mean_subj_w,
            "mean_inflation_ratio": mean_infl,
            "seed0_clip_ci": [seed0["clip_ci_lo"], seed0["clip_ci_hi"]],
            "seed0_subj_ci": [seed0["subj_ci_lo"], seed0["subj_ci_hi"]],
        })
        print(f"{variant:<18s} | {mean_auc:>10.4f} | "
              f"{mean_clip_w:>16.4f} {mean_subj_w:>17.4f} {mean_infl:>14.2f}x")

    # CSV
    head = ["seed", "variant", "auc",
            "clip_ci_lo", "clip_ci_hi", "clip_ci_width", "clip_boot_std",
            "subj_ci_lo", "subj_ci_hi", "subj_ci_width", "subj_boot_std",
            "inflation_ratio"]
    with open(out_dir / "e18_bootstrap_cis.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float)
                             else str(r[k]) for k in head) + "\n")
    print(f"\n[e18] wrote {out_dir/'e18_bootstrap_cis.csv'}")

    # JSON summary
    with open(out_dir / "e18_summary.json", "w") as f:
        json.dump({
            "n_clips": int(n),
            "n_subjects": int(len(unique_subjects)),
            "n_subjects_fake": int(len(fake_subjects)),
            "n_subjects_real": int(len(real_subjects)),
            "n_subjects_both": int(len(np.intersect1d(fake_subjects, real_subjects))),
            "n_boot": 1000,
            "alpha": 0.05,
            "seeds": SEEDS,
            "variants": VARIANTS,
            "per_seed_variant": rows,
            "aggregate_per_variant": agg,
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                              else int(o) if isinstance(o, np.integer)
                              else str(o))
    print(f"[e18] wrote {out_dir/'e18_summary.json'}")

    # Markdown findings
    with open(out_dir / "e18_findings.md", "w", encoding="utf-8") as f:
        f.write("# E18 — Subject-cluster bootstrap for strict-LODO CelebDF\n\n")
        f.write(f"**Date:** 2026-05-10  \n")
        f.write(f"**n_clips:** {n}  \n")
        f.write(f"**n_subjects:** {len(unique_subjects)}\n\n")

        f.write("## 5-seed aggregate (clip-level vs subject-cluster CI)\n\n")
        f.write("| Variant | mean AUC | mean clip CI width | mean subject CI width | inflation |\n")
        f.write("|---|---|---|---|---|\n")
        for r in agg:
            f.write(f"| {r['variant']} | {r['mean_auc']:.4f} | "
                    f"{r['mean_clip_ci_width']:.4f} | {r['mean_subj_ci_width']:.4f} | "
                    f"{r['mean_inflation_ratio']:.2f}x |\n")

        f.write("\n## Headline CI for backbone-only (seed 0)\n\n")
        bb = next(r for r in agg if r["variant"] == "backbone_only")
        f.write(f"- Clip-level 95% CI: [{bb['seed0_clip_ci'][0]:.4f}, "
                f"{bb['seed0_clip_ci'][1]:.4f}]\n")
        f.write(f"- Subject-cluster 95% CI: [{bb['seed0_subj_ci'][0]:.4f}, "
                f"{bb['seed0_subj_ci'][1]:.4f}]\n\n")

        infl = bb["mean_inflation_ratio"]
        if infl > 1.5:
            f.write("**The subject-cluster CI is materially wider than the "
                    "clip-level CI.** This means the clip-level CI we have "
                    "been quoting underestimates the true sampling "
                    "uncertainty. The headline CelebDF strict-LODO AUC is "
                    "still 0.749, but the 95% CI should be quoted as the "
                    "subject-cluster width.\n")
        elif infl > 1.1:
            f.write("**The subject-cluster CI is moderately wider** "
                    f"({infl:.2f}x the clip-level width). The clip-level "
                    "CI under-states uncertainty by ~"
                    f"{(infl-1)*100:.0f}%, which is small but should be "
                    "disclosed.\n")
        else:
            f.write("**The subject-cluster CI is approximately equal to "
                    "the clip-level CI** "
                    f"(inflation = {infl:.2f}x). Within-subject correlation "
                    "is small enough that clip-level CIs are an acceptable "
                    "approximation.\n")

    print(f"[e18] wrote {out_dir/'e18_findings.md'}")
    print(f"\n[e18] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality_csv", required=True)
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
