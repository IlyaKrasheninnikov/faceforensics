"""
E19 — Fusion-fairness sanity check.

Addresses professor v7 review point 4 (verbatim):
   "Add a small fusion-fairness sanity check: train-only z-scoring or
    a regularized logistic probe for backbone-only versus
    backbone+rPPG, backbone+blink, and full fusion. If this cannot
    be done, explicitly state that the conclusion applies to the
    tested raw-concat linear probe."

The concern: the v6/v7 strict-LODO probe uses raw concatenation of the
1024-d CLIP backbone with a 12-d rPPG vector and 16-d blink vector.
Without normalization, the CLIP component has dramatically larger
norm than the physiology components, so the linear probe is dominated
by CLIP. Conversely, if the physiology vectors have outlier scales
they could distort training in unexpected ways. A "fair" probe
would either (a) z-score each block separately on the training set,
or (b) use L2 regularization to keep the per-feature contribution
balanced.

This script reruns the strict-LODO probe with two fairness variants:
  - z-scored: train-set per-feature z-score applied to backbone, rppg,
    blink blocks separately, then concatenated (raw-concat baseline
    for comparison)
  - L2-regularized: scikit-learn LogisticRegression with C=1.0, no
    z-scoring (raw scales)

Both are compared against the raw-concat-linear-AdamW probe used
throughout E14.

The key question: does the v7 finding "physiology variants are
statistically inferior to backbone-only under strict LODO" hold under
the fair-probe comparison? If yes, the conclusion is robust to probe
choice. If no, the conclusion is restricted to the raw-concat probe
and the v7 framing must say so.

ETA on Kaggle T4: ~5 min for one backbone (CLIP) × 5 seeds ×
4 variants × 3 probe families = 60 probe trainings.

Outputs (in --out_dir):
  e19_fairness_aggregate.csv  - mean ± std AUC per (variant × probe family)
  e19_per_seed.csv            - per-seed AUCs
  e19_summary.json
  e19_findings.md             - markdown summary

Usage on Kaggle:
    python w10_stats/e19_fusion_fairness.py \\
        --cache_dir   "$CACHE_CLIP" \\
        --celebdf_root "$CDF" \\
        --out_dir     "$OUT_E19"

(Same paths as KAGGLE_RUN_E14_STRICT_LODO.md — uses the same
cached features; no new extraction.)
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import (
    roc_auc, eer, average_precision, tpr_at_fpr,
    train_linear_probe, predict, identity_split_ff,
    VARIANTS, make_features,
)
from identity_split_sanity import scan_celebdf_with_subject, subject_aware_split


SEEDS = [0, 1, 42, 1337, 2024]
PROBE_FAMILIES = ["raw_concat_adamw", "zscored_adamw", "l2_logistic"]


def fit_predict_l2_logistic(X_tr, y_tr, X_te, seed=0, C=1.0):
    """Closed-form-ish L2 logistic regression via sklearn (no early stopping,
    no probe-init noise). C=1.0 is the standard default."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed,
                             solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def zscore_blocks(X_tr, X_vl, X_te, block_dims):
    """z-score each block separately using train-set mean/std.

    block_dims: list of (start, end) column ranges in the concatenated X.
    """
    Xt = X_tr.copy().astype(np.float32)
    Xv = X_vl.copy().astype(np.float32)
    Xe = X_te.copy().astype(np.float32)
    for (s, e) in block_dims:
        mu = Xt[:, s:e].mean(axis=0, keepdims=True)
        sd = Xt[:, s:e].std(axis=0, keepdims=True) + 1e-7
        Xt[:, s:e] = (Xt[:, s:e] - mu) / sd
        Xv[:, s:e] = (Xv[:, s:e] - mu) / sd
        Xe[:, s:e] = (Xe[:, s:e] - mu) / sd
    return Xt, Xv, Xe


def get_block_dims(variant, bb_dim, rppg_dim, blink_dim):
    """Return list of (start, end) column ranges for the variant."""
    blocks = [(0, bb_dim)]
    cursor = bb_dim
    if variant in ("backbone+rppg", "full_fusion"):
        blocks.append((cursor, cursor + rppg_dim))
        cursor += rppg_dim
    if variant in ("backbone+blink", "full_fusion"):
        blocks.append((cursor, cursor + blink_dim))
    return blocks


def main(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[e19] device={device}  cache_dir={args.cache_dir}")

    cache_dir = Path(args.cache_dir)
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = cache_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"[e19] loaded {tag}: n={len(caches[tag]['labels'])} "
                  f"bb_dim={caches[tag]['backbone'].shape[1]} "
                  f"rppg_dim={caches[tag]['rppg'].shape[1]} "
                  f"blink_dim={caches[tag]['blink'].shape[1]}")

    bb_dim = caches["ff"]["backbone"].shape[1]
    rppg_dim = caches["ff"]["rppg"].shape[1]
    blink_dim = caches["ff"]["blink"].shape[1]

    # Strict LODO test_celebdf: train on FF + DFDC, val from FF identity-aware
    # (matches lodo_probe_strict.py for the test_celebdf config)
    ff_tr, ff_vl, ff_te = identity_split_ff(caches["ff"], seed=42)
    cdf_rows = scan_celebdf_with_subject(args.celebdf_root)
    subject_ids = np.array([r[0] for r in cdf_rows])
    cdf_labels_check = np.array([r[1] for r in cdf_rows])
    if not np.array_equal(caches["celebdf"]["labels"], cdf_labels_check):
        raise RuntimeError("CelebDF cache order mismatch")
    cd_tr_subj, cd_te_subj, _, _ = subject_aware_split(subject_ids, seed=42)

    # Build train pool (FF tr+te + DFDC all), val (FF val), test (CelebDF subj-te)
    train_idx_ff = np.concatenate([ff_tr, ff_te])
    df_n = len(caches["dfdc"]["labels"])
    bb_tr_full = np.concatenate([
        caches["ff"]["backbone"][train_idx_ff],
        caches["dfdc"]["backbone"][np.arange(df_n)],
    ])
    rppg_tr_full = np.concatenate([
        caches["ff"]["rppg"][train_idx_ff],
        caches["dfdc"]["rppg"][np.arange(df_n)],
    ])
    blink_tr_full = np.concatenate([
        caches["ff"]["blink"][train_idx_ff],
        caches["dfdc"]["blink"][np.arange(df_n)],
    ])
    y_tr_full = np.concatenate([
        caches["ff"]["labels"][train_idx_ff],
        caches["dfdc"]["labels"][np.arange(df_n)],
    ])

    bb_vl_full   = caches["ff"]["backbone"][ff_vl]
    rppg_vl_full = caches["ff"]["rppg"][ff_vl]
    blink_vl_full = caches["ff"]["blink"][ff_vl]
    y_vl_full = caches["ff"]["labels"][ff_vl]

    bb_te_full   = caches["celebdf"]["backbone"][cd_te_subj]
    rppg_te_full = caches["celebdf"]["rppg"][cd_te_subj]
    blink_te_full = caches["celebdf"]["blink"][cd_te_subj]
    y_te = caches["celebdf"]["labels"][cd_te_subj]

    print(f"\n[e19] strict LODO test_celebdf: n_train={len(y_tr_full)} "
          f"n_val={len(y_vl_full)} n_test={len(y_te)}")

    # Quick sanity on per-block norms
    print(f"\n[e19] per-block L2 norm on training set (mean):")
    print(f"  backbone (1024-d): {np.linalg.norm(bb_tr_full, axis=1).mean():.3f}")
    print(f"  rppg (12-d):       {np.linalg.norm(rppg_tr_full, axis=1).mean():.3f}")
    print(f"  blink (16-d):      {np.linalg.norm(blink_tr_full, axis=1).mean():.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    rows = []
    print(f"\n[e19] Running 5 seeds × 4 variants × 3 probe families = 60 probes")

    for seed in SEEDS:
        print(f"\n[e19] seed={seed}")
        for variant in VARIANTS:
            X_tr_raw = make_features(bb_tr_full, rppg_tr_full, blink_tr_full, variant)
            X_vl_raw = make_features(bb_vl_full, rppg_vl_full, blink_vl_full, variant)
            X_te_raw = make_features(bb_te_full, rppg_te_full, blink_te_full, variant)

            for family in PROBE_FAMILIES:
                if family == "raw_concat_adamw":
                    probe = train_linear_probe(X_tr_raw, y_tr_full, X_vl_raw,
                                               y_vl_full, device, epochs=20,
                                               lr=1e-3, bs=256, seed=seed)
                    scores = predict(probe, X_te_raw, device)

                elif family == "zscored_adamw":
                    blocks = get_block_dims(variant, bb_dim, rppg_dim, blink_dim)
                    Xt, Xv, Xe = zscore_blocks(X_tr_raw, X_vl_raw, X_te_raw, blocks)
                    probe = train_linear_probe(Xt, y_tr_full, Xv, y_vl_full,
                                               device, epochs=20, lr=1e-3,
                                               bs=256, seed=seed)
                    scores = predict(probe, Xe, device)

                elif family == "l2_logistic":
                    blocks = get_block_dims(variant, bb_dim, rppg_dim, blink_dim)
                    Xt, _, Xe = zscore_blocks(X_tr_raw, X_vl_raw, X_te_raw, blocks)
                    scores = fit_predict_l2_logistic(Xt, y_tr_full, Xe,
                                                     seed=seed, C=1.0)

                auc = roc_auc(y_te, scores)
                row = {
                    "seed": seed, "variant": variant, "family": family,
                    "auc": auc,
                    "ap": average_precision(y_te, scores),
                    "eer": eer(y_te, scores),
                    "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                }
                rows.append(row)
                np.savez(out_dir / "scores" /
                         f"seed{seed}_{variant}_{family}.npz",
                         scores=scores, labels=y_te)
                print(f"  {variant:<18s} {family:<22s}  AUC={auc:.4f}  "
                      f"TPR@5={row['tpr5']:.3f}")

    # Per-seed CSV
    head = ["seed", "variant", "family", "auc", "ap", "eer", "tpr5"]
    with open(out_dir / "e19_per_seed.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float)
                             else str(r[k]) for k in head) + "\n")
    print(f"\n[e19] wrote {out_dir/'e19_per_seed.csv'}")

    # Aggregate
    print(f"\n[e19] 5-seed aggregate AUC by (variant × family):")
    agg = []
    print(f"{'variant':<18s} | {'family':<22s} | {'AUC mean':>10s} {'AUC std':>10s} "
          f"{'TPR@5 mean':>12s}")
    for variant in VARIANTS:
        for family in PROBE_FAMILIES:
            bucket = [r for r in rows if r["variant"] == variant and
                                          r["family"] == family]
            aucs = np.array([b["auc"] for b in bucket])
            tpr5s = np.array([b["tpr5"] for b in bucket])
            agg.append({
                "variant": variant, "family": family,
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std(ddof=1)),
                "tpr5_mean": float(tpr5s.mean()),
                "tpr5_std": float(tpr5s.std(ddof=1)),
            })
            print(f"{variant:<18s} | {family:<22s} | "
                  f"{aucs.mean():>10.4f} {aucs.std(ddof=1):>10.4f} "
                  f"{tpr5s.mean():>12.4f}")
        print()

    head_agg = ["variant", "family", "auc_mean", "auc_std", "tpr5_mean", "tpr5_std"]
    with open(out_dir / "e19_fairness_aggregate.csv", "w") as f:
        f.write(",".join(head_agg) + "\n")
        for r in agg:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float)
                             else str(r[k]) for k in head_agg) + "\n")
    print(f"[e19] wrote {out_dir/'e19_fairness_aggregate.csv'}")

    with open(out_dir / "e19_summary.json", "w") as f:
        json.dump({
            "seeds": SEEDS, "variants": list(VARIANTS),
            "families": PROBE_FAMILIES,
            "n_train": int(len(y_tr_full)),
            "n_val": int(len(y_vl_full)),
            "n_test": int(len(y_te)),
            "block_norms_train": {
                "backbone": float(np.linalg.norm(bb_tr_full, axis=1).mean()),
                "rppg":     float(np.linalg.norm(rppg_tr_full, axis=1).mean()),
                "blink":    float(np.linalg.norm(blink_tr_full, axis=1).mean()),
            },
            "per_seed": rows,
            "aggregate": agg,
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                              else int(o) if isinstance(o, np.integer)
                              else str(o))
    print(f"[e19] wrote {out_dir/'e19_summary.json'}")

    # Markdown findings
    with open(out_dir / "e19_findings.md", "w", encoding="utf-8") as f:
        f.write("# E19 — Fusion-fairness sanity check\n\n")
        f.write("**Date:** 2026-05-10  \n")
        f.write(f"**Setting:** strict LODO CelebDF (n=1758), CLIP backbone\n\n")
        f.write("Three probe families compared:\n")
        f.write("- `raw_concat_adamw`: linear AdamW probe on raw concatenated "
                "features (the v6/v7 default)\n")
        f.write("- `zscored_adamw`: same probe, but each block "
                "(backbone, rppg, blink) is z-scored on the training set\n")
        f.write("- `l2_logistic`: scikit-learn LogisticRegression(C=1.0) "
                "on z-scored features (no probe-init noise)\n\n")
        f.write("## 5-seed aggregate AUC by (variant × family)\n\n")
        f.write("| variant | family | AUC mean ± std | TPR@5% |\n")
        f.write("|---|---|---|---|\n")
        for r in agg:
            f.write(f"| {r['variant']} | {r['family']} | "
                    f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
                    f"{r['tpr5_mean']:.4f} |\n")

        f.write("\n## Decisive question — does the physiology-variant "
                "ordering hold under fair probes?\n\n")
        for family in PROBE_FAMILIES:
            f.write(f"### {family}\n\n")
            for variant in VARIANTS:
                r = next(x for x in agg if x["variant"] == variant and
                                            x["family"] == family)
                f.write(f"- {variant}: {r['auc_mean']:.4f} ± {r['auc_std']:.4f}\n")
            bb = next(x for x in agg if x["variant"] == "backbone_only" and
                                         x["family"] == family)
            fu = next(x for x in agg if x["variant"] == "full_fusion" and
                                         x["family"] == family)
            delta = fu["auc_mean"] - bb["auc_mean"]
            f.write(f"\n  Δ (full_fusion − backbone_only) = {delta:+.4f}\n\n")
        f.write("\nIf Δ stays in the [-0.005, +0.005] band for all three "
                "families, the v7 conclusion is robust to probe choice. "
                "If the z-scored or L2 family flips Δ to positive, the v7 "
                "framing must be restricted to the raw-concat AdamW probe.\n")

    print(f"[e19] wrote {out_dir/'e19_findings.md'}")
    print(f"\n[e19] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--celebdf_root", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
