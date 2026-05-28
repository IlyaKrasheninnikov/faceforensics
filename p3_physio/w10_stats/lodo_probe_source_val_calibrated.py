"""Source-validation-calibrated strict-LODO operating points.

The paper's operating-point analysis (Table 4; the threshold-level
rescue/regression cells) uses ORACLE-FPR thresholds: thresholds chosen
on the test set's own score distribution to hit the nominal FPR
exactly. The paper labels those results diagnostic, not
deployment-grade, precisely because the threshold sees the test set.

This script produces the SOURCE-VALIDATION-CALIBRATED version that the
paper defers to as future work. It reruns the strict-LODO CLIP probe
and, in addition to the test scores, SAVES the held-out source
validation-pool scores. The FPR threshold is then fit on the
validation pool (FF+DFDC source, which the probe legitimately sees at
training time) and applied ONCE to the CelebDF test partition. This is
a deployment-realistic operating point: the threshold never sees the
target distribution.

It is otherwise identical to lodo_probe_strict.py (same partition
logic, same VARIANTS, same training hyperparameters, same runtime
subject-disjointness assertion), so the AUC numbers reproduce the
bundled strict-LODO AUCs; only the threshold calibration source
changes.

Outputs (in --out_dir):
  scores/test_celebdf_s{seed}_{variant}.npz      (test scores+labels)
  scores/val_s{seed}_{variant}.npz               (val scores+labels)   <-- NEW
  source_val_operating_points.csv
      per (variant, seed, fpr): threshold fit on val, TPR achieved on
      test, realised FPR on test (drift from nominal indicates
      source->target shift).
  source_val_summary.md

Run on Kaggle with the SAME dataset paths as the E14 strict-LODO run
(see p3_physio/w10_stats/KAGGLE_RUN_E14_STRICT_LODO.md). CPU works; a
T4 makes it a few minutes. The script needs the repo helpers
lodo_probe_strict.py and multiseed_and_stats.py on the import path
(they sit in p3_physio/w10_stats/ in the main project, or scripts/ in
the artefact bundle):

    CACHES_ROOT=/kaggle/input/datasets/goodboyxdd/feat-caches-b4-clip-dinov2
    CDF=/kaggle/input/datasets/diwakarsehgal/celebdfv2/crop
    python scripts/lodo_probe_source_val_calibrated.py \\
        --cache_dir    "$CACHES_ROOT/feat_cache_clip" \\
        --out_dir      artifacts/csv/source_val_calibrated_clip \\
        --celebdf_root "$CDF" \\
        --seeds 0 1 42 1337 2024

Then locally:
    python scripts/source_val_summary.py   # builds the paper table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

# Reuse the EXACT partition logic + helpers from the strict probe so
# the calibrated run is identical except for threshold sourcing.
from lodo_probe_strict import (  # noqa: E402
    get_train_val_test_indices_strict, stack_indexed, LODO_CONFIGS,
)
from multiseed_and_stats import (  # noqa: E402
    roc_auc, tpr_at_fpr, train_linear_probe, predict,
    VARIANTS, make_features,
)

FPR_LEVELS = [0.01, 0.05, 0.10]


def fpr_threshold_from_val(val_labels, val_scores, target_fpr):
    """Threshold achieving target FPR on the VALIDATION negatives."""
    neg = np.asarray(val_scores)[np.asarray(val_labels) == 0]
    if len(neg) == 0:
        return float(np.max(val_scores))
    # threshold = (1 - fpr) quantile of validation negative scores
    return float(np.quantile(neg, 1.0 - target_fpr))


def tpr_fpr_at_threshold(labels, scores, thr):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    tpr = float(np.mean(pos >= thr)) if len(pos) else float("nan")
    fpr = float(np.mean(neg >= thr)) if len(neg) else float("nan")
    return tpr, fpr


def main(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[source-val] device={device} cache_dir={args.cache_dir}")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)

    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = cache_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"[source-val] loaded {tag}: n={len(caches[tag]['labels'])}")
    if not {"ff", "celebdf", "dfdc"} <= set(caches):
        print("[source-val] requires ff, celebdf, dfdc caches; abort")
        return

    # Only the CelebDF-as-target config is needed for the deployment-
    # realistic operating point on the paper's primary partition.
    config_name, train_datasets, test_dataset = LODO_CONFIGS[0]
    assert test_dataset == "celebdf"

    idx = get_train_val_test_indices_strict(
        caches, train_datasets, test_dataset, args)

    # Runtime subject-disjointness contract (same as strict probe).
    assert len(idx["celebdf"]["train"]) == 0, "CelebDF leaked into train"
    assert len(idx["celebdf"]["val"]) == 0, "CelebDF leaked into val"
    print("[source-val] [audit] target 'celebdf' held out from train AND val OK")

    bb_tr, rppg_tr, blink_tr, y_tr = stack_indexed(caches, idx, "train")
    bb_vl, rppg_vl, blink_vl, y_vl = stack_indexed(caches, idx, "val")
    bb_te, rppg_te, blink_te, y_te = stack_indexed(caches, idx, "test")
    print(f"[source-val] n_train={len(y_tr)} n_val={len(y_vl)} "
          f"(FF+DFDC source) n_test={len(y_te)} (CelebDF)")

    op_rows = []
    for seed in args.seeds:
        for variant in VARIANTS:
            X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
            X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
            X_te = make_features(bb_te, rppg_te, blink_te, variant)

            probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                       epochs=args.epochs, lr=args.lr,
                                       bs=args.batch, seed=seed)
            val_scores = predict(probe, X_vl, device)
            test_scores = predict(probe, X_te, device)

            np.savez(out_dir / "scores" / f"test_celebdf_s{seed}_{variant}.npz",
                     scores=test_scores, labels=y_te)
            np.savez(out_dir / "scores" / f"val_s{seed}_{variant}.npz",
                     scores=val_scores, labels=y_vl)

            test_auc = roc_auc(y_te, test_scores)
            for fpr in FPR_LEVELS:
                thr = fpr_threshold_from_val(y_vl, val_scores, fpr)
                tpr_sv, fpr_realised = tpr_fpr_at_threshold(
                    y_te, test_scores, thr)
                tpr_oracle = tpr_at_fpr(y_te, test_scores, fpr)
                op_rows.append({
                    "variant": variant, "seed": seed,
                    "nominal_fpr": fpr,
                    "test_auc": round(test_auc, 4),
                    "thr_from_val": round(thr, 4),
                    "tpr_source_val": round(tpr_sv, 4),
                    "fpr_realised_on_test": round(fpr_realised, 4),
                    "tpr_oracle_fpr": round(tpr_oracle, 4),
                    "tpr_gap_oracle_minus_sourceval":
                        round(tpr_oracle - tpr_sv, 4),
                })
            print(f"  seed={seed} {variant:16s} AUC={test_auc:.4f}")

    head = list(op_rows[0].keys())
    with open(out_dir / "source_val_operating_points.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head)
        w.writeheader()
        w.writerows(op_rows)
    print(f"[source-val] wrote source_val_operating_points.csv "
          f"({len(op_rows)} rows)")

    # Seed-averaged summary per (variant, fpr).
    agg = defaultdict(list)
    for r in op_rows:
        agg[(r["variant"], r["nominal_fpr"])].append(r)
    summary = []
    for (variant, fpr), bucket in agg.items():
        summary.append({
            "variant": variant, "nominal_fpr": fpr,
            "tpr_source_val_mean":
                round(float(np.mean([b["tpr_source_val"] for b in bucket])), 4),
            "tpr_oracle_mean":
                round(float(np.mean([b["tpr_oracle_fpr"] for b in bucket])), 4),
            "fpr_realised_mean":
                round(float(np.mean([b["fpr_realised_on_test"] for b in bucket])), 4),
        })
    with open(out_dir / "source_val_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[source-val] wrote source_val_summary.json")
    print("[source-val] done. Copy the out_dir back to the artefact bundle "
          "and run scripts/source_val_summary.py locally to build the table.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--celebdf_root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[0, 1, 42, 1337, 2024])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    main(ap.parse_args())
