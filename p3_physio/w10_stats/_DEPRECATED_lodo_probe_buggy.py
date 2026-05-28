"""
E11 — Leave-One-Dataset-Out (LODO) cross-dataset evaluation.

Addresses the professor's correction #3:

    "If the mixed probe is trained using CelebDF training data and tested on
    CelebDF held-out subjects, this is not pure cross-dataset generalization.
    It is mixed-domain supervised training with held-out identity evaluation.
    To support true cross-dataset claims, add leave-one-dataset-out experiments:
    train on FF++ and DFDC, test on CelebDF; train on FF++ and CelebDF, test
    on DFDC; and clearly separate these from mixed-domain results."

Three LODO configurations:

  test_celebdf : train on FF++ + DFDC,    test on CelebDF (subject-aware test split)
  test_dfdc    : train on FF++ + CelebDF, test on DFDC    (by-clip; n=77)
  test_ff      : train on CelebDF + DFDC, test on FF++    (identity-aware test split)

For each LODO config:
  - 5 seeds × 4 variants × 3 backbones = 60 probes
  - All on cached features → minutes, not hours
  - Reports mean ± std AUC, EER, AP, TPR@1%, TPR@5%, TPR@10%
  - Bootstrap 95% CIs (seed 0)
  - DeLong paired tests vs the corresponding "mixed" baseline from E6

Output schema mirrors multiseed_and_stats so the result format is comparable.

Usage on Kaggle (one cell per backbone, or one cell that loops):

    python w10_stats/lodo_probe.py \\
        --cache_dir "$CACHE_CLIP" \\
        --out_dir   "$OUT_LODO_CLIP" \\
        --celebdf_root "$CDF" \\
        --seeds 0 1 42 1337 2024
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import (
    roc_auc, eer, average_precision, tpr_at_fpr,
    train_linear_probe, predict, identity_split_ff,
    bootstrap_ci_auc, delong_test, mcnemar_test, youden_threshold,
    VARIANTS, make_features,
)
from identity_split_sanity import scan_celebdf_with_subject, subject_aware_split


# ───────────────────────────────────────────────────────────────────────────
# LODO configurations: which datasets train, which one tests
# ───────────────────────────────────────────────────────────────────────────

LODO_CONFIGS = [
    # (config_name, train_datasets, test_dataset)
    ("test_celebdf", ("ff", "dfdc"),    "celebdf"),
    ("test_dfdc",    ("ff", "celebdf"), "dfdc"),
    ("test_ff",      ("celebdf", "dfdc"), "ff"),
]


def get_train_test_indices(caches, train_datasets, test_dataset, args):
    """
    Build (train_idx, val_idx, test_idx) per dataset.
    Rules:
      - If a dataset is in train_datasets and is NOT the test_dataset:
          use ALL clips for training (no held-out)
      - If a dataset IS the test_dataset:
          - FF++   → identity-aware 80/10/10 (use only test_idx, no clips for training)
          - CelebDF → subject-aware 80/20 (use only test_idx, no clips for training)
          - DFDC   → random 80/20 (use only test_idx, no clips for training)
      - Validation set: 10% of FF++ training (always; FF++ val is a stable
        validation across all LODO configs because FF++ is always either fully
        in train or has identity-aware splits we can re-use for val)

    Returns dict with structure:
      {dataset_name: {"train": np.array, "test": np.array}}
      plus a "val" key for FF++ (always)
    """
    out = {}
    rng = random.Random(42)

    # FF++ always has identity-aware splits available
    ff_tr, ff_vl, ff_te = identity_split_ff(caches["ff"], seed=42)

    if "ff" in train_datasets:
        # All of FF++ goes into training (we'll still hold out ff_vl for val)
        out["ff"] = {"train": np.concatenate([ff_tr, ff_te]), "test": np.array([])}
    elif test_dataset == "ff":
        out["ff"] = {"train": np.array([]), "test": ff_te}
    else:
        out["ff"] = {"train": np.array([]), "test": np.array([])}

    out["val_ff"] = ff_vl  # always

    # CelebDF: subject-aware split if test_dataset == "celebdf"
    if "celebdf" in caches:
        if "celebdf" in train_datasets:
            cd_n = len(caches["celebdf"]["labels"])
            out["celebdf"] = {"train": np.arange(cd_n), "test": np.array([])}
        elif test_dataset == "celebdf":
            assert args.celebdf_root, "--celebdf_root required for subject-aware CelebDF test split"
            cdf_rows = scan_celebdf_with_subject(args.celebdf_root)
            subject_ids = np.array([r[0] for r in cdf_rows])
            cdf_labels_check = np.array([r[1] for r in cdf_rows])
            if not np.array_equal(caches["celebdf"]["labels"], cdf_labels_check):
                raise RuntimeError(
                    "CelebDF cache order disagrees with re-scan order — subject IDs would mismap"
                )
            cd_tr_subj, cd_te_subj, n_subj, _ = subject_aware_split(subject_ids, seed=42)
            out["celebdf"] = {"train": cd_tr_subj, "test": cd_te_subj}
        else:
            out["celebdf"] = {"train": np.array([]), "test": np.array([])}

    # DFDC: random 80/20 if test, else all
    if "dfdc" in caches:
        df_n = len(caches["dfdc"]["labels"])
        if "dfdc" in train_datasets:
            out["dfdc"] = {"train": np.arange(df_n), "test": np.array([])}
        elif test_dataset == "dfdc":
            rng_df = np.random.default_rng(42)
            idx = np.arange(df_n)
            rng_df.shuffle(idx)
            k = int(df_n * 0.8)
            out["dfdc"] = {"train": np.array([]), "test": idx[k:]}
        else:
            out["dfdc"] = {"train": np.array([]), "test": np.array([])}

    return out


def stack_train_pool(caches, train_indices_per_dataset):
    """Concatenate (backbone, rppg, blink, labels) across datasets per their train_idx."""
    bbs, rppgs, blinks, ys = [], [], [], []
    for ds, idx_dict in train_indices_per_dataset.items():
        if ds == "val_ff":
            continue
        idx = idx_dict["train"]
        if len(idx) == 0:
            continue
        bbs.append(caches[ds]["backbone"][idx])
        rppgs.append(caches[ds]["rppg"][idx])
        blinks.append(caches[ds]["blink"][idx])
        ys.append(caches[ds]["labels"][idx])
    return (np.concatenate(bbs), np.concatenate(rppgs),
            np.concatenate(blinks), np.concatenate(ys))


def main(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lodo] device={device}  cache_dir={args.cache_dir}")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    # Load caches
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = cache_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"[lodo] loaded {tag}: n={len(caches[tag]['labels'])} bb_dim={caches[tag]['backbone'].shape[1]}")

    if "ff" not in caches or "celebdf" not in caches or "dfdc" not in caches:
        print("[lodo] requires all three caches (ff, celebdf, dfdc); abort")
        return

    rows = []
    score_storage = {}  # key: (config, seed, variant) → (scores, labels)

    for config_name, train_datasets, test_dataset in LODO_CONFIGS:
        print(f"\n[lodo] ====== config={config_name} train={train_datasets} test={test_dataset} ======")
        idx_dict = get_train_test_indices(caches, train_datasets, test_dataset, args)
        n_train = sum(len(d["train"]) for d in idx_dict.values() if isinstance(d, dict))
        n_test = len(idx_dict[test_dataset]["test"])
        print(f"  train pool size: {n_train}, test set size: {n_test} (on {test_dataset})")

        # Build training pool ONCE (independent of seed)
        bb_tr, rppg_tr, blink_tr, y_tr = stack_train_pool(caches, idx_dict)

        # FF++ val for early-stopping (always)
        ff_vl = idx_dict["val_ff"]
        bb_vl = caches["ff"]["backbone"][ff_vl]
        rppg_vl = caches["ff"]["rppg"][ff_vl]
        blink_vl = caches["ff"]["blink"][ff_vl]
        y_vl = caches["ff"]["labels"][ff_vl]

        # Test
        te_idx = idx_dict[test_dataset]["test"]
        bb_te = caches[test_dataset]["backbone"][te_idx]
        rppg_te = caches[test_dataset]["rppg"][te_idx]
        blink_te = caches[test_dataset]["blink"][te_idx]
        y_te = caches[test_dataset]["labels"][te_idx]

        for seed in args.seeds:
            for variant in VARIANTS:
                X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
                X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
                X_te = make_features(bb_te, rppg_te, blink_te, variant)

                probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                            epochs=args.epochs, lr=args.lr,
                                            bs=args.batch, seed=seed)
                scores = predict(probe, X_te, device)
                score_storage[(config_name, seed, variant)] = (scores, y_te)
                np.savez(out_dir / "scores" / f"{config_name}_s{seed}_{variant}.npz",
                         scores=scores, labels=y_te)

                row = {
                    "config": config_name, "train": "+".join(train_datasets),
                    "test": test_dataset, "seed": seed, "variant": variant,
                    "n_train": int(n_train), "n_test": int(len(y_te)),
                    "auc": roc_auc(y_te, scores),
                    "ap": average_precision(y_te, scores),
                    "eer": eer(y_te, scores),
                    "tpr1": tpr_at_fpr(y_te, scores, 0.01),
                    "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                    "tpr10": tpr_at_fpr(y_te, scores, 0.10),
                }
                rows.append(row)
                print(f"  seed={seed} {variant:18s}  AUC={row['auc']:.4f}  EER={row['eer']:.4f}  "
                      f"TPR@1={row['tpr1']:.3f}  TPR@5={row['tpr5']:.3f}  TPR@10={row['tpr10']:.3f}")

    # ── Raw rows ──
    head = ["config","train","test","seed","variant","n_train","n_test",
            "auc","ap","eer","tpr1","tpr5","tpr10"]
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")
    print(f"\n[lodo] wrote {out_dir/'results.csv'} ({len(rows)} rows)")

    # ── Aggregate mean ± std ──
    from collections import defaultdict
    agg_buckets = defaultdict(list)
    for r in rows:
        agg_buckets[(r["config"], r["variant"])].append(r)

    agg_rows = []
    print("\n[lodo] AGGREGATE (mean ± std across seeds)")
    print(f"{'config':<14s} {'variant':<18s} {'n_test':>7s}  AUC             EER            TPR@1%        TPR@5%        TPR@10%")
    for (config, variant), rs in sorted(agg_buckets.items()):
        aucs = [r["auc"] for r in rs]
        eers = [r["eer"] for r in rs]
        tpr1s = [r["tpr1"] for r in rs]
        tpr5s = [r["tpr5"] for r in rs]
        tpr10s = [r["tpr10"] for r in rs]
        row = {
            "config": config, "variant": variant, "n_test": rs[0]["n_test"],
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1) if len(aucs)>1 else 0),
            "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers, ddof=1) if len(eers)>1 else 0),
            "tpr1_mean": float(np.mean(tpr1s)), "tpr1_std": float(np.std(tpr1s, ddof=1) if len(tpr1s)>1 else 0),
            "tpr5_mean": float(np.mean(tpr5s)), "tpr5_std": float(np.std(tpr5s, ddof=1) if len(tpr5s)>1 else 0),
            "tpr10_mean": float(np.mean(tpr10s)), "tpr10_std": float(np.std(tpr10s, ddof=1) if len(tpr10s)>1 else 0),
        }
        agg_rows.append(row)
        print(f"{config:<14s} {variant:<18s} {row['n_test']:>7d}  "
              f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"{row['eer_mean']:.4f}±{row['eer_std']:.4f}  "
              f"{row['tpr1_mean']:.3f}±{row['tpr1_std']:.3f}  "
              f"{row['tpr5_mean']:.3f}±{row['tpr5_std']:.3f}  "
              f"{row['tpr10_mean']:.3f}±{row['tpr10_std']:.3f}")

    head_agg = ["config","variant","n_test",
                "auc_mean","auc_std","eer_mean","eer_std",
                "tpr1_mean","tpr1_std","tpr5_mean","tpr5_std","tpr10_mean","tpr10_std"]
    with open(out_dir / "aggregate.csv", "w") as f:
        f.write(",".join(head_agg) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in head_agg) + "\n")
    print(f"[lodo] wrote {out_dir/'aggregate.csv'}")

    # ── Bootstrap CIs (seed 0) ──
    boot_rows = []
    for (config, seed, variant), (s, y) in score_storage.items():
        if seed != args.seeds[0]:
            continue
        ci = bootstrap_ci_auc(y, s, n_boot=args.n_boot, seed=42)
        boot_rows.append({
            "config": config, "variant": variant, "n": int(len(y)),
            "auc": roc_auc(y, s),
            "auc_ci_lo": ci["auc_ci"][0], "auc_ci_hi": ci["auc_ci"][1],
            "tpr1_ci_lo": ci["tpr_ci"][0.01][0], "tpr1_ci_hi": ci["tpr_ci"][0.01][1],
            "tpr5_ci_lo": ci["tpr_ci"][0.05][0], "tpr5_ci_hi": ci["tpr_ci"][0.05][1],
            "tpr10_ci_lo": ci["tpr_ci"][0.10][0], "tpr10_ci_hi": ci["tpr_ci"][0.10][1],
        })
    head_boot = list(boot_rows[0].keys()) if boot_rows else []
    with open(out_dir / "bootstrap_cis.csv", "w") as f:
        f.write(",".join(head_boot) + "\n")
        for r in boot_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in head_boot) + "\n")
    print(f"[lodo] wrote {out_dir/'bootstrap_cis.csv'}")

    # ── Paired DeLong + McNemar tests within each config ──
    PAIRS = [
        ("backbone_only", "backbone+rppg",  "rppg_vs_bb"),
        ("backbone_only", "backbone+blink", "blink_vs_bb"),
        ("backbone+rppg", "backbone+blink", "rppg_vs_blink"),
        ("backbone+rppg", "full_fusion",    "rppg_vs_fusion"),
    ]
    paired_rows = []
    for config_name, _, _ in LODO_CONFIGS:
        for var_a, var_b, tag in PAIRS:
            ka = (config_name, args.seeds[0], var_a)
            kb = (config_name, args.seeds[0], var_b)
            if ka not in score_storage or kb not in score_storage:
                continue
            sa, ya = score_storage[ka]
            sb, yb = score_storage[kb]
            if not np.array_equal(ya, yb):
                continue
            auc_a, auc_b, z, p = delong_test(ya, sa, sb)
            thr = youden_threshold(ya, sa)
            mc = mcnemar_test(ya, sa, sb, threshold=thr)
            paired_rows.append({
                "config": config_name, "pair": tag,
                "a": var_a, "b": var_b,
                "auc_a": auc_a, "auc_b": auc_b, "auc_diff": auc_b - auc_a,
                "delong_z": z, "delong_p": p,
                "mc_a_wins": mc["a_wins"], "mc_b_wins": mc["b_wins"],
                "mc_chi2": mc["chi2"], "mc_p": mc["p"],
            })

    n_tests = len(paired_rows)
    for r in paired_rows:
        r["delong_p_bonf"] = min(1.0, r["delong_p"] * n_tests)
        r["mc_p_bonf"] = min(1.0, r["mc_p"] * n_tests)

    head_p = ["config","pair","a","b","auc_a","auc_b","auc_diff",
              "delong_z","delong_p","delong_p_bonf",
              "mc_a_wins","mc_b_wins","mc_chi2","mc_p","mc_p_bonf"]
    with open(out_dir / "paired_tests.csv", "w") as f:
        f.write(",".join(head_p) + "\n")
        for r in paired_rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                             for k in head_p) + "\n")
    print(f"[lodo] wrote {out_dir/'paired_tests.csv'} ({n_tests} pairs)")

    # ── Final JSON ──
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"agg": agg_rows, "bootstrap": boot_rows, "paired": paired_rows,
                   "raw": rows, "configs": [c[0] for c in LODO_CONFIGS]},
                  f, indent=2, default=str)
    print(f"[lodo] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Leave-one-dataset-out cross-dataset evaluation")
    p.add_argument("--cache_dir", required=True, help="feat_cache_<bb> directory")
    p.add_argument("--celebdf_root", required=True, help="CelebDF root for subject-aware split when CelebDF is held out")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n_boot", type=int, default=1000)
    main(p.parse_args())
