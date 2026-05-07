"""
E14 — Strict LODO probe (no target-domain validation leak).

Fix for the v3-review professor correction: in the original
`lodo_probe.py`, the validation partition used for probe early stopping
was always drawn from FF++ (`out["val_ff"] = ff_vl`). When
`test_dataset == "ff"`, that's a target-domain leak.

This script reruns ALL THREE LODO configs with a leak-free protocol:
the validation partition is always drawn from a non-target dataset.

Validation rules (strict):
  - test_dataset == "celebdf"  →  val from FF++ identity-aware val split
  - test_dataset == "dfdc"     →  val from FF++ identity-aware val split
  - test_dataset == "ff"       →  val from CelebDF + DFDC training pool
                                  (held-out 10% of training data,
                                   subject-aware on the CelebDF portion)

For test_celebdf and test_dfdc the val partition is unchanged from the
original run (FF was a non-target training dataset; FF val was clean).
For test_ff this is the new strict configuration.

We re-run ALL three configs anyway to ensure (a) results are reproducible
on the new code path and (b) seed-by-seed outputs are directly comparable
between strict and previous (now relabeled "quasi-strict") runs.

Output schema mirrors `lodo_probe.py` so downstream aggregation,
bootstrap, and figure scripts continue to work unchanged. Output dir is
expected to be different (e.g. `e14_lodo_strict_clip/`) so the v3 outputs
in `e11_lodo_clip/` are not overwritten.

Usage on Kaggle:

    python w10_stats/lodo_probe_strict.py \\
        --cache_dir   "$CACHE_CLIP" \\
        --out_dir     "$OUT_LODO_STRICT_CLIP" \\
        --celebdf_root "$CDF" \\
        --seeds 0 1 42 1337 2024
"""

import argparse
import json
import random
import sys
from collections import defaultdict
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


LODO_CONFIGS = [
    ("test_celebdf", ("ff", "dfdc"),    "celebdf"),
    ("test_dfdc",    ("ff", "celebdf"), "dfdc"),
    ("test_ff",      ("celebdf", "dfdc"), "ff"),
]

VAL_FRAC = 0.10  # 10% of training pool used for early-stopping val


def _shuffle(rng_seed, arr):
    rng = np.random.default_rng(rng_seed)
    a = np.asarray(arr).copy()
    rng.shuffle(a)
    return a


def get_train_val_test_indices_strict(caches, train_datasets, test_dataset, args):
    """
    Build (train_idx, val_idx, test_idx) per dataset under STRICT LODO.

    Rules:
      - Training datasets contribute their full clip set to training EXCEPT
        a held-out fraction (VAL_FRAC) used for validation. The held-out
        fraction is taken from a non-target training dataset.
      - For CelebDF when it is a training dataset, the val split is
        subject-aware (no subject overlap between train and val).
      - For FF++ when it is a training dataset, the val split uses the
        identity-aware val partition from `identity_split_ff` (clean).
      - For DFDC when it is a training dataset, the val split is random
        (no subject info available); only used when DFDC is the only
        non-target training dataset (does not happen in our 3 configs).
      - The test partition for the target dataset uses subject-aware /
        identity-aware splits identical to the original lodo_probe.py.

    Returns dict:
      {dataset_name: {"train": np.array, "val": np.array, "test": np.array}}
    """
    out = {ds: {"train": np.array([], dtype=np.int64),
                "val":   np.array([], dtype=np.int64),
                "test":  np.array([], dtype=np.int64)}
           for ds in ("ff", "celebdf", "dfdc")}

    # ── FF++ identity-aware split (always available) ──
    ff_tr, ff_vl, ff_te = identity_split_ff(caches["ff"], seed=42)

    # ── CelebDF subject-aware split (when CelebDF is in caches) ──
    cd_subj_tr = cd_subj_te = None
    if "celebdf" in caches and args.celebdf_root:
        cdf_rows = scan_celebdf_with_subject(args.celebdf_root)
        subject_ids = np.array([r[0] for r in cdf_rows])
        cdf_labels_check = np.array([r[1] for r in cdf_rows])
        if not np.array_equal(caches["celebdf"]["labels"], cdf_labels_check):
            raise RuntimeError(
                "CelebDF cache order disagrees with re-scan order — subject IDs would mismap"
            )
        cd_subj_tr, cd_subj_te, _, _ = subject_aware_split(subject_ids, seed=42)

    # ── DFDC: random 80/20 split ──
    df_tr = df_te = None
    if "dfdc" in caches:
        df_n = len(caches["dfdc"]["labels"])
        df_idx = _shuffle(42, np.arange(df_n))
        k = int(df_n * 0.8)
        df_tr = df_idx[:k]
        df_te = df_idx[k:]

    # ──────────────────────────────────────────────────────────
    # Target-test partition (held entirely out of train and val)
    # ──────────────────────────────────────────────────────────
    if test_dataset == "ff":
        out["ff"]["test"] = ff_te
    elif test_dataset == "celebdf":
        out["celebdf"]["test"] = cd_subj_te
    elif test_dataset == "dfdc":
        out["dfdc"]["test"] = df_te

    # ──────────────────────────────────────────────────────────
    # Training-dataset contributions
    # ──────────────────────────────────────────────────────────
    # Pick which non-target training dataset provides the val partition.
    # Priority: FF++ (cleanest, identity-aware) > CelebDF (subject-aware)
    # > DFDC (random). For our 3 LODO configs:
    #   test_celebdf: train=(ff,dfdc) → val from FF++
    #   test_dfdc:    train=(ff,celebdf) → val from FF++
    #   test_ff:      train=(celebdf,dfdc) → val from CelebDF (subject-aware)

    # FF++ contribution
    if "ff" in train_datasets:
        # All of FF goes to training; val for non-FF target uses ff_vl
        out["ff"]["train"] = np.concatenate([ff_tr, ff_te])
        if test_dataset != "ff":
            out["ff"]["val"] = ff_vl

    # CelebDF contribution
    if "celebdf" in train_datasets:
        if test_dataset == "ff":
            # We need val from CelebDF (subject-aware) since FF is target
            # Use cd_subj_tr for training, then within cd_subj_tr split off
            # 10% (subject-aware) for val by sampling subjects.
            cdf_rows = scan_celebdf_with_subject(args.celebdf_root)
            subject_ids = np.array([r[0] for r in cdf_rows])
            tr_subjects = np.unique(subject_ids[cd_subj_tr])
            rng = np.random.default_rng(123)
            tr_subjects_shuf = rng.permutation(tr_subjects)
            n_val_subj = max(1, int(len(tr_subjects_shuf) * VAL_FRAC))
            val_subjects = set(tr_subjects_shuf[:n_val_subj])
            train_mask = np.array([sid not in val_subjects
                                   for sid in subject_ids[cd_subj_tr]])
            out["celebdf"]["train"] = cd_subj_tr[train_mask]
            out["celebdf"]["val"]   = cd_subj_tr[~train_mask]
            # CelebDF test is fully held out (it is not the target here)
        else:
            # CelebDF is just a training dataset; ALL goes to train
            cd_n = len(caches["celebdf"]["labels"])
            out["celebdf"]["train"] = np.arange(cd_n)
            # CelebDF val unused (FF val is the early-stopping signal here)

    # DFDC contribution
    if "dfdc" in train_datasets:
        df_n = len(caches["dfdc"]["labels"])
        out["dfdc"]["train"] = np.arange(df_n)
        # DFDC val unused (FF/CelebDF provides the val signal)

    return out


def stack_indexed(caches, idx_dict, key):
    bbs, rppgs, blinks, ys = [], [], [], []
    for ds in ("ff", "celebdf", "dfdc"):
        if ds not in caches:
            continue
        idx = idx_dict[ds][key]
        if len(idx) == 0:
            continue
        bbs.append(caches[ds]["backbone"][idx])
        rppgs.append(caches[ds]["rppg"][idx])
        blinks.append(caches[ds]["blink"][idx])
        ys.append(caches[ds]["labels"][idx])
    if len(bbs) == 0:
        return None, None, None, None
    return (np.concatenate(bbs), np.concatenate(rppgs),
            np.concatenate(blinks), np.concatenate(ys))


def main(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[strict-lodo] device={device}  cache_dir={args.cache_dir}")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = cache_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"[strict-lodo] loaded {tag}: n={len(caches[tag]['labels'])} "
                  f"bb_dim={caches[tag]['backbone'].shape[1]}")

    if "ff" not in caches or "celebdf" not in caches or "dfdc" not in caches:
        print("[strict-lodo] requires all three caches (ff, celebdf, dfdc); abort")
        return

    rows = []

    for config_name, train_datasets, test_dataset in LODO_CONFIGS:
        print(f"\n[strict-lodo] ====== config={config_name} "
              f"train={train_datasets} test={test_dataset} ======")
        idx_dict = get_train_val_test_indices_strict(
            caches, train_datasets, test_dataset, args)

        bb_tr, rppg_tr, blink_tr, y_tr = stack_indexed(caches, idx_dict, "train")
        bb_vl, rppg_vl, blink_vl, y_vl = stack_indexed(caches, idx_dict, "val")
        bb_te, rppg_te, blink_te, y_te = stack_indexed(caches, idx_dict, "test")

        # Diagnostics
        n_train = len(y_tr) if y_tr is not None else 0
        n_val   = len(y_vl) if y_vl is not None else 0
        n_test  = len(y_te) if y_te is not None else 0
        val_source = ("FF++ (identity-aware)" if test_dataset != "ff"
                      else "CelebDF (subject-aware)")
        print(f"  n_train={n_train}  n_val={n_val} (from {val_source})  n_test={n_test}")

        # Sanity: target dataset must NOT contribute to train or val
        if test_dataset == "ff":
            assert len(idx_dict["ff"]["train"]) == 0, "FF leaked into train"
            assert len(idx_dict["ff"]["val"])   == 0, "FF leaked into val"
        if test_dataset == "celebdf":
            assert len(idx_dict["celebdf"]["train"]) == 0, "CelebDF leaked into train"
            assert len(idx_dict["celebdf"]["val"])   == 0, "CelebDF leaked into val"
        if test_dataset == "dfdc":
            assert len(idx_dict["dfdc"]["train"]) == 0, "DFDC leaked into train"
            assert len(idx_dict["dfdc"]["val"])   == 0, "DFDC leaked into val"
        print(f"  [audit] target {test_dataset!r} fully held out from train AND val ✓")

        for seed in args.seeds:
            for variant in VARIANTS:
                X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
                X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
                X_te = make_features(bb_te, rppg_te, blink_te, variant)

                probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                           epochs=args.epochs, lr=args.lr,
                                           bs=args.batch, seed=seed)
                scores = predict(probe, X_te, device)
                np.savez(out_dir / "scores" / f"{config_name}_s{seed}_{variant}.npz",
                         scores=scores, labels=y_te)

                row = {
                    "config": config_name, "train": "+".join(train_datasets),
                    "test": test_dataset, "seed": seed, "variant": variant,
                    "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
                    "val_source": val_source,
                    "auc": roc_auc(y_te, scores),
                    "ap": average_precision(y_te, scores),
                    "eer": eer(y_te, scores),
                    "tpr1": tpr_at_fpr(y_te, scores, 0.01),
                    "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                    "tpr10": tpr_at_fpr(y_te, scores, 0.10),
                }
                rows.append(row)
                print(f"  seed={seed} {variant:18s}  AUC={row['auc']:.4f}  "
                      f"EER={row['eer']:.4f}  TPR@5={row['tpr5']:.3f}")

    # ── Raw rows ──
    head = ["config","train","test","seed","variant","n_train","n_val","n_test",
            "val_source","auc","ap","eer","tpr1","tpr5","tpr10"]
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")
    print(f"\n[strict-lodo] wrote {out_dir/'results.csv'} ({len(rows)} rows)")

    # ── Aggregate ──
    agg_buckets = defaultdict(list)
    for r in rows:
        agg_buckets[(r["config"], r["variant"])].append(r)

    agg_rows = []
    for (config, variant), bucket in agg_buckets.items():
        n_test = bucket[0]["n_test"]
        aucs = np.array([b["auc"]   for b in bucket])
        eers = np.array([b["eer"]   for b in bucket])
        tpr1 = np.array([b["tpr1"]  for b in bucket])
        tpr5 = np.array([b["tpr5"]  for b in bucket])
        tpr10 = np.array([b["tpr10"] for b in bucket])
        agg_rows.append({
            "config": config, "variant": variant, "n_test": n_test,
            "val_source": bucket[0]["val_source"],
            "auc_mean": aucs.mean(), "auc_std": aucs.std(ddof=1),
            "eer_mean": eers.mean(), "eer_std": eers.std(ddof=1),
            "tpr1_mean": tpr1.mean(), "tpr1_std": tpr1.std(ddof=1),
            "tpr5_mean": tpr5.mean(), "tpr5_std": tpr5.std(ddof=1),
            "tpr10_mean": tpr10.mean(), "tpr10_std": tpr10.std(ddof=1),
        })

    agg_head = ["config","variant","n_test","val_source",
                "auc_mean","auc_std","eer_mean","eer_std",
                "tpr1_mean","tpr1_std","tpr5_mean","tpr5_std",
                "tpr10_mean","tpr10_std"]
    with open(out_dir / "aggregate.csv", "w") as f:
        f.write(",".join(agg_head) + "\n")
        for r in agg_rows:
            f.write(",".join(str(r[k]) for k in agg_head) + "\n")
    print(f"[strict-lodo] wrote {out_dir/'aggregate.csv'} ({len(agg_rows)} rows)")

    # ── Summary JSON ──
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n_seeds": len(args.seeds),
            "seeds": list(args.seeds),
            "configs": [c[0] for c in LODO_CONFIGS],
            "variants": list(VARIANTS),
            "agg": agg_rows,
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"[strict-lodo] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",   required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--celebdf_root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42, 1337, 2024])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--batch",  type=int, default=256)
    main(ap.parse_args())
