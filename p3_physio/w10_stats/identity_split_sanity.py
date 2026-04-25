"""
E7 — Identity-aware split sanity check.

Why this exists:
    The multiseed_and_stats.py probe splits CelebDF 80/20 by clip
    (random.Random(42).shuffle).  CelebDF clip names look like
    "id0_id1_0001" where id0 is the SUBJECT id.  CelebDF is built from a
    closed pool of celebrities — each celebrity appears in many videos.
    A by-clip split therefore puts the SAME celebrity into both train and
    test, which is identity leakage and inflates the AUC.

    On a strong backbone (CLIP), this could be the difference between
    "we beat SOTA" (0.974) and "we trail SOTA" (~0.85).  Need to know
    which one is real before we lock in the thesis claim.

Design:
    Re-extracts the CelebDF subject_ids in the same order as the cache
    was built (deterministic — both scanners sort identically), then re-
    runs the full multi-seed probe with two split flavours:

      A) by_clip   — random 80/20 (current behaviour, identity-leaky)
      B) by_subject — random 80/20 over UNIQUE subject IDs, then assign
                      every clip of subject S to the same partition
                      (no identity leakage)

    Reports both AUCs side-by-side per (backbone, regime, variant).

Usage on Kaggle:

    python w10_stats/identity_split_sanity.py \\
        --celebdf_root  "$CDF" \\
        --cache_dir     "$CACHE_CLIP" \\
        --out_dir       /kaggle/working/sanity_clip_idsplit \\
        --seeds 0 1 42 1337 2024
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the metric helpers + probe from multiseed_and_stats
sys.path.insert(0, str(Path(__file__).parent))
from multiseed_and_stats import (
    roc_auc, eer, average_precision, tpr_at_fpr,
    train_linear_probe, predict, identity_split_ff, random_split,
    bootstrap_ci_auc,
    VARIANTS, REGIMES, make_features,
)

# Same scanner used by Stage 1 of multiseed_and_stats — guarantees same order
from multiseed_and_stats import scan_celebdf as _scan_celebdf_paths


def parse_celebdf_subject_id(video_dir_name):
    """
    CelebDF clip naming convention:
      id0_id1_0001   → subject id0  (first 'id<N>' token)
      id0_0001       → subject id0
      id123_id56_0007 → subject id123

    The convention is: subject id is everything up to the second underscore
    if the clip has a swap-target id, or up to the first underscore otherwise.
    Practical rule: the subject id is the first '_'-separated token.
    """
    return video_dir_name.split("_")[0]


def scan_celebdf_with_subject(root):
    """Scan CelebDF and produce list of (subject_id, label) in cache order."""
    root = Path(root)
    rows = []
    for split in ["Test", "Train"]:
        for lname in ["real", "fake"]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            for sd in sorted(d for d in ldir.iterdir() if d.is_dir()):
                if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                    rows.append((parse_celebdf_subject_id(sd.name),
                                 0 if lname == "real" else 1,
                                 sd.name))
    return rows


def subject_aware_split(subject_ids, seed=42, frac=0.8):
    """
    Split clip indices 80/20 by SUBJECT ID.
    Every clip of subject S goes to the same partition.
    """
    unique_subjects = sorted(set(subject_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)
    n_train_subjects = int(len(unique_subjects) * frac)
    train_subjects = set(unique_subjects[:n_train_subjects])
    train_idx, test_idx = [], []
    for i, sid in enumerate(subject_ids):
        if sid in train_subjects:
            train_idx.append(i)
        else:
            test_idx.append(i)
    return np.array(train_idx), np.array(test_idx), len(unique_subjects), len(train_subjects)


def main(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] device={device}")
    print(f"[sanity] cache_dir={args.cache_dir}")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reconstruct subject_ids per CelebDF clip in cache order ─────────
    print("[sanity] re-scanning CelebDF in cache order to derive subject IDs ...")
    cdf_rows = scan_celebdf_with_subject(args.celebdf_root)
    subject_ids = np.array([r[0] for r in cdf_rows])
    cdf_labels_check = np.array([r[1] for r in cdf_rows])
    cdf_clip_names = np.array([r[2] for r in cdf_rows])
    print(f"[sanity] re-scanned {len(cdf_rows)} CelebDF clips, "
          f"{len(set(subject_ids))} unique subjects")

    # Real vs fake distribution per subject
    subj_class = {}
    for sid, lbl in zip(subject_ids, cdf_labels_check):
        subj_class.setdefault(sid, []).append(int(lbl))
    n_real_subj = sum(1 for s, lbls in subj_class.items() if 0 in lbls)
    n_fake_subj = sum(1 for s, lbls in subj_class.items() if 1 in lbls)
    n_both_subj = sum(1 for s, lbls in subj_class.items() if 0 in lbls and 1 in lbls)
    print(f"[sanity] subjects with real clips: {n_real_subj}, "
          f"with fake clips: {n_fake_subj}, with BOTH: {n_both_subj}")

    # ── Load caches ───────────────────────────────────────────────────────
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        fpath = cache_dir / f"{tag}.npz"
        if fpath.exists():
            caches[tag] = {k: v for k, v in np.load(fpath, allow_pickle=True).items()}
            print(f"[sanity] loaded {tag}: n={len(caches[tag]['labels'])} "
                  f"bb_dim={caches[tag]['backbone'].shape[1]}")

    # Sanity: check labels match between cache and re-scan (catches order drift)
    cdf_cache_labels = caches["celebdf"]["labels"]
    if not np.array_equal(cdf_cache_labels, cdf_labels_check):
        print(f"[sanity] WARNING: cache labels disagree with re-scan order")
        print(f"  first 10 cache:  {cdf_cache_labels[:10].tolist()}")
        print(f"  first 10 rescan: {cdf_labels_check[:10].tolist()}")
        diff_count = (cdf_cache_labels != cdf_labels_check).sum()
        print(f"  {diff_count} mismatches out of {len(cdf_cache_labels)}")
        if diff_count > 0:
            print("[sanity] ABORTING — order divergence means subject IDs would mismap.")
            return
    else:
        print(f"[sanity] OK: cache order == re-scan order ({len(cdf_cache_labels)} clips)")

    # ── Identity-aware split for FF++ (unchanged) ─────────────────────────
    tr_ff_idx, vl_ff_idx, te_ff_idx = identity_split_ff(caches["ff"], seed=42)
    print(f"[sanity] FF++ identity split: train={len(tr_ff_idx)} val={len(vl_ff_idx)} test={len(te_ff_idx)}")

    # ── Two CelebDF splits to compare ─────────────────────────────────────
    cd_tr_byclip, cd_te_byclip = random_split(len(cdf_cache_labels), seed=42)
    cd_tr_bysubj, cd_te_bysubj, n_subj, n_train_subj = subject_aware_split(subject_ids, seed=42)
    print(f"[sanity] CelebDF by-clip split: train={len(cd_tr_byclip)} test={len(cd_te_byclip)}")
    print(f"[sanity] CelebDF by-subject split: train={len(cd_tr_bysubj)} test={len(cd_te_bysubj)} "
          f"({n_train_subj}/{n_subj} subjects)")

    # Show whether by-clip split actually leaks
    train_subjects_byclip = set(subject_ids[cd_tr_byclip].tolist())
    test_subjects_byclip = set(subject_ids[cd_te_byclip].tolist())
    leaked = train_subjects_byclip & test_subjects_byclip
    print(f"[sanity] by-clip split: {len(leaked)}/{len(test_subjects_byclip)} test subjects also in train (LEAKAGE)")

    # DFDC (random split, n=77 either way; subject leakage is irrelevant at this n)
    df_tr, df_te = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    # ── Train probes for both CelebDF splits side-by-side ─────────────────
    rows = []
    for seed in args.seeds:
        print(f"\n[sanity] ====== seed={seed} ======")
        for regime in REGIMES:
            for cdf_split_name, (cd_tr_idx, cd_te_idx) in [
                ("by_clip",    (cd_tr_byclip, cd_te_byclip)),
                ("by_subject", (cd_tr_bysubj, cd_te_bysubj)),
            ]:
                # Build training pool
                if regime == "ff_only":
                    bb_tr = caches["ff"]["backbone"][tr_ff_idx]
                    rppg_tr = caches["ff"]["rppg"][tr_ff_idx]
                    blink_tr = caches["ff"]["blink"][tr_ff_idx]
                    y_tr = caches["ff"]["labels"][tr_ff_idx]
                else:  # mixed
                    pools_bb = [caches["ff"]["backbone"][tr_ff_idx], caches["celebdf"]["backbone"][cd_tr_idx]]
                    pools_rppg = [caches["ff"]["rppg"][tr_ff_idx], caches["celebdf"]["rppg"][cd_tr_idx]]
                    pools_blink = [caches["ff"]["blink"][tr_ff_idx], caches["celebdf"]["blink"][cd_tr_idx]]
                    pools_y = [caches["ff"]["labels"][tr_ff_idx], caches["celebdf"]["labels"][cd_tr_idx]]
                    if df_tr is not None:
                        pools_bb.append(caches["dfdc"]["backbone"][df_tr])
                        pools_rppg.append(caches["dfdc"]["rppg"][df_tr])
                        pools_blink.append(caches["dfdc"]["blink"][df_tr])
                        pools_y.append(caches["dfdc"]["labels"][df_tr])
                    bb_tr = np.concatenate(pools_bb, axis=0)
                    rppg_tr = np.concatenate(pools_rppg, axis=0)
                    blink_tr = np.concatenate(pools_blink, axis=0)
                    y_tr = np.concatenate(pools_y, axis=0)

                bb_vl = caches["ff"]["backbone"][vl_ff_idx]
                rppg_vl = caches["ff"]["rppg"][vl_ff_idx]
                blink_vl = caches["ff"]["blink"][vl_ff_idx]
                y_vl = caches["ff"]["labels"][vl_ff_idx]

                # Test on the appropriate CelebDF test split
                bb_te = caches["celebdf"]["backbone"][cd_te_idx]
                rppg_te = caches["celebdf"]["rppg"][cd_te_idx]
                blink_te = caches["celebdf"]["blink"][cd_te_idx]
                y_te = caches["celebdf"]["labels"][cd_te_idx]

                for variant in VARIANTS:
                    X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
                    X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
                    X_te = make_features(bb_te, rppg_te, blink_te, variant)
                    probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                                epochs=args.epochs, lr=args.lr,
                                                bs=args.batch, seed=seed)
                    scores = predict(probe, X_te, device)
                    row = {
                        "seed": seed, "regime": regime, "variant": variant,
                        "cdf_split": cdf_split_name,
                        "n": int(len(y_te)),
                        "auc": roc_auc(y_te, scores),
                        "ap": average_precision(y_te, scores),
                        "eer": eer(y_te, scores),
                        "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                    }
                    rows.append(row)
                    print(f"  seed={seed} {regime:8s} {variant:18s} {cdf_split_name:11s} "
                          f"AUC={row['auc']:.4f} EER={row['eer']:.4f} TPR@5={row['tpr5']:.3f}")

    # ── Write raw + aggregate ─────────────────────────────────────────────
    header = ["seed","regime","variant","cdf_split","n","auc","ap","eer","tpr5"]
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    print(f"\n[sanity] wrote {out_dir/'results.csv'} ({len(rows)} rows)")

    # Aggregate side-by-side
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["regime"], r["variant"], r["cdf_split"])].append(r)

    print("\n[sanity] ====== HEADLINE COMPARISON: by-clip vs by-subject (mean ± std) ======")
    print(f"{'regime':<9s} {'variant':<18s}  by_clip            by_subject         ΔAUC")
    by_clip_auc = {}
    by_subj_auc = {}
    for (regime, variant, split), rs in sorted(agg.items()):
        aucs = [r["auc"] for r in rs]
        m, s = float(np.mean(aucs)), float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0)
        if split == "by_clip":
            by_clip_auc[(regime, variant)] = (m, s)
        else:
            by_subj_auc[(regime, variant)] = (m, s)

    for regime in REGIMES:
        for variant in VARIANTS:
            cm, cs = by_clip_auc.get((regime, variant), (float("nan"), float("nan")))
            sm, ss = by_subj_auc.get((regime, variant), (float("nan"), float("nan")))
            delta = sm - cm
            print(f"{regime:<9s} {variant:<18s}  {cm:.4f}±{cs:.4f}     {sm:.4f}±{ss:.4f}     {delta:+.4f}")

    # Save aggregate
    agg_rows = []
    for (regime, variant, split), rs in sorted(agg.items()):
        aucs = [r["auc"] for r in rs]
        eers = [r["eer"] for r in rs]
        tpr5s = [r["tpr5"] for r in rs]
        agg_rows.append({
            "regime": regime, "variant": variant, "cdf_split": split, "n": rs[0]["n"],
            "auc_mean": float(np.mean(aucs)),
            "auc_std":  float(np.std(aucs, ddof=1) if len(aucs)>1 else 0),
            "eer_mean": float(np.mean(eers)),
            "eer_std":  float(np.std(eers, ddof=1) if len(eers)>1 else 0),
            "tpr5_mean": float(np.mean(tpr5s)),
            "tpr5_std":  float(np.std(tpr5s, ddof=1) if len(tpr5s)>1 else 0),
        })
    with open(out_dir / "aggregate.csv", "w") as f:
        head = ["regime","variant","cdf_split","n","auc_mean","auc_std","eer_mean","eer_std","tpr5_mean","tpr5_std"]
        f.write(",".join(head) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k]) for k in head) + "\n")
    print(f"[sanity] wrote {out_dir/'aggregate.csv'}")

    # Also dump JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n_unique_subjects_celebdf": int(n_subj),
            "n_train_subjects_byclip_leak": len(leaked),
            "by_clip_split_size": [int(len(cd_tr_byclip)), int(len(cd_te_byclip))],
            "by_subject_split_size": [int(len(cd_tr_bysubj)), int(len(cd_te_bysubj))],
            "agg": agg_rows,
        }, f, indent=2, default=str)
    print(f"[sanity] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Identity-aware CelebDF split sanity check")
    p.add_argument("--celebdf_root", required=True)
    p.add_argument("--cache_dir", required=True,
                   help="Path to existing feat_cache_{b4,clip,dinov2} directory")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    main(p.parse_args())
