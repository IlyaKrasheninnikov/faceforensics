"""Emit per-clip subject IDs for the strict-LODO FF and DFDC test partitions.

The strict-LODO score NPZs in
``artifacts/csv/strict_lodo_bundle/e14_lodo_strict_*/scores/`` store only
``scores`` and ``labels`` per clip. To run a *subject-cluster* paired
bootstrap on the FF-as-target and DFDC-as-target strict-LODO partitions
(broadening the audit beyond CelebDF), we need the per-clip subject id
in the SAME ORDER as those cached scores.

This script re-derives the exact test partitions used by
``lodo_probe_strict.py`` from the c23 feature caches and writes a CSV
of (clip_idx, subject_id, label) per target, in cached-score order.

It must be run where the c23 feature caches live (e.g. the Kaggle
dataset), because those caches are not redistributed in the artefact
bundle. The output CSVs are tiny (a few KB) and are then placed under
``artifacts/csv/strict_lodo_bundle/`` so the local bootstrap can read
them.

Partition logic (identical to lodo_probe_strict.py):
  - test_ff:   ff identity-aware test split (seed 42); subject = src_id.
  - test_dfdc: dfdc random 80/20 split (seed 42), test = last 20%;
               DFDC has NO subject field, so subject_id == clip index
               (each clip is its own cluster -> clip-level bootstrap).

Usage (on Kaggle, same paths as the E14 strict-LODO run that produced
the cached scores -- see p3_physio/w10_stats/KAGGLE_RUN_E14_STRICT_LODO.md).
The CLIP, B4, and DINOv2 caches share identical labels/src_id ordering
per dataset, so any one of the three feat_cache_* dirs works; CLIP is
the canonical choice:

    CACHES_ROOT=/kaggle/input/datasets/goodboyxdd/feat-caches-b4-clip-dinov2
    python scripts/emit_target_subject_ids.py \\
        --cache_dir "$CACHES_ROOT/feat_cache_clip" \\
        --out_dir   artifacts/csv/strict_lodo_bundle

The cache_dir must contain ff.npz, celebdf.npz, dfdc.npz (it does:
the E14 probe loads exactly those filenames from feat_cache_clip/).

Produces:
    strict_lodo_bundle/test_ff_subject_ids.csv
    strict_lodo_bundle/test_dfdc_subject_ids.csv

Each row: clip_order_idx,subject_id,label
where clip_order_idx is the position in the cached score array.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np


def identity_split_ff(cache, seed=42):
    """Identity-aware 80/10/10 split on FF++ (matches multiseed_and_stats)."""
    src_ids = cache["src_id"]
    unique_ids = sorted(set(src_ids.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n_ids = len(unique_ids)
    n_tr = int(n_ids * 0.8)
    n_vl = int(n_ids * 0.1)
    tr_set = set(unique_ids[:n_tr])
    vl_set = set(unique_ids[n_tr:n_tr + n_vl])
    tr_idx, vl_idx, te_idx = [], [], []
    for i, sid in enumerate(src_ids):
        if sid in tr_set:
            tr_idx.append(i)
        elif sid in vl_set:
            vl_idx.append(i)
        else:
            te_idx.append(i)
    return np.array(tr_idx), np.array(vl_idx), np.array(te_idx)


def dfdc_test_split(n, seed=42):
    """DFDC random 80/20 split; test = last 20% (matches lodo_probe_strict)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(n * 0.8)
    return idx[k:]


def _subject_field(cache):
    """Return the per-clip subject array, or None if absent."""
    for key in ("src_id", "subject_id", "subject", "id"):
        if key in cache:
            return np.asarray(cache[key])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, type=Path,
                    help="dir holding ff.npz, dfdc.npz (c23 strict-LODO caches)")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── FF as target ──
    ff_path = args.cache_dir / "ff.npz"
    if ff_path.exists():
        ff = {k: v for k, v in np.load(ff_path, allow_pickle=True).items()}
        _, _, ff_te = identity_split_ff(ff, seed=42)
        ff_subj = _subject_field(ff)
        if ff_subj is None:
            raise RuntimeError("FF cache has no subject field (src_id)")
        out_ff = args.out_dir / "test_ff_subject_ids.csv"
        with open(out_ff, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_order_idx", "subject_id", "label"])
            for order_idx, clip_idx in enumerate(ff_te):
                w.writerow([order_idx,
                            f"ff_{ff_subj[clip_idx]}",
                            int(ff["labels"][clip_idx])])
        print(f"[emit] wrote {out_ff}  n={len(ff_te)}  "
              f"n_subjects={len(set(ff_subj[ff_te].tolist()))}")
    else:
        print(f"[emit] SKIP FF: {ff_path} not found")

    # ── DFDC as target ──
    dfdc_path = args.cache_dir / "dfdc.npz"
    if dfdc_path.exists():
        dfdc = {k: v for k, v in np.load(dfdc_path, allow_pickle=True).items()}
        n = len(dfdc["labels"])
        df_te = dfdc_test_split(n, seed=42)
        df_subj = _subject_field(dfdc)
        out_df = args.out_dir / "test_dfdc_subject_ids.csv"
        with open(out_df, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_order_idx", "subject_id", "label"])
            for order_idx, clip_idx in enumerate(df_te):
                if df_subj is not None:
                    sid = f"dfdc_{df_subj[clip_idx]}"
                else:
                    # No subject field: each clip is its own cluster.
                    sid = f"dfdc_clip_{clip_idx}"
                w.writerow([order_idx, sid, int(dfdc["labels"][clip_idx])])
        has_subj = df_subj is not None
        n_subj = (len(set(df_subj[df_te].tolist())) if has_subj else len(df_te))
        print(f"[emit] wrote {out_df}  n={len(df_te)}  "
              f"has_subject_field={has_subj}  n_clusters={n_subj}")
    else:
        print(f"[emit] SKIP DFDC: {dfdc_path} not found")

    print("[emit] done. Place the CSVs under "
          "artifacts/csv/strict_lodo_bundle/ and run "
          "scripts/broaden_lodo_bootstrap.py locally.")


if __name__ == "__main__":
    main()
