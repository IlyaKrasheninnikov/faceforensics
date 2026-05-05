"""
E13 — External-benchmark evaluation (DF40 / Deepfake-Eval-2024 / etc.).

Addresses the professor's correction #5:

    "Add at least one modern external benchmark or strong public baseline if
    time allows. A current reviewer will not be satisfied with FF++, CelebDF,
    and a very small DFDC split alone. Even a limited DF40 or Deepfake-Eval-2024
    evaluation, or a comparison against a public DeepfakeBench/CLIP-based
    baseline, would make the work much more credible."

Two sub-commands:

  extract_external — extract backbone features on a generic external-benchmark
                     directory (mp4 or per-video-folder of frames).  Supports
                     CLIP / DINOv2 / EfficientNet-B4 v13.

  eval_external    — train probe on c23 mixed pool, evaluate on the external
                     benchmark cache.  Reports per-method AUC, AP, EER,
                     TPR@1/5/10%, and bootstrap 95% CIs.

Expected directory layout for `extract_external`:

    <root>/
      <method_name>/
        real/
          *.mp4   OR   <video_id>/*.png
        fake/
          *.mp4   OR   <video_id>/*.png

  e.g. for DF40:
    <root>/
      sd25/                  # Stable Diffusion 2.5 face swaps
        real/   *.mp4
        fake/   *.mp4
      sora/
      faceshifter_df40/
      ...

  Layout auto-detected: if real/ contains .mp4 files → mp4 mode, else
  per-video-folder mode (same as CelebDF).

Usage:

    # Stage A: extract DF40 features (one backbone + one root)
    python w10_stats/external_benchmark.py extract_external \\
        --backbone clip_vitl14 \\
        --external_root "$DF40_ROOT" \\
        --cache_dir /kaggle/working/feat_cache_clip_df40 \\
        --batch_size 4

    # Stage B: train + eval against c23 mixed pool
    python w10_stats/external_benchmark.py eval_external \\
        --external_cache /kaggle/working/feat_cache_clip_df40 \\
        --c23_cache      "$CACHE_CLIP" \\
        --out_dir        /kaggle/working/e13_clip_df40 \\
        --seeds 0 1 42 1337 2024
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import (
    roc_auc, eer, average_precision, tpr_at_fpr,
    train_linear_probe, predict, identity_split_ff, random_split,
    bootstrap_ci_auc,
    VARIANTS, make_features,
)
from optional_experiments import (
    extract_backbone_with_perturbation, PERTURBATIONS,
    _load_mp4_frames_uniform,
)


# ───────────────────────────────────────────────────────────────────────────
# DF40 JSON-manifest scanner (preferred for DF40 — uses ground-truth labels)
# ───────────────────────────────────────────────────────────────────────────

# Map JSON path prefixes → on-disk Kaggle dataset prefixes.
# Order matters: first matching prefix wins.
DF40_PATH_REWRITES = [
    ("deepfakes_detection_datasets/DF40/sadtalker/cdf/", "sadtalker_test/cdf/"),
    ("deepfakes_detection_datasets/DF40/sadtalker/ff/",  "sadtalker_test/ff/"),
    ("deepfakes_detection_datasets/DF40/simswap/cdf/",   "simswap_test/cdf/"),
    ("deepfakes_detection_datasets/DF40/simswap/ff/",    "simswap_test/ff/"),
    ("deepfakes_detection_datasets/Celeb-DF-v2/",        "Celeb-DF-v2/"),
    ("deepfakes_detection_datasets/FaceForensics++/",    "FaceForensics++/"),
]


def _rewrite_df40_path(p, root):
    """Map a JSON-stored path to an on-disk path under the Kaggle dataset root."""
    for src, dst in DF40_PATH_REWRITES:
        if p.startswith(src):
            return os.path.join(root, dst + p[len(src):])
    # Fallback: assume already-relative; just join.
    return os.path.join(root, p)


def scan_df40_json(json_path, dataset_root, split="test", min_frames=8, verbose=True):
    """
    Read a DF40 manifest like sadtalker_cdf.json and return per-clip frame lists
    rebased against `dataset_root`. Filters out clips where < min_frames frames
    exist on disk.

    Returns (paths, labels, methods, src_ids) — same shape as scan_external().
      - paths: list-of-frame-paths per clip (sorted by filename)
      - labels: 0 real / 1 fake (from JSON label field, not folder name)
      - methods: e.g. "sadtalker_cdf"  (method × source bucket)
      - src_ids: clip_id from the JSON (e.g. "00170" or "id0_8meWY...")
    """
    with open(json_path, "r") as f:
        manifest = json.load(f)
    top_keys = list(manifest.keys())
    if len(top_keys) != 1:
        raise ValueError(f"Expected one top-level key in {json_path}, got {top_keys}")
    top = top_keys[0]                      # e.g. "sadtalker_cdf"
    method_tag = top                       # use as the per-clip "method" identifier

    paths, labels, methods, src_ids = [], [], [], []
    n_skip_missing = 0
    n_skip_thin = 0

    for cls_key, clips in manifest[top].items():
        # cls_key is "sadtalker_Real" or "sadtalker_Fake"
        if cls_key.endswith("_Real"):
            label = 0
        elif cls_key.endswith("_Fake"):
            label = 1
        else:
            if verbose:
                print(f"  [df40-scan] WARN unknown class key {cls_key}; skipping")
            continue

        if split not in clips:
            continue
        for clip_id, entry in clips[split].items():
            if not clip_id:                # skip empty placeholder keys
                continue
            frame_rel = entry.get("frames", []) if isinstance(entry, dict) else []
            if not frame_rel:
                continue
            frame_abs = [_rewrite_df40_path(fp, dataset_root) for fp in frame_rel]
            frame_abs = [p for p in frame_abs if os.path.exists(p)]
            if len(frame_abs) == 0:
                n_skip_missing += 1
                continue
            if len(frame_abs) < min_frames:
                n_skip_thin += 1
                continue
            frame_abs.sort()
            paths.append(frame_abs)
            labels.append(label)
            methods.append(method_tag)
            src_ids.append(clip_id)

    if verbose:
        from collections import Counter
        print(f"  [df40-scan] {json_path}")
        print(f"    kept={len(paths)} (real={labels.count(0)} fake={labels.count(1)})")
        print(f"    skipped: all-missing={n_skip_missing} thin(<{min_frames}f)={n_skip_thin}")
    return paths, labels, methods, src_ids


def cmd_probe_df40(args):
    """Dry-run: load JSON(s), rewrite paths, report on-disk hit rate per source bucket.
       Useful to catch dataset layout problems before launching extraction."""
    from collections import defaultdict
    print(f"[df40-probe] dataset_root={args.dataset_root}")
    if not os.path.exists(args.dataset_root):
        print(f"  ERROR: dataset_root not found")
        return
    print(f"[df40-probe] expected subdirs:")
    for sub in ["Celeb-DF-v2", "FaceForensics++",
                "sadtalker_test", "simswap_test"]:
        p = os.path.join(args.dataset_root, sub)
        print(f"    {sub:20s}  exists={os.path.exists(p)}")

    for jp in args.json:
        with open(jp, "r") as f:
            manifest = json.load(f)
        top = list(manifest.keys())[0]
        n_total, n_hit = 0, 0
        first_miss = []
        per_prefix = defaultdict(lambda: [0, 0])  # [hit, total]
        for cls_key, clips in manifest[top].items():
            for clip_id, entry in clips.get(args.split, {}).items():
                if not clip_id:
                    continue
                for fp in entry.get("frames", []):
                    n_total += 1
                    abs_p = _rewrite_df40_path(fp, args.dataset_root)
                    pref = fp.split("/")[1] if "/" in fp else fp
                    if "DF40" in fp:
                        pref = "/".join(fp.split("/")[1:4])  # DF40/<method>/<source>
                    per_prefix[pref][1] += 1
                    if os.path.exists(abs_p):
                        n_hit += 1
                        per_prefix[pref][0] += 1
                    elif len(first_miss) < 3:
                        first_miss.append((fp, abs_p))
        print(f"\n[df40-probe] {jp}")
        print(f"  total frames referenced: {n_total}")
        print(f"  on-disk hits:            {n_hit}  ({100*n_hit/max(n_total,1):.1f}%)")
        for pref, (h, t) in sorted(per_prefix.items()):
            print(f"    {pref:55s}  {h}/{t}  ({100*h/max(t,1):.1f}%)")
        if first_miss:
            print(f"  first misses (showing up to 3):")
            for src, dst in first_miss:
                print(f"    JSON:  {src}")
                print(f"    DISK:  {dst}")


def cmd_extract_df40(args):
    """Extract backbone features for DF40 clips listed in one or more JSON manifests."""
    print(f"[df40-extract] {args.backbone} on {args.dataset_root}")
    print(f"[df40-extract] manifests: {args.json}")

    all_paths, all_labels, all_methods, all_src_ids = [], [], [], []
    for jp in args.json:
        p, l, m, s = scan_df40_json(jp, args.dataset_root,
                                     split=args.split, min_frames=args.min_frames)
        all_paths.extend(p); all_labels.extend(l)
        all_methods.extend(m); all_src_ids.extend(s)

    if not all_paths:
        print(f"[df40-extract] no clips loaded; check --dataset_root and JSON path rewrites")
        return

    from collections import Counter
    print(f"[df40-extract] TOTAL clips: {len(all_paths)}")
    print(f"[df40-extract] per-method counts: {dict(Counter(all_methods))}")
    print(f"[df40-extract] real={all_labels.count(0)} fake={all_labels.count(1)}")

    out_npz = str(Path(args.cache_dir) / "external.npz")
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    extract_backbone_with_perturbation(
        args, args.backbone, all_paths, all_labels, PERTURBATIONS["clean"],
        rppg_cache=None, blink_cache=None,
        cache_key_fn=external_cache_key,
        tag="external",
        out_path=out_npz,
        manips=all_methods, src_ids=all_src_ids,
        b4_ckpt=getattr(args, "b4_ckpt", None),
    )
    print(f"[df40-extract] DONE. Cache at {out_npz}")


# ───────────────────────────────────────────────────────────────────────────
# Generic scanner for external benchmarks (legacy — kept for non-DF40 use)
# ───────────────────────────────────────────────────────────────────────────

def scan_external(root):
    """
    Scan an external benchmark with the layout:
        <root>/<method>/(real|fake)/(*.mp4 | <video_id>/*.png)

    Returns (paths, labels, methods, src_ids).
      - paths: mp4 file path OR list-of-frame-paths (per-video-folder mode)
      - labels: 0 for real, 1 for fake
      - methods: per-clip method name (e.g. "sd25", "faceshifter_df40")
      - src_ids: per-clip source identifier (filename stem)
    """
    root = Path(root)
    if not root.exists():
        return [], [], [], []
    paths, labels, methods, src_ids = [], [], [], []
    method_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for method_dir in method_dirs:
        method_name = method_dir.name
        for cls_name, label in [("real", 0), ("fake", 1)]:
            cls_dir = method_dir / cls_name
            if not cls_dir.exists():
                continue
            mp4s = sorted(cls_dir.glob("*.mp4"))
            if mp4s:
                # mp4 mode
                for mp4 in mp4s:
                    paths.append(str(mp4))
                    labels.append(label)
                    methods.append(method_name)
                    src_ids.append(mp4.stem)
            else:
                # per-video-folder mode
                video_dirs = sorted(d for d in cls_dir.iterdir() if d.is_dir())
                for vd in video_dirs:
                    frames = sorted(
                        os.path.join(vd, f) for f in os.listdir(vd)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    )
                    if frames:
                        paths.append(frames)
                        labels.append(label)
                        methods.append(method_name)
                        src_ids.append(vd.name)
    return paths, labels, methods, src_ids


def external_cache_key(_):
    """External clips have no precomputed physiology — always return (None, None)."""
    return None, None


# ───────────────────────────────────────────────────────────────────────────
# extract_external — wrap optional_experiments.extract_backbone_with_perturbation
# ───────────────────────────────────────────────────────────────────────────

def cmd_extract_external(args):
    print(f"[ext-extract] {args.backbone} on {args.external_root}")
    paths, labels, methods, src_ids = scan_external(args.external_root)
    print(f"[ext-extract] found {len(paths)} clips")
    if not paths:
        print(f"[ext-extract] empty; check the directory layout")
        return

    from collections import Counter
    print(f"[ext-extract] per-method counts: {dict(Counter(methods))}")
    print(f"[ext-extract] real={labels.count(0)} fake={labels.count(1)}")

    extract_backbone_with_perturbation(
        args, args.backbone, paths, labels, PERTURBATIONS["clean"],
        rppg_cache=None, blink_cache=None,
        cache_key_fn=external_cache_key,
        tag="external",
        out_path=str(Path(args.cache_dir) / "external.npz"),
        manips=methods, src_ids=src_ids,
        b4_ckpt=getattr(args, "b4_ckpt", None),
    )
    print(f"[ext-extract] DONE. Cache at {args.cache_dir}/external.npz")


# ───────────────────────────────────────────────────────────────────────────
# eval_external — train mixed probe on c23, evaluate on the external cache
# ───────────────────────────────────────────────────────────────────────────

def cmd_eval_external(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ext-eval] device={device}")

    c23_dir = Path(args.c23_cache)
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = c23_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"  loaded c23 {tag}: n={len(caches[tag]['labels'])}")

    ext_npz = Path(args.external_cache) / "external.npz"
    if not ext_npz.exists():
        print(f"  external cache not found at {ext_npz}; abort")
        return
    ext = {k: v for k, v in np.load(ext_npz, allow_pickle=True).items()}
    print(f"  loaded external: n={len(ext['labels'])} bb_dim={ext['backbone'].shape[1]}")
    if "manip" in ext:
        from collections import Counter
        method_counts = Counter(ext["manip"].tolist())
        print(f"  external methods: {dict(method_counts)}")

    tr_ff_idx, vl_ff_idx, _ = identity_split_ff(caches["ff"], seed=42)
    cd_tr, _ = random_split(len(caches["celebdf"]["labels"]), seed=42)
    df_tr, _ = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    rows = []
    score_storage = {}

    for seed in args.seeds:
        # Build mixed training pool (FF c23 + CDF + DFDC)
        pools_bb = [caches["ff"]["backbone"][tr_ff_idx], caches["celebdf"]["backbone"][cd_tr]]
        pools_rppg = [caches["ff"]["rppg"][tr_ff_idx], caches["celebdf"]["rppg"][cd_tr]]
        pools_blink = [caches["ff"]["blink"][tr_ff_idx], caches["celebdf"]["blink"][cd_tr]]
        pools_y = [caches["ff"]["labels"][tr_ff_idx], caches["celebdf"]["labels"][cd_tr]]
        if df_tr is not None:
            pools_bb.append(caches["dfdc"]["backbone"][df_tr])
            pools_rppg.append(caches["dfdc"]["rppg"][df_tr])
            pools_blink.append(caches["dfdc"]["blink"][df_tr])
            pools_y.append(caches["dfdc"]["labels"][df_tr])
        bb_tr = np.concatenate(pools_bb)
        rppg_tr = np.concatenate(pools_rppg)
        blink_tr = np.concatenate(pools_blink)
        y_tr = np.concatenate(pools_y)

        bb_vl = caches["ff"]["backbone"][vl_ff_idx]
        rppg_vl = caches["ff"]["rppg"][vl_ff_idx]
        blink_vl = caches["ff"]["blink"][vl_ff_idx]
        y_vl = caches["ff"]["labels"][vl_ff_idx]

        for variant in VARIANTS:
            X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
            X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
            probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                        epochs=args.epochs, lr=args.lr,
                                        bs=args.batch, seed=seed)

            # Overall external metrics
            bb_te = ext["backbone"]
            rppg_te = ext["rppg"]
            blink_te = ext["blink"]
            y_te = ext["labels"]
            X_te = make_features(bb_te, rppg_te, blink_te, variant)
            scores = predict(probe, X_te, device)
            score_storage[(seed, variant, "ALL")] = (scores, y_te)
            np.savez(out_dir / "scores" / f"s{seed}_{variant}_ALL.npz",
                     scores=scores, labels=y_te)
            row = {
                "seed": seed, "variant": variant, "method": "ALL",
                "n": int(len(y_te)),
                "auc": roc_auc(y_te, scores),
                "ap": average_precision(y_te, scores),
                "eer": eer(y_te, scores),
                "tpr1": tpr_at_fpr(y_te, scores, 0.01),
                "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                "tpr10": tpr_at_fpr(y_te, scores, 0.10),
            }
            rows.append(row)
            print(f"  s={seed} {variant:18s} {'ALL':<25s} AUC={row['auc']:.4f} TPR@5={row['tpr5']:.3f}")

            # Per-method metrics if 'manip' present
            if "manip" in ext:
                methods = ext["manip"]
                for m in sorted(set(methods.tolist())):
                    idx_m = np.where(methods == m)[0]
                    y_m = y_te[idx_m]; s_m = scores[idx_m]
                    # Pair real/fake for an AUC; if only one class, skip
                    if len(set(y_m.tolist())) < 2:
                        continue
                    score_storage[(seed, variant, m)] = (s_m, y_m)
                    np.savez(out_dir / "scores" / f"s{seed}_{variant}_{m}.npz",
                             scores=s_m, labels=y_m)
                    row_m = {
                        "seed": seed, "variant": variant, "method": m,
                        "n": int(len(y_m)),
                        "auc": roc_auc(y_m, s_m),
                        "ap": average_precision(y_m, s_m),
                        "eer": eer(y_m, s_m),
                        "tpr1": tpr_at_fpr(y_m, s_m, 0.01),
                        "tpr5": tpr_at_fpr(y_m, s_m, 0.05),
                        "tpr10": tpr_at_fpr(y_m, s_m, 0.10),
                    }
                    rows.append(row_m)
                    print(f"  s={seed} {variant:18s} {m:<25s} AUC={row_m['auc']:.4f}")

    # Raw rows
    head = ["seed","variant","method","n","auc","ap","eer","tpr1","tpr5","tpr10"]
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")

    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["method"])].append(r)

    agg_rows = []
    print("\n[ext-eval] AGGREGATE (mean ± std across seeds)")
    print(f"{'variant':<18s} {'method':<25s} {'n':>5s}  AUC             TPR@1%       TPR@5%       TPR@10%")
    for (var, method), rs in sorted(agg.items()):
        def ms(k):
            xs = [r[k] for r in rs]
            return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs)>1 else 0)
        am, asd = ms("auc")
        em, es = ms("eer")
        t1m, t1s = ms("tpr1"); t5m, t5s = ms("tpr5"); t10m, t10s = ms("tpr10")
        agg_rows.append({
            "variant": var, "method": method, "n": rs[0]["n"],
            "auc_mean": am, "auc_std": asd,
            "eer_mean": em, "eer_std": es,
            "tpr1_mean": t1m, "tpr1_std": t1s,
            "tpr5_mean": t5m, "tpr5_std": t5s,
            "tpr10_mean": t10m, "tpr10_std": t10s,
        })
        print(f"{var:<18s} {method:<25s} {rs[0]['n']:>5d}  "
              f"{am:.4f}±{asd:.4f}  "
              f"{t1m:.3f}±{t1s:.3f}  {t5m:.3f}±{t5s:.3f}  {t10m:.3f}±{t10s:.3f}")

    head_agg = list(agg_rows[0].keys()) if agg_rows else []
    with open(out_dir / "aggregate.csv", "w") as f:
        f.write(",".join(head_agg) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in head_agg) + "\n")

    # Bootstrap CIs (seed 0, ALL only — per-method CIs add a lot of clutter)
    boot_rows = []
    for (seed, variant, method), (s, y) in score_storage.items():
        if seed != args.seeds[0] or method != "ALL":
            continue
        ci = bootstrap_ci_auc(y, s, n_boot=args.n_boot, seed=42)
        boot_rows.append({
            "variant": variant, "method": method, "n": int(len(y)),
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

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"agg": agg_rows, "bootstrap": boot_rows, "raw": rows},
                  f, indent=2, default=str)
    print(f"\n[ext-eval] wrote {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="External-benchmark evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    e1 = sub.add_parser("extract_external")
    e1.add_argument("--backbone", choices=["clip_vitl14", "dinov2_vitb14", "efficientnet_b4_v13"], required=True)
    e1.add_argument("--b4_ckpt", default=None)
    e1.add_argument("--external_root", required=True)
    e1.add_argument("--cache_dir", required=True)
    e1.add_argument("--clip_len", type=int, default=16)
    e1.add_argument("--batch_size", type=int, default=4)
    e1.add_argument("--num_workers", type=int, default=2)

    e2 = sub.add_parser("eval_external")
    e2.add_argument("--external_cache", required=True)
    e2.add_argument("--c23_cache", required=True)
    e2.add_argument("--out_dir", required=True)
    e2.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    e2.add_argument("--epochs", type=int, default=20)
    e2.add_argument("--lr", type=float, default=1e-3)
    e2.add_argument("--batch", type=int, default=256)
    e2.add_argument("--n_boot", type=int, default=1000)

    e3 = sub.add_parser("probe_df40",
        help="Dry-run: load DF40 JSON(s), check on-disk hit rate. No extraction.")
    e3.add_argument("--dataset_root", required=True,
        help="Kaggle dataset root, e.g. /kaggle/input/dataset-df40-slice/dataset_df40_2")
    e3.add_argument("--json", nargs="+", required=True,
        help="One or more DF40 manifest JSONs (sadtalker_cdf.json, simswap_ff.json, ...)")
    e3.add_argument("--split", default="test")

    e4 = sub.add_parser("extract_df40",
        help="Extract backbone features for DF40 clips listed in JSON manifest(s).")
    e4.add_argument("--backbone",
        choices=["clip_vitl14", "dinov2_vitb14", "efficientnet_b4_v13"], required=True)
    e4.add_argument("--b4_ckpt", default=None)
    e4.add_argument("--dataset_root", required=True)
    e4.add_argument("--json", nargs="+", required=True)
    e4.add_argument("--split", default="test")
    e4.add_argument("--min_frames", type=int, default=8,
        help="Skip clips with fewer than this many on-disk frames")
    e4.add_argument("--cache_dir", required=True)
    e4.add_argument("--clip_len", type=int, default=16)
    e4.add_argument("--batch_size", type=int, default=4)
    e4.add_argument("--num_workers", type=int, default=2)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "extract_external":
        cmd_extract_external(args)
    elif args.cmd == "eval_external":
        cmd_eval_external(args)
    elif args.cmd == "probe_df40":
        cmd_probe_df40(args)
    elif args.cmd == "extract_df40":
        cmd_extract_df40(args)
