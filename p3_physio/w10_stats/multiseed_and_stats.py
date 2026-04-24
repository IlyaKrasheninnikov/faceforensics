"""
E1 + E2: Multi-seed linear probe + bootstrap CIs + paired significance tests.

Closes gaps G1 (multiple seeds), G2 (DeLong/McNemar paired
tests) and G3 (bootstrap CIs on every headline number).

Two-stage design:

  STAGE 1 ("extract") — expensive, run ONCE:
      Loads FF++ / CelebDF / DFDC, forwards through the frozen v13 backbone,
      loads cached rPPG (12-d) and blink (16-d) per clip, and writes a single
      npz per dataset to --cache_dir. Takes 30-60 min on a T4.

  STAGE 2 ("probe") — cheap, run MANY TIMES:
      Loads the npz caches, trains 4 variants x 2 regimes (FF-only and mixed)
      x 5 seeds = 40 linear probes, each <30s. Writes scores and aggregated
      statistics to --out_dir. Takes ~20-30 min end-to-end.

  STAGE 3 ("stats") — cheap, runs automatically after "probe":
      Stratified bootstrap (n=1000) CIs on every (variant, regime, split)
      cell. DeLong paired test on variant pairs per dataset per seed.
      McNemar test at Youden-optimal thresholds.

Usage on Kaggle (one cell):

    # Stage 1 (once)
    !python p3_physio/w10_stats/multiseed_and_stats.py extract \\
        --ff_root         /kaggle/input/faceforensics-c23-processed/ff/ff++/frames \\
        --celebdf_root    /kaggle/input/celebdfv2/crop \\
        --dfdc_faces_root /kaggle/input/dfdc-faces/dfdc-faces \\
        --resume_ckpt     /kaggle/input/v13-best-chkpnt/physio_rppg_v13_best.pt \\
        --rppg_cache      /kaggle/input/rppg-v2-300/rppg_v2_300 \\
        --blink_cache     /kaggle/input/blink-v1/blink \\
        --cache_dir       /kaggle/working/feat_cache

    # Stage 2 + 3 (fast; re-run after any probe change)
    !python p3_physio/w10_stats/multiseed_and_stats.py probe \\
        --cache_dir /kaggle/working/feat_cache \\
        --out_dir   /kaggle/working/w10_stats \\
        --seeds 0 1 42 1337 2024

All outputs land in --out_dir and should be copy-pasted back to the
assistant along with the stdout log.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: one-time feature extraction.  Mirrors w4_full_train/train_mixed_probe_biosig.
# ─────────────────────────────────────────────────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0, "Deepfakes": 1, "Face2Face": 1,
    "FaceSwap": 1, "NeuralTextures": 1, "FaceShifter": 1,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def scan_ff(ff_root):
    ff_root = Path(ff_root)
    dirs, labels, manips, src_ids = [], [], [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        mdir = ff_root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                dirs.append(str(sd))
                labels.append(label)
                manips.append(manip)
                src_ids.append(sd.name.split("_")[0])
    return dirs, labels, manips, src_ids


def scan_celebdf(root):
    root = Path(root)
    dirs, labels = [], []
    for split in ["Test", "Train"]:
        for lname, label in [("real", 0), ("fake", 1)]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            for sd in sorted(d for d in ldir.iterdir() if d.is_dir()):
                if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                    dirs.append(str(sd))
                    labels.append(label)
    return dirs, labels


def scan_dfdc_faces(root):
    root = Path(root)
    dirs, labels = [], []
    for split in ["validation", "train"]:
        for lname, label in [("real", 0), ("fake", 1)]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            vid_to_files = {}
            for f in ldir.iterdir():
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    stem = f.stem
                    vid_id = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem.split("_")[0]
                    vid_to_files.setdefault(vid_id, []).append(str(f))
            for vid_id, files in sorted(vid_to_files.items()):
                dirs.append(files)
                labels.append(label)
    return dirs, labels


def do_extract(args):
    """Stage 1: extract backbone + rPPG + blink features per dataset, save to npz."""
    import cv2
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from w2_model.model import PhysioNet, ModelConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}")

    class ClipDataset(Dataset):
        """
        Cache-key functions receive the raw entry from video_dirs (a str path for
        FF++/CelebDF, a list-of-files for DFDC) and must return (class_name, video_id)
        — the two subfolders that form <cache_root>/<class>/<video_id>/rppg_v2_feat.npy.

        The default (ff_cache_key) reproduces the FF++ layout used by the original
        rPPG/blink extractor.
        """

        def __init__(self, video_dirs, labels, rppg_cache, rppg_dim, blink_cache,
                     cache_key_fn=None):
            self.labels = labels
            self.rppg_cache = Path(rppg_cache) if rppg_cache else None
            self.rppg_dim = rppg_dim
            self.blink_cache = Path(blink_cache) if blink_cache else None
            self.video_dirs_raw = video_dirs
            self.cache_key_fn = cache_key_fn or self._ff_cache_key
            self.frame_paths = []
            for vd in video_dirs:
                if isinstance(vd, list):
                    self.frame_paths.append(sorted(vd))
                else:
                    self.frame_paths.append(sorted(
                        os.path.join(vd, f) for f in os.listdir(vd)
                        if f.endswith((".png", ".jpg", ".jpeg"))
                    ))

        @staticmethod
        def _ff_cache_key(vd_raw):
            """FF++ layout: <root>/<manip>/<video_id> → (manip, video_id)."""
            vp = Path(vd_raw)
            return vp.parent.name, vp.name

        def __len__(self):
            return len(self.frame_paths)

        def __getitem__(self, idx):
            label = self.labels[idx]
            frames = self.frame_paths[idx]
            n = len(frames)
            if n == 0:
                clip = np.zeros((args.clip_len, args.img_size, args.img_size, 3), dtype=np.float32)
            else:
                start = max(0, n - args.clip_len) // 2
                indices = [(start + i) % n for i in range(args.clip_len)]
                imgs = []
                for fi in indices:
                    img = cv2.imread(frames[fi])
                    if img is None:
                        img = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (args.img_size, args.img_size))
                    imgs.append(img)
                clip = np.stack(imgs, axis=0).astype(np.float32) / 255.0
            clip = (clip - IMAGENET_MEAN) / IMAGENET_STD

            vd_raw = self.video_dirs_raw[idx]
            class_name, video_id = self.cache_key_fn(vd_raw)

            rppg_feat = np.zeros(self.rppg_dim, dtype=np.float32)
            if self.rppg_cache is not None and class_name and video_id:
                cp = self.rppg_cache / class_name / video_id / "rppg_v2_feat.npy"
                if cp.exists():
                    loaded = np.load(str(cp)).astype(np.float32)
                    if len(loaded) <= self.rppg_dim:
                        rppg_feat[: len(loaded)] = loaded

            blink_feat = np.zeros(16, dtype=np.float32)
            if self.blink_cache is not None and class_name and video_id:
                bp = self.blink_cache / class_name / video_id / "blink_feat.npy"
                if bp.exists():
                    loaded = np.load(str(bp)).astype(np.float32)
                    if len(loaded) == 16:
                        blink_feat = loaded

            return {
                "frames": torch.from_numpy(clip).permute(0, 3, 1, 2).float(),
                "label": torch.tensor(label, dtype=torch.float32),
                "rppg": torch.from_numpy(rppg_feat),
                "blink": torch.from_numpy(blink_feat),
            }

    # ── Cache key functions per dataset layout (must match E3 extractor layout) ──
    def celebdf_cache_key(vd_raw):
        # vd_raw: .../celebdfv2/crop/<split>/<class>/<video_id>
        # E3 cache: <cache>/<split>_<class>/<video_id>
        vp = Path(vd_raw)
        lname = vp.parent.name       # real/fake
        split = vp.parent.parent.name  # Train/Test
        return f"{split}_{lname}", vp.name

    def dfdc_cache_key(vd_raw):
        # vd_raw: list of flat file paths like .../<split>/<class>/<vid>_<frame>.jpg
        # E3 cache: <cache>/<split>_<class>/<video_id>
        if not isinstance(vd_raw, list) or not vd_raw:
            return None, None
        first = Path(vd_raw[0])
        lname = first.parent.name
        split = first.parent.parent.name
        stem = first.stem
        vid_id = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem.split("_")[0]
        return f"{split}_{lname}", vid_id

    cfg = ModelConfig(
        backbone="efficientnet_b4", backbone_pretrained=False,
        temporal_model="mean", temporal_dim=0,
        clip_len=args.clip_len, img_size=args.img_size, dropout=0.0,
        use_physio_fusion=False, use_pulse_head=False,
        use_blink_head=False, use_motion_model=False,
    )
    model = PhysioNet(cfg).to(device)
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        bb_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(bb_state, strict=False)
        print(f"[extract] loaded {len(bb_state)} backbone tensors from {args.resume_ckpt}")
    else:
        print(f"[extract] WARNING: no checkpoint at {args.resume_ckpt}; using random init")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    @torch.no_grad()
    def run_one(dirs, labels, tag, rppg_cache, blink_cache, manips=None, src_ids=None,
                cache_key_fn=None):
        if len(dirs) == 0:
            print(f"[extract] {tag}: EMPTY, skipping")
            return
        ds = ClipDataset(dirs, labels, rppg_cache, args.rppg_dim, blink_cache,
                         cache_key_fn=cache_key_fn)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        bb_list, rppg_list, blink_list, label_list = [], [], [], []
        t0 = time.time()
        for batch in tqdm(dl, desc=tag, leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ff = model.frame_encoder(frames)
                pooled = ff.mean(dim=1)
            bb_list.append(pooled.float().cpu().numpy())
            rppg_list.append(batch["rppg"].numpy())
            blink_list.append(batch["blink"].numpy())
            label_list.append(batch["label"].numpy())
        bb = np.concatenate(bb_list, axis=0)
        bb = np.nan_to_num(bb, nan=0.0, posinf=0.0, neginf=0.0)
        rppg = np.concatenate(rppg_list, axis=0)
        blink = np.concatenate(blink_list, axis=0)
        lbls = np.concatenate(label_list, axis=0)

        out = {
            "backbone": bb, "rppg": rppg, "blink": blink, "labels": lbls,
        }
        if manips is not None:
            out["manip"] = np.array(manips)
        if src_ids is not None:
            out["src_id"] = np.array(src_ids)

        rppg_hit = (np.abs(rppg).sum(1) > 0).mean() * 100
        blink_hit = (np.abs(blink).sum(1) > 0).mean() * 100
        print(f"[extract] {tag}: n={len(lbls)} real={int((lbls==0).sum())} fake={int((lbls==1).sum())} "
              f"rppg_hit={rppg_hit:.0f}% blink_hit={blink_hit:.0f}% "
              f"time={time.time()-t0:.1f}s")

        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(args.cache_dir) / f"{tag}.npz", **out)

    # FF++ with manipulation labels + source IDs for identity-aware splits
    print("\n[extract] scanning FF++")
    ff_dirs, ff_labels, ff_manips, ff_src = scan_ff(args.ff_root)
    print(f"[extract] FF++: {len(ff_dirs)} videos")
    run_one(ff_dirs, ff_labels, "ff", args.rppg_cache, args.blink_cache,
            manips=ff_manips, src_ids=ff_src)

    # CelebDF (pass real rPPG/blink caches from E3 if available; otherwise zero-pad)
    if args.celebdf_root:
        print("\n[extract] scanning CelebDF")
        cd_dirs, cd_labels = scan_celebdf(args.celebdf_root)
        print(f"[extract] CelebDF: {len(cd_dirs)} videos")
        run_one(cd_dirs, cd_labels, "celebdf",
                args.celebdf_rppg_cache, args.celebdf_blink_cache,
                cache_key_fn=celebdf_cache_key)

    # DFDC
    if args.dfdc_faces_root:
        print("\n[extract] scanning DFDC")
        df_dirs, df_labels = scan_dfdc_faces(args.dfdc_faces_root)
        print(f"[extract] DFDC: {len(df_dirs)} groups")
        run_one(df_dirs, df_labels, "dfdc",
                args.dfdc_rppg_cache, args.dfdc_blink_cache,
                cache_key_fn=dfdc_cache_key)

    print(f"\n[extract] DONE. Caches in {args.cache_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: multi-seed linear probe
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]
REGIMES = ["ff_only", "mixed"]


def make_features(bb, rppg, blink, variant):
    if variant == "backbone_only":
        return bb
    elif variant == "backbone+rppg":
        return np.concatenate([bb, rppg], axis=1)
    elif variant == "backbone+blink":
        return np.concatenate([bb, blink], axis=1)
    elif variant == "full_fusion":
        return np.concatenate([bb, rppg, blink], axis=1)
    else:
        raise ValueError(variant)


def identity_split_ff(cache, seed=42):
    """Identity-aware 80/10/10 split on FF++ (matches w5_ablation)."""
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


def random_split(n, seed=42, frac=0.8):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(n * frac)
    return idx[:k], idx[k:]


def train_linear_probe(X_tr, y_tr, X_vl, y_vl, device, epochs=20, lr=1e-3, bs=256, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_vl).float().to(device)
    yv = torch.from_numpy(y_vl.astype(np.float32)).to(device)

    probe = nn.Linear(Xt.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_loss = float("inf")
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(Xt.shape[0], device=device)
        for i in range(0, perm.numel(), bs):
            idx = perm[i:i + bs]
            logits = probe(Xt[idx]).squeeze(-1)
            loss = crit(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            vl_logits = probe(Xv).squeeze(-1)
            vl_loss = crit(vl_logits, yv).item()
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()
    return probe


def predict(probe, X, device):
    import torch
    with torch.no_grad():
        logits = probe(torch.from_numpy(X).float().to(device)).squeeze(-1)
        probs = torch.sigmoid(logits.clamp(-20, 20)).cpu().numpy()
    return np.nan_to_num(probs, nan=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: metrics, CIs, paired tests
# ─────────────────────────────────────────────────────────────────────────────

def roc_auc(y, s):
    """AUC via rank formula — no sklearn dependency for the stats module."""
    y = np.asarray(y)
    s = np.asarray(s)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    unique, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    rank_sum = np.zeros_like(unique, dtype=np.float64)
    np.add.at(rank_sum, inv, ranks)
    avg_ranks = rank_sum / counts
    ranks = avg_ranks[inv]
    return float((ranks[pos].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def eer(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(-s)
    y_sorted = y[order]
    P = (y == 1).sum()
    N = (y == 0).sum()
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / P
    fpr = fp / N
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def tpr_at_fpr(y, s, target_fpr):
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(-s)
    y_sorted = y[order]
    P = (y == 1).sum()
    N = (y == 0).sum()
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / P
    fpr = fp / N
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def average_precision(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(-s)
    y_sorted = y[order]
    P = (y == 1).sum()
    if P == 0:
        return float("nan")
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    recall = tp / P
    recall_diff = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * recall_diff).sum())


def bootstrap_ci_auc(y, s, n_boot=1000, seed=42, target_fprs=(0.01, 0.05, 0.10)):
    """Stratified bootstrap: resample positives and negatives independently."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return {"auc_ci": (float("nan"),) * 2, "ap_ci": (float("nan"),) * 2,
                "tpr_ci": {f: (float("nan"),) * 2 for f in target_fprs}}
    aucs, aps = [], []
    tprs = {f: [] for f in target_fprs}
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        yy, ss = y[idx], s[idx]
        aucs.append(roc_auc(yy, ss))
        aps.append(average_precision(yy, ss))
        for f in target_fprs:
            tprs[f].append(tpr_at_fpr(yy, ss, f))
    return {
        "auc_ci": (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))),
        "ap_ci": (float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))),
        "tpr_ci": {f: (float(np.percentile(tprs[f], 2.5)),
                       float(np.percentile(tprs[f], 97.5))) for f in target_fprs},
    }


# ─── DeLong's paired AUC test (Sun & Xu 2014 fast algorithm) ────────────────

def _midrank(x):
    """Mid-ranks used by the fast DeLong algorithm."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(pos_scores, neg_scores):
    """
    Fast DeLong covariance for a pair of prediction vectors.
    Returns (aucs, covariance matrix) for 2 classifiers.

    pos_scores: (2, n_pos), neg_scores: (2, n_neg).
    """
    m = pos_scores.shape[1]
    n = neg_scores.shape[1]
    k = pos_scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos_scores[r])
        ty[r] = _midrank(neg_scores[r])
        tz[r] = _midrank(np.concatenate([pos_scores[r], neg_scores[r]]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = sx.reshape(1, 1); sy = sy.reshape(1, 1)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true, scores_a, scores_b):
    """Two-sided DeLong paired AUC test. Returns (auc_a, auc_b, z, p)."""
    y_true = np.asarray(y_true)
    scores = np.vstack([np.asarray(scores_a), np.asarray(scores_b)])
    pos = scores[:, y_true == 1]
    neg = scores[:, y_true == 0]
    aucs, cov = _fast_delong(pos, neg)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / math.sqrt(var_diff)
    # two-sided p from normal
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def mcnemar_test(y_true, scores_a, scores_b, threshold=0.5):
    """McNemar paired test on binary decisions at a fixed threshold."""
    y_true = np.asarray(y_true)
    a = (np.asarray(scores_a) >= threshold).astype(int)
    b = (np.asarray(scores_b) >= threshold).astype(int)
    correct_a = (a == y_true).astype(int)
    correct_b = (b == y_true).astype(int)
    b_wins = int(((correct_b == 1) & (correct_a == 0)).sum())
    a_wins = int(((correct_a == 1) & (correct_b == 0)).sum())
    if a_wins + b_wins == 0:
        return {"a_wins": a_wins, "b_wins": b_wins, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(a_wins - b_wins) - 1) ** 2 / (a_wins + b_wins)
    # one degree of freedom chi-square survival fn ≈ erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"a_wins": a_wins, "b_wins": b_wins, "chi2": float(chi2), "p": float(p)}


def youden_threshold(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(-s)
    y_sorted = y[order]; s_sorted = s[order]
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0:
        return 0.5
    tp = np.cumsum(y_sorted == 1); fp = np.cumsum(y_sorted == 0)
    tpr = tp / P; fpr = fp / N
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(s_sorted[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2/3 driver
# ─────────────────────────────────────────────────────────────────────────────

def do_probe(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    # Load caches
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        fpath = cache_dir / f"{tag}.npz"
        if fpath.exists():
            caches[tag] = {k: v for k, v in np.load(fpath, allow_pickle=True).items()}
            print(f"[probe] loaded {tag}: n={len(caches[tag]['labels'])} "
                  f"bb_dim={caches[tag]['backbone'].shape[1]} "
                  f"rppg_dim={caches[tag]['rppg'].shape[1]} "
                  f"blink_dim={caches[tag]['blink'].shape[1]}")
        else:
            print(f"[probe] MISSING {fpath}")

    if "ff" not in caches:
        print("[probe] FF++ cache is required, aborting.")
        return

    # Identity-aware FF++ split (fixed split seed 42; probe seeds are different)
    tr_idx, vl_idx, te_idx = identity_split_ff(caches["ff"], seed=42)
    print(f"[probe] FF++ split: train={len(tr_idx)} val={len(vl_idx)} test={len(te_idx)}")

    # Per-dataset 80/20 train/test splits (match train_mixed_probe_biosig)
    cd_splits = random_split(len(caches["celebdf"]["labels"]), seed=42) if "celebdf" in caches else (None, None)
    df_splits = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    all_rows = []       # per (seed, variant, regime, split) metrics
    all_scores = {}     # for paired tests across seeds

    for seed in args.seeds:
        print(f"\n[probe] ====== seed={seed} ======")
        for regime in REGIMES:
            # Build training pool
            if regime == "ff_only":
                bb_tr = caches["ff"]["backbone"][tr_idx]
                rppg_tr = caches["ff"]["rppg"][tr_idx]
                blink_tr = caches["ff"]["blink"][tr_idx]
                y_tr = caches["ff"]["labels"][tr_idx]
                bb_vl = caches["ff"]["backbone"][vl_idx]
                rppg_vl = caches["ff"]["rppg"][vl_idx]
                blink_vl = caches["ff"]["blink"][vl_idx]
                y_vl = caches["ff"]["labels"][vl_idx]
            else:  # mixed
                pools_bb = [caches["ff"]["backbone"][tr_idx]]
                pools_rppg = [caches["ff"]["rppg"][tr_idx]]
                pools_blink = [caches["ff"]["blink"][tr_idx]]
                pools_y = [caches["ff"]["labels"][tr_idx]]
                if cd_splits[0] is not None:
                    tr, _ = cd_splits
                    pools_bb.append(caches["celebdf"]["backbone"][tr])
                    pools_rppg.append(caches["celebdf"]["rppg"][tr])
                    pools_blink.append(caches["celebdf"]["blink"][tr])
                    pools_y.append(caches["celebdf"]["labels"][tr])
                if df_splits[0] is not None:
                    tr, _ = df_splits
                    pools_bb.append(caches["dfdc"]["backbone"][tr])
                    pools_rppg.append(caches["dfdc"]["rppg"][tr])
                    pools_blink.append(caches["dfdc"]["blink"][tr])
                    pools_y.append(caches["dfdc"]["labels"][tr])
                bb_tr = np.concatenate(pools_bb, axis=0)
                rppg_tr = np.concatenate(pools_rppg, axis=0)
                blink_tr = np.concatenate(pools_blink, axis=0)
                y_tr = np.concatenate(pools_y, axis=0)
                # Validate on FF++ val split (unchanged) for early stopping
                bb_vl = caches["ff"]["backbone"][vl_idx]
                rppg_vl = caches["ff"]["rppg"][vl_idx]
                blink_vl = caches["ff"]["blink"][vl_idx]
                y_vl = caches["ff"]["labels"][vl_idx]

            for variant in VARIANTS:
                X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
                X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
                probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                            epochs=args.epochs, lr=args.lr,
                                            bs=args.batch, seed=seed)

                # Test on each split
                for split_name, tags in [
                    ("ff_test", ("ff", te_idx)),
                    ("celebdf_test", ("celebdf", cd_splits[1] if cd_splits[0] is not None else None)),
                    ("dfdc_test", ("dfdc", df_splits[1] if df_splits[0] is not None else None)),
                ]:
                    tag, idx = tags
                    if tag not in caches or idx is None:
                        continue
                    bb_te = caches[tag]["backbone"][idx]
                    rppg_te = caches[tag]["rppg"][idx]
                    blink_te = caches[tag]["blink"][idx]
                    y_te = caches[tag]["labels"][idx]
                    X_te = make_features(bb_te, rppg_te, blink_te, variant)
                    scores = predict(probe, X_te, device)

                    # Save scores for stats stage
                    np.savez(
                        out_dir / "scores" / f"s{seed}_{regime}_{variant}_{split_name}.npz",
                        scores=scores, labels=y_te,
                    )
                    key = (seed, regime, variant, split_name)
                    all_scores[key] = (scores, y_te)

                    row = {
                        "seed": seed, "regime": regime, "variant": variant, "split": split_name,
                        "n": int(len(y_te)),
                        "auc": roc_auc(y_te, scores),
                        "ap": average_precision(y_te, scores),
                        "eer": eer(y_te, scores),
                        "tpr_at_fpr01": tpr_at_fpr(y_te, scores, 0.01),
                        "tpr_at_fpr05": tpr_at_fpr(y_te, scores, 0.05),
                        "tpr_at_fpr10": tpr_at_fpr(y_te, scores, 0.10),
                    }
                    all_rows.append(row)
                    print(f"  seed={seed} {regime:8s} {variant:18s} {split_name:14s} "
                          f"AUC={row['auc']:.4f}  EER={row['eer']:.4f}  "
                          f"TPR@5={row['tpr_at_fpr05']:.3f}")

    # ─── Write raw rows ─────────────────────────────────────────────────────
    header = ["seed", "regime", "variant", "split", "n", "auc", "ap", "eer",
              "tpr_at_fpr01", "tpr_at_fpr05", "tpr_at_fpr10"]
    with open(out_dir / "multiseed_results.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    print(f"\n[probe] wrote {out_dir/'multiseed_results.csv'} ({len(all_rows)} rows)")

    # ─── Aggregate mean±std ─────────────────────────────────────────────────
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        k = (r["regime"], r["variant"], r["split"])
        agg[k].append(r)

    agg_header = ["regime", "variant", "split", "n",
                  "auc_mean", "auc_std", "ap_mean", "ap_std",
                  "eer_mean", "eer_std",
                  "tpr5_mean", "tpr5_std",
                  "tpr10_mean", "tpr10_std"]
    agg_rows = []
    for (regime, variant, split), rs in sorted(agg.items()):
        aucs = [r["auc"] for r in rs]
        aps = [r["ap"] for r in rs]
        eers = [r["eer"] for r in rs]
        t5s = [r["tpr_at_fpr05"] for r in rs]
        t10s = [r["tpr_at_fpr10"] for r in rs]
        agg_rows.append({
            "regime": regime, "variant": variant, "split": split,
            "n": rs[0]["n"],
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0),
            "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps, ddof=1) if len(aps) > 1 else 0.0),
            "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers, ddof=1) if len(eers) > 1 else 0.0),
            "tpr5_mean": float(np.mean(t5s)), "tpr5_std": float(np.std(t5s, ddof=1) if len(t5s) > 1 else 0.0),
            "tpr10_mean": float(np.mean(t10s)), "tpr10_std": float(np.std(t10s, ddof=1) if len(t10s) > 1 else 0.0),
        })

    with open(out_dir / "multiseed_aggregate.csv", "w") as f:
        f.write(",".join(agg_header) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in agg_header) + "\n")
    print(f"[probe] wrote {out_dir/'multiseed_aggregate.csv'}")

    # Print summary table
    print("\n[probe] ====== AGGREGATE SUMMARY (mean±std across seeds) ======")
    print(f"{'regime':<9s} {'variant':<18s} {'split':<14s}  AUC             EER            TPR@5%")
    for r in agg_rows:
        print(f"{r['regime']:<9s} {r['variant']:<18s} {r['split']:<14s} "
              f" {r['auc_mean']:.4f}±{r['auc_std']:.4f} "
              f" {r['eer_mean']:.4f}±{r['eer_std']:.4f} "
              f" {r['tpr5_mean']:.3f}±{r['tpr5_std']:.3f}")

    # ─── Bootstrap CIs (one per seed=0 to keep cost bounded) ────────────────
    print("\n[probe] computing bootstrap CIs (n=1000) for seed 0 ...")
    boot_rows = []
    for (seed, regime, variant, split), (s, y) in all_scores.items():
        if seed != args.seeds[0]:
            continue
        ci = bootstrap_ci_auc(y, s, n_boot=args.n_boot, seed=42)
        boot_rows.append({
            "regime": regime, "variant": variant, "split": split, "n": len(y),
            "auc": roc_auc(y, s),
            "auc_ci_lo": ci["auc_ci"][0], "auc_ci_hi": ci["auc_ci"][1],
            "ap_ci_lo": ci["ap_ci"][0], "ap_ci_hi": ci["ap_ci"][1],
            "tpr5_ci_lo": ci["tpr_ci"][0.05][0], "tpr5_ci_hi": ci["tpr_ci"][0.05][1],
            "tpr10_ci_lo": ci["tpr_ci"][0.10][0], "tpr10_ci_hi": ci["tpr_ci"][0.10][1],
        })

    bh = ["regime", "variant", "split", "n", "auc", "auc_ci_lo", "auc_ci_hi",
          "ap_ci_lo", "ap_ci_hi", "tpr5_ci_lo", "tpr5_ci_hi", "tpr10_ci_lo", "tpr10_ci_hi"]
    with open(out_dir / "bootstrap_cis.csv", "w") as f:
        f.write(",".join(bh) + "\n")
        for r in boot_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in bh) + "\n")
    print(f"[probe] wrote {out_dir/'bootstrap_cis.csv'}")

    # ─── Paired DeLong + McNemar tests (seed 0, key pairs) ──────────────────
    print("\n[probe] running paired tests (seed 0) ...")
    PAIRS = [
        # (regime_a, variant_a, regime_b, variant_b, label)
        ("ff_only", "backbone_only", "mixed", "backbone_only", "pooling_helps"),
        ("mixed", "backbone_only", "mixed", "backbone+rppg", "mixed_rppg_vs_bb"),
        ("mixed", "backbone_only", "mixed", "backbone+blink", "mixed_blink_vs_bb"),
        ("mixed", "backbone+rppg", "mixed", "backbone+blink", "mixed_rppg_vs_blink"),
        ("mixed", "backbone+rppg", "mixed", "full_fusion", "mixed_rppg_vs_fusion"),
        ("ff_only", "backbone_only", "ff_only", "backbone+rppg", "ff_rppg_vs_bb"),
    ]
    pair_rows = []
    for ra, va, rb, vb, tag in PAIRS:
        for split in ["ff_test", "celebdf_test", "dfdc_test"]:
            ka = (args.seeds[0], ra, va, split)
            kb = (args.seeds[0], rb, vb, split)
            if ka not in all_scores or kb not in all_scores:
                continue
            sa, ya = all_scores[ka]
            sb, yb = all_scores[kb]
            if not np.array_equal(ya, yb):
                print(f"  WARNING: label mismatch for {tag}/{split}")
                continue
            auc_a, auc_b, z, p = delong_test(ya, sa, sb)
            # Youden threshold on variant A then McNemar
            thr = youden_threshold(ya, sa)
            mc = mcnemar_test(ya, sa, sb, threshold=thr)
            pair_rows.append({
                "pair": tag, "a": f"{ra}/{va}", "b": f"{rb}/{vb}", "split": split,
                "auc_a": auc_a, "auc_b": auc_b, "auc_diff": auc_b - auc_a,
                "delong_z": z, "delong_p": p,
                "youden_thr_on_a": thr,
                "mc_a_wins": mc["a_wins"], "mc_b_wins": mc["b_wins"],
                "mc_chi2": mc["chi2"], "mc_p": mc["p"],
            })
            print(f"  {tag:25s} {split:14s} AUC_A={auc_a:.4f} AUC_B={auc_b:.4f} "
                  f"ΔAUC={auc_b-auc_a:+.4f} DeLong p={p:.4g}  "
                  f"McNemar (a_wins={mc['a_wins']} b_wins={mc['b_wins']}) p={mc['p']:.4g}")

    # Bonferroni correction across the family
    n_tests = len(pair_rows)
    for r in pair_rows:
        r["delong_p_bonf"] = min(1.0, r["delong_p"] * n_tests)
        r["mc_p_bonf"] = min(1.0, r["mc_p"] * n_tests)

    ph = ["pair", "a", "b", "split", "auc_a", "auc_b", "auc_diff",
          "delong_z", "delong_p", "delong_p_bonf",
          "youden_thr_on_a", "mc_a_wins", "mc_b_wins", "mc_chi2", "mc_p", "mc_p_bonf"]
    with open(out_dir / "paired_tests.csv", "w") as f:
        f.write(",".join(ph) + "\n")
        for r in pair_rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                             for k in ph) + "\n")
    print(f"[probe] wrote {out_dir/'paired_tests.csv'} ({n_tests} comparisons, Bonferroni-corrected)")

    # ─── Final summary JSON for easy paste-back ──────────────────────────────
    summary = {
        "seeds": list(args.seeds),
        "n_agg_cells": len(agg_rows),
        "n_boot_cells": len(boot_rows),
        "n_paired_tests": n_tests,
        "agg": agg_rows,
        "bootstrap": boot_rows,
        "paired": pair_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[probe] wrote {out_dir/'summary.json'}")
    print("\n[probe] DONE. Deliverables:")
    print(f"  - multiseed_results.csv  (per-seed raw metrics)")
    print(f"  - multiseed_aggregate.csv (mean±std across seeds)")
    print(f"  - bootstrap_cis.csv       (95% CIs, seed 0)")
    print(f"  - paired_tests.csv        (DeLong + McNemar + Bonferroni)")
    print(f"  - summary.json            (everything, copy-pasteable)")
    print(f"  - scores/*.npz            (per-clip scores per (seed,regime,variant,split))")


# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="E1+E2 multi-seed probe + paired stats")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Stage 1: extract features, save npz per dataset")
    ex.add_argument("--ff_root", required=True)
    ex.add_argument("--celebdf_root", default=None)
    ex.add_argument("--dfdc_faces_root", default=None)
    ex.add_argument("--resume_ckpt", required=True)
    ex.add_argument("--rppg_cache", default=None, help="FF++ rPPG cache (v2 12-d)")
    ex.add_argument("--blink_cache", default=None, help="FF++ blink cache")
    ex.add_argument("--celebdf_rppg_cache", default=None)
    ex.add_argument("--celebdf_blink_cache", default=None)
    ex.add_argument("--dfdc_rppg_cache", default=None)
    ex.add_argument("--dfdc_blink_cache", default=None)
    ex.add_argument("--cache_dir", required=True)
    ex.add_argument("--clip_len", type=int, default=16)
    ex.add_argument("--img_size", type=int, default=224)
    ex.add_argument("--rppg_dim", type=int, default=12)
    ex.add_argument("--batch_size", type=int, default=8)
    ex.add_argument("--num_workers", type=int, default=2)

    pr = sub.add_parser("probe", help="Stage 2+3: multi-seed probe + stats")
    pr.add_argument("--cache_dir", required=True)
    pr.add_argument("--out_dir", required=True)
    pr.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    pr.add_argument("--epochs", type=int, default=20)
    pr.add_argument("--lr", type=float, default=1e-3)
    pr.add_argument("--batch", type=int, default=256)
    pr.add_argument("--n_boot", type=int, default=1000)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "extract":
        do_extract(args)
    elif args.cmd == "probe":
        do_probe(args)
