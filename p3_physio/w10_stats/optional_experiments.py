"""
E8 + E9 + E10 — optional defense-strengthening experiments.

All three reuse the existing E6 caches (feat_cache_b4 / feat_cache_clip /
feat_cache_dinov2) where possible, so we don't re-extract the FF++ c23 /
CelebDF / DFDC features.

Sub-commands:

  ─────────────────────────────────────────────────────────────────────
  E8 — extract_c40         FF++ c40 backbone features
       eval_c40            Apply pre-trained probe to c40 features
  ─────────────────────────────────────────────────────────────────────
  E9 — extract_robust      FF++ c23 features under 12 perturbations
       eval_robust         Apply pre-trained probe to perturbed features
  ─────────────────────────────────────────────────────────────────────
  E10 — calibrate          Platt scaling on the mixed probe + ECE report
  ─────────────────────────────────────────────────────────────────────

Usage examples:

  # E8 (~80 min CLIP)
  python optional_experiments.py extract_c40 \
      --backbone clip_vitl14 \
      --ff_c40_root "$FF_C40" \
      --rppg_cache  "$RPPG" --blink_cache "$BLINK" \
      --cache_dir   /kaggle/working/feat_cache_clip_c40 \
      --batch_size 4

  python optional_experiments.py eval_c40 \
      --c40_cache /kaggle/working/feat_cache_clip_c40/ff_c40.npz \
      --c23_cache /kaggle/input/.../feat_cache_clip \
      --out_dir   /kaggle/working/e8_clip_c40 \
      --seeds 0 1 42 1337 2024

  # E9 (~80 min CLIP per perturbation × 12 = ~16 h; recommend run subset)
  python optional_experiments.py extract_robust \
      --backbone clip_vitl14 \
      --ff_root "$FF" \
      --rppg_cache "$RPPG" --blink_cache "$BLINK" \
      --cache_dir /kaggle/working/feat_cache_clip_robust \
      --perturbations clean,jpeg_q50,jpeg_q30,jpeg_q10,blur_s1,blur_s2,blur_s3,noise_s5,noise_s10,noise_s20,downscale_2x,downscale_4x \
      --batch_size 4

  python optional_experiments.py eval_robust \
      --robust_cache_dir /kaggle/working/feat_cache_clip_robust \
      --c23_cache       /kaggle/input/.../feat_cache_clip \
      --out_dir         /kaggle/working/e9_clip_robust \
      --seeds 0 1 42 1337 2024

  # E10 (~5 min, no extraction)
  python optional_experiments.py calibrate \
      --cache_dir /kaggle/input/.../feat_cache_clip \
      --out_dir   /kaggle/working/e10_clip_calibration \
      --seeds 0 1 42 1337 2024
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


# ───────────────────────────────────────────────────────────────────────────
# Reuse the proven multiseed_and_stats helpers
# ───────────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import (
    roc_auc, eer, average_precision, tpr_at_fpr,
    train_linear_probe, predict, identity_split_ff, random_split,
    bootstrap_ci_auc,
    VARIANTS, REGIMES, make_features,
)
from extract_clip_backbone import (
    scan_ff, scan_celebdf, scan_dfdc_faces,
    ff_cache_key, celebdf_cache_key, dfdc_cache_key,
    load_backbone,
    FF_MANIPULATION_TYPES,
)


# ───────────────────────────────────────────────────────────────────────────
# FF++ c40 mp4 scanner (wahabarabo/ff-c40-videos layout)
# ───────────────────────────────────────────────────────────────────────────

def scan_ff_c40_mp4(root):
    """
    wahabarabo/ff-c40-videos layout:
      <root>/c40_original/<vid>.mp4              → manip='original',  src_id=<vid>
      <root>/c40_Deepfakes/<src>_<tgt>.mp4       → manip='Deepfakes', src_id=<src>
      <root>/c40_Face2Face/<src>_<tgt>.mp4       → manip='Face2Face', src_id=<src>
      <root>/c40_FaceSwap/<src>_<tgt>.mp4        → manip='FaceSwap',  src_id=<src>
      <root>/c40_NeuralTextures/<src>_<tgt>.mp4  → manip='NeuralTextures', src_id=<src>
      <root>/c40_FaceShifter/<src>_<tgt>.mp4     → manip='FaceShifter', src_id=<src>

    Returns (paths, labels, manips, src_ids) — paths are mp4 files (strings),
    not directories.  ClipDataset will detect mp4 vs dir and decode accordingly.
    """
    root = Path(root)
    paths, labels, manips, src_ids = [], [], [], []
    name_map = {
        "c40_original": "original",
        "c40_Deepfakes": "Deepfakes",
        "c40_Face2Face": "Face2Face",
        "c40_FaceSwap": "FaceSwap",
        "c40_NeuralTextures": "NeuralTextures",
        "c40_FaceShifter": "FaceShifter",
    }
    for c40_name, manip in name_map.items():
        d = root / c40_name
        if not d.exists():
            continue
        label = FF_MANIPULATION_TYPES.get(manip, 1)
        for f in sorted(d.glob("*.mp4")):
            paths.append(str(f))
            labels.append(label)
            manips.append(manip)
            # source id is the first '_' token of the stem (e.g. "000_003" → "000")
            src_ids.append(f.stem.split("_")[0])
    return paths, labels, manips, src_ids


def ff_c40_cache_key(path):
    """For c40 mp4 paths, manip + video_stem.  Used only if rPPG/blink caches
    were ever extracted on c40 — currently we pass None for both."""
    p = Path(path)
    return p.parent.name.replace("c40_", ""), p.stem


def _load_mp4_frames_uniform(mp4_path, n_frames):
    """Open mp4 with cv2, uniformly sample n_frames RGB uint8 frames."""
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * n_frames
    # Sample evenly
    target_idx = np.linspace(0, total - 1, n_frames).astype(int)
    target_set = set(target_idx.tolist())
    frames = {}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in target_set:
            frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(frames) == n_frames:
                break
        fi += 1
    cap.release()
    out = []
    for ti in target_idx:
        if ti in frames:
            out.append(frames[ti])
        else:
            out.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return out


# ───────────────────────────────────────────────────────────────────────────
# Perturbation library (E9)
# ───────────────────────────────────────────────────────────────────────────

import cv2  # noqa: E402

PERTURBATIONS = {
    # name → callable: uint8 HxWx3 RGB → uint8 HxWx3 RGB
    "clean":         lambda img: img,
    "jpeg_q50":      lambda img: _jpeg(img, 50),
    "jpeg_q30":      lambda img: _jpeg(img, 30),
    "jpeg_q10":      lambda img: _jpeg(img, 10),
    "blur_s1":       lambda img: _gauss_blur(img, 1.0),
    "blur_s2":       lambda img: _gauss_blur(img, 2.0),
    "blur_s3":       lambda img: _gauss_blur(img, 3.0),
    "noise_s5":      lambda img: _gauss_noise(img, 5.0),
    "noise_s10":     lambda img: _gauss_noise(img, 10.0),
    "noise_s20":     lambda img: _gauss_noise(img, 20.0),
    "downscale_2x":  lambda img: _downscale_then_up(img, 2),
    "downscale_4x":  lambda img: _downscale_then_up(img, 4),
}


def _jpeg(img_rgb, quality):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img_rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _gauss_blur(img_rgb, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img_rgb, (k, k), sigma)


def _gauss_noise(img_rgb, sigma):
    noisy = img_rgb.astype(np.float32) + np.random.normal(0, sigma, img_rgb.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _downscale_then_up(img_rgb, factor):
    H, W = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (max(1, W // factor), max(1, H // factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


# ───────────────────────────────────────────────────────────────────────────
# B4 (v13) backbone loader — separate from CLIP/DINOv2 because v13 needs
# its own checkpoint and ImageNet-style preprocessing.
# ───────────────────────────────────────────────────────────────────────────

def load_b4_v13_backbone(ckpt_path, device):
    """
    Load the v13 EfficientNet-B4 backbone. Returns (forward_fn, preprocess_batch, feat_dim).
    Compatible with the (forward_fn, preprocess_batch, feat_dim) interface
    that extract_backbone_with_perturbation expects.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from w2_model.model import PhysioNet, ModelConfig

    cfg = ModelConfig(
        backbone="efficientnet_b4", backbone_pretrained=False,
        temporal_model="mean", temporal_dim=0,
        clip_len=16, img_size=224, dropout=0.0,
        use_physio_fusion=False, use_pulse_head=False,
        use_blink_head=False, use_motion_model=False,
    )
    model = PhysioNet(cfg).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        bb_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(bb_state, strict=False)
        print(f"[B4] loaded {len(bb_state)} backbone tensors from {ckpt_path}")
    else:
        print(f"[B4] WARNING: no checkpoint at {ckpt_path}; using random init")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_batch(imgs_hwc_rgb):
        # imgs are uint8 RGB at native resolution; resize to 224 and normalize
        out = np.zeros((len(imgs_hwc_rgb), 3, 224, 224), dtype=np.float32)
        for i, img in enumerate(imgs_hwc_rgb):
            img224 = cv2.resize(img, (224, 224))
            img_f = img224.astype(np.float32) / 255.0
            img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
            out[i] = img_f.transpose(2, 0, 1)
        return torch.from_numpy(out).to(device, non_blocking=True)

    @torch.no_grad()
    def forward(x):
        # PhysioNet's frame_encoder expects (B, T, 3, H, W); we pass single frames
        # by introducing a T=1 dim then squeezing.
        B = x.shape[0]
        x_5d = x.unsqueeze(1)  # (B, 1, 3, H, W)
        feats = model.frame_encoder(x_5d)  # (B, 1, 1792)
        return feats.squeeze(1).float()

    feat_dim = 1792
    return forward, preprocess_batch, feat_dim


# ───────────────────────────────────────────────────────────────────────────
# Generic backbone-extraction helper (parameterised by perturbation)
# ───────────────────────────────────────────────────────────────────────────

def extract_backbone_with_perturbation(
    args, backbone_name, dirs, labels, perturb_fn,
    rppg_cache, blink_cache, cache_key_fn,
    tag, out_path, manips=None, src_ids=None,
    b4_ckpt=None,
):
    """
    Extract pooled backbone features for `dirs`, applying `perturb_fn` to each
    loaded frame BEFORE the HF preprocessor.  Saves npz with the same schema
    as multiseed_and_stats `extract`.

    backbone_name='efficientnet_b4_v13' uses the v13 checkpoint at b4_ckpt.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone_name == "efficientnet_b4_v13":
        forward_fn, preprocess_batch, feat_dim = load_b4_v13_backbone(b4_ckpt, device)
    else:
        forward_fn, preprocess_batch, feat_dim = load_backbone(backbone_name, device)

    class ClipDataset(Dataset):
        """
        Three input modes, auto-detected per-entry:
          1. video_dirs[i] is a list of frame paths       → use as-is
          2. video_dirs[i] is a directory of frame images → list its frames
          3. video_dirs[i] is a path ending in .mp4       → decode with cv2
        """
        def __init__(self, video_dirs, labels):
            self.labels = labels
            self.video_dirs_raw = video_dirs
            # Pre-compute mode + sorted frame list per entry
            self.modes = []
            self.frame_paths = []
            for vd in video_dirs:
                if isinstance(vd, list):
                    self.modes.append("frames")
                    self.frame_paths.append(sorted(vd))
                elif isinstance(vd, str) and vd.lower().endswith((".mp4", ".avi", ".mov")):
                    self.modes.append("mp4")
                    self.frame_paths.append([vd])  # single mp4 path
                else:
                    self.modes.append("frames")
                    self.frame_paths.append(sorted(
                        os.path.join(vd, f) for f in os.listdir(vd)
                        if f.endswith((".png", ".jpg", ".jpeg"))
                    ))

        def __len__(self):
            return len(self.frame_paths)

        def __getitem__(self, idx):
            mode = self.modes[idx]
            frames = self.frame_paths[idx]
            imgs = []
            if mode == "mp4":
                # Decode mp4, sample uniformly, apply perturbation
                raw = _load_mp4_frames_uniform(frames[0], args.clip_len)
                imgs = [perturb_fn(img) for img in raw]
            else:
                n = len(frames)
                if n == 0:
                    imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(args.clip_len)]
                else:
                    start = max(0, n - args.clip_len) // 2
                    indices = [(start + i) % n for i in range(args.clip_len)]
                    for fi in indices:
                        img = cv2.imread(frames[fi])
                        if img is None:
                            img = np.zeros((224, 224, 3), dtype=np.uint8)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = perturb_fn(img)            # ← perturbation here
                        imgs.append(img)

            vd = self.video_dirs_raw[idx]
            class_name, video_id = cache_key_fn(vd)

            rppg_feat = np.zeros(12, dtype=np.float32)
            if rppg_cache and class_name and video_id:
                cp = Path(rppg_cache) / class_name / video_id / "rppg_v2_feat.npy"
                if cp.exists():
                    loaded = np.load(str(cp)).astype(np.float32)
                    if len(loaded) <= 12:
                        rppg_feat[: len(loaded)] = loaded

            blink_feat = np.zeros(16, dtype=np.float32)
            if blink_cache and class_name and video_id:
                bp = Path(blink_cache) / class_name / video_id / "blink_feat.npy"
                if bp.exists():
                    loaded = np.load(str(bp)).astype(np.float32)
                    if len(loaded) == 16:
                        blink_feat = loaded

            return {"imgs": imgs, "label": float(self.labels[idx]),
                    "rppg": rppg_feat, "blink": blink_feat}

    def collate(batch):
        return batch

    ds = ClipDataset(dirs, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False,
                    collate_fn=collate)

    bb_list, rppg_list, blink_list, label_list = [], [], [], []
    t0 = time.time()
    import torch
    for batch in tqdm(dl, desc=tag, leave=False):
        all_imgs = []
        for s in batch:
            all_imgs.extend(s["imgs"])
        pixel_values = preprocess_batch(all_imgs)
        chunk = 32
        chunks = []
        with torch.no_grad():
            for i in range(0, pixel_values.shape[0], chunk):
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    chunks.append(forward_fn(pixel_values[i:i + chunk]))
        feats = torch.cat(chunks, dim=0)
        feats = feats.view(len(batch), args.clip_len, -1).mean(dim=1)
        feats = feats.float().cpu().numpy()
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        bb_list.append(feats)
        rppg_list.append(np.stack([s["rppg"] for s in batch]))
        blink_list.append(np.stack([s["blink"] for s in batch]))
        label_list.append(np.array([s["label"] for s in batch], dtype=np.float32))

    bb = np.concatenate(bb_list, axis=0)
    rppg = np.concatenate(rppg_list, axis=0)
    blink = np.concatenate(blink_list, axis=0)
    lbls = np.concatenate(label_list, axis=0)

    out = {"backbone": bb, "rppg": rppg, "blink": blink, "labels": lbls}
    if manips is not None:
        out["manip"] = np.array(manips)
    if src_ids is not None:
        out["src_id"] = np.array(src_ids)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    rppg_hit = (np.abs(rppg).sum(1) > 0).mean() * 100
    blink_hit = (np.abs(blink).sum(1) > 0).mean() * 100
    print(f"  {tag}: n={len(lbls)} feat_dim={bb.shape[1]} "
          f"rppg_hit={rppg_hit:.0f}% blink_hit={blink_hit:.0f}% "
          f"time={time.time()-t0:.1f}s → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# E8 — FF++ c40 evaluation
# ═══════════════════════════════════════════════════════════════════════════

def cmd_extract_c40(args):
    """
    Extract backbone features on FF++ c40 mp4 videos.

    The c40 dataset (wahabarabo/ff-c40-videos) ships mp4s, not pre-extracted
    frames, so the scanner returns mp4 paths and ClipDataset decodes them
    on the fly. rPPG/blink are not used for c40 — physiology features were
    only extracted on c23 frames; the c40 evaluation isolates the
    backbone-compression effect.
    """
    print(f"[E8] extracting {args.backbone} features on FF++ c40")
    print(f"     ff_c40_root = {args.ff_c40_root}")
    if args.c40_layout == "mp4":
        ff_paths, ff_labels, ff_manips, ff_src = scan_ff_c40_mp4(args.ff_c40_root)
        cache_key_fn = ff_c40_cache_key
    else:  # 'frames' — same layout as c23
        ff_paths, ff_labels, ff_manips, ff_src = scan_ff(args.ff_c40_root)
        cache_key_fn = ff_cache_key
    print(f"[E8] FF++ c40: {len(ff_paths)} videos")
    if len(ff_paths) == 0:
        print(f"[E8] No videos found at {args.ff_c40_root} — check path/layout")
        return

    # Per-class breakdown
    from collections import Counter
    by_manip = Counter(ff_manips)
    print(f"[E8] per-manipulation: {dict(by_manip)}")

    # Pass None for both physiology caches — they are c23-frame-derived, not
    # applicable to c40 mp4s. Backbone features are what we care about for E8.
    extract_backbone_with_perturbation(
        args, args.backbone, ff_paths, ff_labels, PERTURBATIONS["clean"],
        rppg_cache=None, blink_cache=None,
        cache_key_fn=cache_key_fn,
        tag="ff_c40",
        out_path=str(Path(args.cache_dir) / "ff_c40.npz"),
        manips=ff_manips, src_ids=ff_src,
        b4_ckpt=getattr(args, "b4_ckpt", None),
    )
    print(f"[E8] DONE. Cache at {args.cache_dir}/ff_c40.npz")


def cmd_eval_c40(args):
    """
    Train the same multi-seed mixed probe on c23 cache, then evaluate on the
    c40 test split.  Reports per-seed AUC and mean ± std plus bootstrap CIs.

    The c40 test split is the **same source IDs** as the c23 test split (the
    identity-aware FF++ split is deterministic), so AUC is directly comparable.
    """
    print(f"[E8] eval c40")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load c23 training caches (FF + CDF + DFDC, mixed regime)
    c23_dir = Path(args.c23_cache)
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = c23_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"  loaded c23 {tag}: n={len(caches[tag]['labels'])} bb_dim={caches[tag]['backbone'].shape[1]}")

    c40 = {k: v for k, v in np.load(args.c40_cache, allow_pickle=True).items()}
    print(f"  loaded c40: n={len(c40['labels'])} bb_dim={c40['backbone'].shape[1]}")

    # Identity-aware FF++ split (same as multiseed_and_stats: train/val/test on src_id)
    tr_ff_idx, vl_ff_idx, te_ff_idx = identity_split_ff(caches["ff"], seed=42)
    # Apply the SAME source-ID partition to c40 — c40 has identical src_ids
    # to c23 since they're the same videos at different compression.
    c40_te_mask = np.isin(c40["src_id"], caches["ff"]["src_id"][te_ff_idx])
    c40_te_idx = np.where(c40_te_mask)[0]
    print(f"  c40 test split (matched to c23 test src_ids): n={len(c40_te_idx)}")

    cd_tr, cd_te = random_split(len(caches["celebdf"]["labels"]), seed=42)
    df_tr, df_te = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    rows = []
    for seed in args.seeds:
        for variant in VARIANTS:
            # Build mixed pool (FF c23 train + CDF train + DFDC train)
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

            # Validation = c23 FF val
            bb_vl = caches["ff"]["backbone"][vl_ff_idx]
            rppg_vl = caches["ff"]["rppg"][vl_ff_idx]
            blink_vl = caches["ff"]["blink"][vl_ff_idx]
            y_vl = caches["ff"]["labels"][vl_ff_idx]

            X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
            X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
            probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                        epochs=args.epochs, lr=args.lr,
                                        bs=args.batch, seed=seed)

            # Evaluate on c23 test (sanity), then c40 test
            for split_name, bb_te, rppg_te, blink_te, y_te in [
                ("ff_c23_test",
                 caches["ff"]["backbone"][te_ff_idx],
                 caches["ff"]["rppg"][te_ff_idx],
                 caches["ff"]["blink"][te_ff_idx],
                 caches["ff"]["labels"][te_ff_idx]),
                ("ff_c40_test",
                 c40["backbone"][c40_te_idx],
                 c40["rppg"][c40_te_idx],
                 c40["blink"][c40_te_idx],
                 c40["labels"][c40_te_idx]),
            ]:
                X_te = make_features(bb_te, rppg_te, blink_te, variant)
                scores = predict(probe, X_te, device)
                row = {
                    "seed": seed, "variant": variant, "split": split_name,
                    "n": int(len(y_te)),
                    "auc": roc_auc(y_te, scores),
                    "ap": average_precision(y_te, scores),
                    "eer": eer(y_te, scores),
                    "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                }
                rows.append(row)
                print(f"  seed={seed} {variant:18s} {split_name:14s} AUC={row['auc']:.4f} EER={row['eer']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_aggregate(rows, out_dir, key=("variant", "split"))
    print(f"[E8] wrote {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# E9 — Robustness on CLIP under 12 perturbations
# ═══════════════════════════════════════════════════════════════════════════

def cmd_extract_robust(args):
    """Extract FF++ features under each requested perturbation. CelebDF/DFDC unchanged."""
    print(f"[E9] extracting {args.backbone} on FF++ under perturbations")
    perturbs = [p.strip() for p in args.perturbations.split(",")]
    for p in perturbs:
        if p not in PERTURBATIONS:
            print(f"[E9] unknown perturbation: {p} — skipping")
            continue

    ff_dirs, ff_labels, ff_manips, ff_src = scan_ff(args.ff_root)
    print(f"[E9] FF++: {len(ff_dirs)} videos")

    for p in perturbs:
        if p not in PERTURBATIONS:
            continue
        out_path = Path(args.cache_dir) / f"ff_{p}.npz"
        if out_path.exists() and not args.force:
            print(f"  {p}: already cached at {out_path}, skipping")
            continue
        extract_backbone_with_perturbation(
            args, args.backbone, ff_dirs, ff_labels, PERTURBATIONS[p],
            args.rppg_cache, args.blink_cache, ff_cache_key,
            tag=f"ff_{p}",
            out_path=str(out_path),
            manips=ff_manips, src_ids=ff_src,
            b4_ckpt=getattr(args, "b4_ckpt", None),
        )
    print(f"[E9] DONE. Caches in {args.cache_dir}")


def cmd_eval_robust(args):
    """
    Train the same probe regimes on the CLEAN c23 training pool, then
    evaluate on each perturbed FF++ test split.  Reports per-seed AUC drop
    relative to the clean baseline.
    """
    print(f"[E9] eval robust")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c23_dir = Path(args.c23_cache)
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = c23_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}

    # Robust caches keyed by perturbation name
    robust_dir = Path(args.robust_cache_dir)
    perturb_caches = {}
    for npz in robust_dir.glob("ff_*.npz"):
        pname = npz.stem.replace("ff_", "")
        perturb_caches[pname] = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
        print(f"  loaded perturb '{pname}': n={len(perturb_caches[pname]['labels'])}")

    tr_ff_idx, vl_ff_idx, te_ff_idx = identity_split_ff(caches["ff"], seed=42)
    cd_tr, _ = random_split(len(caches["celebdf"]["labels"]), seed=42)
    df_tr, _ = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    test_src_ids = caches["ff"]["src_id"][te_ff_idx]

    rows = []
    for seed in args.seeds:
        for variant in VARIANTS:
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

            X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
            X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
            probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                        epochs=args.epochs, lr=args.lr,
                                        bs=args.batch, seed=seed)

            for pname, pcache in perturb_caches.items():
                # Match by src_id to test partition
                te_mask = np.isin(pcache["src_id"], test_src_ids)
                idx = np.where(te_mask)[0]
                if len(idx) == 0:
                    continue
                bb_te = pcache["backbone"][idx]
                rppg_te = pcache["rppg"][idx]
                blink_te = pcache["blink"][idx]
                y_te = pcache["labels"][idx]
                X_te = make_features(bb_te, rppg_te, blink_te, variant)
                scores = predict(probe, X_te, device)
                row = {
                    "seed": seed, "variant": variant, "perturbation": pname,
                    "n": int(len(y_te)),
                    "auc": roc_auc(y_te, scores),
                    "eer": eer(y_te, scores),
                    "tpr5": tpr_at_fpr(y_te, scores, 0.05),
                }
                rows.append(row)
                print(f"  seed={seed} {variant:18s} {pname:13s} AUC={row['auc']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_aggregate(rows, out_dir, key=("variant", "perturbation"))
    print(f"[E9] wrote {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# E10 — Platt scaling on the mixed probe
# ═══════════════════════════════════════════════════════════════════════════

def _expected_calibration_error(y_true, probs, n_bins=15):
    """Standard ECE with equal-width bins."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def _platt_fit(scores, labels):
    """
    Platt scaling: fit logistic regression p(y=1) = sigmoid(a * score + b).
    Returns (a, b).
    """
    import torch
    import torch.nn as nn
    device = torch.device("cpu")  # tiny problem, CPU is fine
    s = torch.tensor(scores, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    a = torch.zeros(1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=200)
    crit = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = crit(a * s + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.item()), float(b.item())


def _platt_apply(scores, a, b):
    return 1.0 / (1.0 + np.exp(-(a * scores + b)))


def cmd_calibrate(args):
    """
    For each (regime=mixed, variant) and seed, fit Platt scaling on the FF++
    val split, then report ECE before/after on FF++ test, CelebDF test, DFDC test.
    """
    print(f"[E10] calibration on cache {args.cache_dir}")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.cache_dir)
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = cache_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            print(f"  loaded {tag}: n={len(caches[tag]['labels'])}")

    tr_ff_idx, vl_ff_idx, te_ff_idx = identity_split_ff(caches["ff"], seed=42)
    cd_tr, cd_te = random_split(len(caches["celebdf"]["labels"]), seed=42)
    df_tr, df_te = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    rows = []
    for seed in args.seeds:
        for variant in VARIANTS:
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

            X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
            X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
            probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                        epochs=args.epochs, lr=args.lr,
                                        bs=args.batch, seed=seed)

            # Compute val-set scores → fit Platt
            val_scores = predict(probe, X_vl, device)
            a, b = _platt_fit(val_scores, y_vl)

            for split_name, bb_te, rppg_te, blink_te, y_te in [
                ("ff_test", caches["ff"]["backbone"][te_ff_idx], caches["ff"]["rppg"][te_ff_idx],
                 caches["ff"]["blink"][te_ff_idx], caches["ff"]["labels"][te_ff_idx]),
                ("celebdf_test", caches["celebdf"]["backbone"][cd_te], caches["celebdf"]["rppg"][cd_te],
                 caches["celebdf"]["blink"][cd_te], caches["celebdf"]["labels"][cd_te]),
            ] + ([("dfdc_test", caches["dfdc"]["backbone"][df_te], caches["dfdc"]["rppg"][df_te],
                   caches["dfdc"]["blink"][df_te], caches["dfdc"]["labels"][df_te])] if df_te is not None else []):
                X_te = make_features(bb_te, rppg_te, blink_te, variant)
                raw = predict(probe, X_te, device)
                cal = _platt_apply(raw, a, b)
                row = {
                    "seed": seed, "variant": variant, "split": split_name,
                    "n": int(len(y_te)),
                    "auc": roc_auc(y_te, raw),
                    "ece_raw": _expected_calibration_error(y_te, raw),
                    "ece_platt": _expected_calibration_error(y_te, cal),
                    "platt_a": a, "platt_b": b,
                }
                rows.append(row)
                print(f"  seed={seed} {variant:18s} {split_name:14s} "
                      f"AUC={row['auc']:.4f}  ECE: {row['ece_raw']:.4f} → {row['ece_platt']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Custom aggregation since we want raw / platt ECE side-by-side
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["split"])].append(r)

    print("\n[E10] AGGREGATE (mean ± std across seeds)")
    print(f"{'variant':<18s} {'split':<14s}  AUC             ECE raw         ECE Platt        ΔECE")
    agg_rows = []
    for (var, split), rs in sorted(agg.items()):
        aucs = [r["auc"] for r in rs]
        ece_r = [r["ece_raw"] for r in rs]
        ece_p = [r["ece_platt"] for r in rs]
        am, asd = float(np.mean(aucs)), float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0)
        em, esd = float(np.mean(ece_r)), float(np.std(ece_r, ddof=1) if len(ece_r) > 1 else 0)
        pm, psd = float(np.mean(ece_p)), float(np.std(ece_p, ddof=1) if len(ece_p) > 1 else 0)
        delta = pm - em
        print(f"{var:<18s} {split:<14s}  {am:.4f}±{asd:.4f}  {em:.4f}±{esd:.4f}  {pm:.4f}±{psd:.4f}  {delta:+.4f}")
        agg_rows.append({"variant": var, "split": split, "n": rs[0]["n"],
                         "auc_mean": am, "auc_std": asd,
                         "ece_raw_mean": em, "ece_raw_std": esd,
                         "ece_platt_mean": pm, "ece_platt_std": psd,
                         "delta_ece": delta})

    head = list(agg_rows[0].keys()) if agg_rows else []
    with open(out_dir / "aggregate.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k]) for k in head) + "\n")

    raw_head = list(rows[0].keys()) if rows else []
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(raw_head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in raw_head) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"agg": agg_rows, "raw": rows}, f, indent=2, default=str)

    print(f"[E10] wrote {out_dir}")


# ───────────────────────────────────────────────────────────────────────────
# Generic aggregation helper
# ───────────────────────────────────────────────────────────────────────────

def _write_aggregate(rows, out_dir, key):
    """Write multiseed_results.csv + aggregate.csv given a (variant, …) key tuple."""
    from collections import defaultdict
    out_dir = Path(out_dir)

    head = list(rows[0].keys()) if rows else []
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")

    agg = defaultdict(list)
    for r in rows:
        agg[tuple(r[k] for k in key)].append(r)

    print(f"\nAGGREGATE (mean ± std across seeds)")
    head_agg = list(key) + ["n", "auc_mean", "auc_std", "eer_mean", "eer_std", "tpr5_mean", "tpr5_std"]
    agg_rows = []
    for k_vals, rs in sorted(agg.items()):
        aucs = [r["auc"] for r in rs]
        eers = [r["eer"] for r in rs]
        tprs = [r.get("tpr5", float("nan")) for r in rs]
        row = dict(zip(key, k_vals))
        row["n"] = rs[0]["n"]
        row["auc_mean"] = float(np.mean(aucs))
        row["auc_std"] = float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0)
        row["eer_mean"] = float(np.mean(eers))
        row["eer_std"] = float(np.std(eers, ddof=1) if len(eers) > 1 else 0)
        row["tpr5_mean"] = float(np.mean(tprs))
        row["tpr5_std"] = float(np.std(tprs, ddof=1) if len(tprs) > 1 else 0)
        agg_rows.append(row)
        print(f"  {' '.join(str(v) for v in k_vals):<35s} AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}")

    with open(out_dir / "aggregate.csv", "w") as f:
        f.write(",".join(head_agg) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k]) for k in head_agg) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"agg": agg_rows, "raw": rows}, f, indent=2, default=str)


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="E8/E9/E10 optional experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    # E8: extract_c40
    e8a = sub.add_parser("extract_c40")
    e8a.add_argument("--backbone", choices=["clip_vitl14", "dinov2_vitb14", "efficientnet_b4_v13"], required=True)
    e8a.add_argument("--b4_ckpt", default=None, help="Required when --backbone efficientnet_b4_v13")
    e8a.add_argument("--ff_c40_root", required=True)
    e8a.add_argument("--c40_layout", choices=["mp4", "frames"], default="mp4",
                     help="'mp4' for wahabarabo/ff-c40-videos style (default); "
                          "'frames' for the same per-video PNG layout as c23.")
    e8a.add_argument("--rppg_cache", default=None)
    e8a.add_argument("--blink_cache", default=None)
    e8a.add_argument("--cache_dir", required=True)
    e8a.add_argument("--clip_len", type=int, default=16)
    e8a.add_argument("--batch_size", type=int, default=4)
    e8a.add_argument("--num_workers", type=int, default=2)

    # E8: eval_c40
    e8b = sub.add_parser("eval_c40")
    e8b.add_argument("--c40_cache", required=True, help="Path to ff_c40.npz")
    e8b.add_argument("--c23_cache", required=True, help="Path to feat_cache_<bb>/ directory (with ff/celebdf/dfdc.npz)")
    e8b.add_argument("--out_dir", required=True)
    e8b.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    e8b.add_argument("--epochs", type=int, default=20)
    e8b.add_argument("--lr", type=float, default=1e-3)
    e8b.add_argument("--batch", type=int, default=256)

    # E9: extract_robust
    e9a = sub.add_parser("extract_robust")
    e9a.add_argument("--backbone", choices=["clip_vitl14", "dinov2_vitb14", "efficientnet_b4_v13"], required=True)
    e9a.add_argument("--b4_ckpt", default=None, help="Required when --backbone efficientnet_b4_v13")
    e9a.add_argument("--ff_root", required=True)
    e9a.add_argument("--rppg_cache", default=None)
    e9a.add_argument("--blink_cache", default=None)
    e9a.add_argument("--cache_dir", required=True)
    e9a.add_argument("--perturbations", default="clean,jpeg_q50,jpeg_q30,jpeg_q10,blur_s1,blur_s2,blur_s3,noise_s5,noise_s10,noise_s20,downscale_2x,downscale_4x")
    e9a.add_argument("--clip_len", type=int, default=16)
    e9a.add_argument("--batch_size", type=int, default=4)
    e9a.add_argument("--num_workers", type=int, default=2)
    e9a.add_argument("--force", action="store_true")

    # E9: eval_robust
    e9b = sub.add_parser("eval_robust")
    e9b.add_argument("--robust_cache_dir", required=True)
    e9b.add_argument("--c23_cache", required=True)
    e9b.add_argument("--out_dir", required=True)
    e9b.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    e9b.add_argument("--epochs", type=int, default=20)
    e9b.add_argument("--lr", type=float, default=1e-3)
    e9b.add_argument("--batch", type=int, default=256)

    # E10: calibrate
    e10 = sub.add_parser("calibrate")
    e10.add_argument("--cache_dir", required=True)
    e10.add_argument("--out_dir", required=True)
    e10.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    e10.add_argument("--epochs", type=int, default=20)
    e10.add_argument("--lr", type=float, default=1e-3)
    e10.add_argument("--batch", type=int, default=256)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "extract_c40":
        cmd_extract_c40(args)
    elif args.cmd == "eval_c40":
        cmd_eval_c40(args)
    elif args.cmd == "extract_robust":
        cmd_extract_robust(args)
    elif args.cmd == "eval_robust":
        cmd_eval_robust(args)
    elif args.cmd == "calibrate":
        cmd_calibrate(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
