"""
W3: PhysioNet training on pre-extracted PNGs.

Loads consecutive PNG frames from FaceForensics++_c23_processed as video clips —
no cv2.VideoCapture, pure image loading. This allows batch_size=4-8 with clip_len=16
while keeping per-step time under 2s on T4 (vs 10s+ with video decoding).

Workflow (3 stages, run in order):
  Stage 1: Load baseline backbone weights → freeze backbone → train temporal + heads only
           Proves temporal encoder can learn on top of frozen backbone.
           --baseline_ckpt /kaggle/working/checkpoints/baseline_best.pt
           --freeze_backbone_epochs 5  (backbone frozen, only temporal/heads trained)
           --epochs 10  --w_pulse 0  --w_blink 0

  Stage 2: Unfreeze backbone → fine-tune everything end-to-end (classification only)
           --baseline_ckpt ...  --freeze_backbone_epochs 0  --epochs 20

  Stage 3: Enable physio losses → full PhysioNet multi-task
           --w_pulse 0.4  --w_blink 0.3  --w_contrastive 0.1

Usage (Kaggle, Stage 1):
    python w3_train/train_physio_png.py \
        --ff_root /kaggle/input/ff-c23-processed/FaceForensics++_c23_processed \
        --baseline_ckpt /kaggle/working/checkpoints/baseline_best.pt \
        --out_dir /kaggle/working/checkpoints \
        --log_dir /kaggle/working/logs \
        --epochs 10 --clip_len 16 --batch_size 6 \
        --freeze_backbone_epochs 5 \
        --w_pulse 0.0 --w_blink 0.0 \
        --run_name physio_png_stage1
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    raise ImportError("sklearn required: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig
from w2_model.losses import PhysioMultiTaskLoss

try:
    from w1_setup.trackio_init import ExperimentLogger
    TRACKIO_AVAILABLE = True
except Exception:
    TRACKIO_AVAILABLE = False


# ─── Constants ───────────────────────────────────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Dataset ─────────────────────────────────────────────────────────────────

def scan_video_folders(ff_root: str) -> Tuple[List[str], List[int], List[str]]:
    """Scan FF++ processed directory. Returns (video_dirs, labels, src_ids)."""
    ff_root = Path(ff_root)
    video_dirs, labels, src_ids = [], [], []

    for manip, label in FF_MANIPULATION_TYPES.items():
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            print(f"  {manip}: NOT FOUND at {manip_dir}")
            continue

        subdirs = sorted([d for d in manip_dir.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"  {manip}: no subfolders found")
            continue

        valid = 0
        for sd in subdirs:
            has_frames = any(sd.glob("*.png")) or any(sd.glob("*.jpg"))
            if not has_frames:
                continue
            video_dirs.append(str(sd))
            labels.append(label)
            src_ids.append(sd.name.split("_")[0])
            valid += 1

        print(f"  {manip}: {valid} video folders")

    return video_dirs, labels, src_ids


class PNGClipDataset(Dataset):
    """
    Loads a clip of T consecutive PNG frames from a video folder.
    Much faster than cv2.VideoCapture — pure image I/O.
    """

    def __init__(
        self,
        video_dirs: List[str],
        labels: List[int],
        clip_len: int = 16,
        img_size: int = 224,
        augment: bool = False,
        clips_per_video: int = 1,
        max_videos: Optional[int] = None,
        rppg_cache_dir: Optional[str] = None,
        rppg_dim: int = 128,
        rppg_version: int = 1,
        blink_cache_dir: Optional[str] = None,
    ):
        if max_videos is not None and max_videos < len(video_dirs):
            # Stratified cap: keep proportional real/fake ratio
            indices = list(range(len(video_dirs)))
            random.shuffle(indices)
            indices = indices[:max_videos]
            video_dirs = [video_dirs[i] for i in indices]
            labels     = [labels[i]     for i in indices]
        self.video_dirs = video_dirs
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.augment = augment
        self.clips_per_video = clips_per_video
        self.rppg_cache_dir = Path(rppg_cache_dir) if rppg_cache_dir else None
        self.rppg_dim = rppg_dim
        self.rppg_version = rppg_version  # 1 = v1 (128-d FFT), 2 = v2 (12-d sync)
        self.blink_cache_dir = Path(blink_cache_dir) if blink_cache_dir else None

        # Pre-scan frame lists
        self.frame_lists: List[List[str]] = []
        for vd in video_dirs:
            frames = sorted([
                f for f in os.listdir(vd)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.frame_lists.append(frames)

    def __len__(self) -> int:
        return len(self.video_dirs) * self.clips_per_video

    def __getitem__(self, idx: int) -> dict:
        video_idx = idx % len(self.video_dirs)
        vdir = self.video_dirs[video_idx]
        label = self.labels[video_idx]
        all_frames = self.frame_lists[video_idx]
        n = len(all_frames)

        if n == 0:
            clip = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            max_start = max(0, n - self.clip_len)
            if self.augment:
                # Training: random start from middle 80%
                lo = int(max_start * 0.1)
                hi = max(lo, int(max_start * 0.9))
                start = random.randint(lo, hi)
            else:
                # Val/test: fixed center clip — deterministic, no epoch-to-epoch variance
                start = max_start // 2

            # Gather clip_len frames, cycling if video is shorter
            indices = [(start + i) % n for i in range(self.clip_len)]
            frames = []
            for fi in indices:
                fpath = os.path.join(vdir, all_frames[fi])
                img = cv2.imread(fpath)
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                frames.append(img)

            clip = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3) BGR

        # BGR → RGB for all frames in clip
        clip = clip[:, :, :, ::-1].copy()

        # Augmentation — clip-consistent (same transform applied to all frames)
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                clip = clip[:, :, ::-1, :].copy()
            # Brightness jitter — same factor for temporal consistency
            if random.random() > 0.5:
                factor = random.uniform(0.85, 1.15)
                clip = np.clip(clip * factor, 0, 1).astype(np.float32)
            # Color jitter — per-channel slight shift (temporal consistent)
            if random.random() > 0.5:
                for c in range(3):
                    shift = random.uniform(-0.05, 0.05)
                    clip[:, :, :, c] = np.clip(clip[:, :, :, c] + shift, 0, 1)
            # Random grayscale (destroys color artifacts, forces shape learning)
            if random.random() > 0.85:
                gray = clip[:, :, :, 0] * 0.299 + clip[:, :, :, 1] * 0.587 + clip[:, :, :, 2] * 0.114
                clip[:, :, :, 0] = clip[:, :, :, 1] = clip[:, :, :, 2] = gray
            # Random erasing — mask a patch across all frames (forces non-local features)
            if random.random() > 0.5:
                h, w = clip.shape[1], clip.shape[2]
                eh, ew = random.randint(h // 8, h // 3), random.randint(w // 8, w // 3)
                ey, ex = random.randint(0, h - eh), random.randint(0, w - ew)
                clip[:, ey:ey+eh, ex:ex+ew, :] = np.random.uniform(0, 1, (clip.shape[0], eh, ew, 3)).astype(np.float32)

        # ImageNet normalize: (T, H, W, 3) → (T, 3, H, W)
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip_tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float()  # (T, 3, H, W)

        # Load pre-extracted rPPG feature from cache dir if available
        # v1: cache_dir/<manip>/<video_name>/rppg_feat.npy     (128-d FFT)
        # v2: cache_dir/<manip>/<video_name>/rppg_v2_feat.npy  (12-d sync features)
        rppg_feat = torch.zeros(self.rppg_dim)
        if self.rppg_cache_dir is not None:
            vpath = Path(vdir)
            if self.rppg_version == 2:
                cache_feat = self.rppg_cache_dir / vpath.parent.name / vpath.name / "rppg_v2_feat.npy"
            else:
                cache_feat = self.rppg_cache_dir / vpath.parent.name / vpath.name / "rppg_feat.npy"
            if cache_feat.exists():
                loaded = np.load(str(cache_feat)).astype(np.float32)
                if len(loaded) == self.rppg_dim:
                    rppg_feat = torch.from_numpy(loaded)
                elif len(loaded) < self.rppg_dim:
                    # Zero-pad shorter features (e.g., 12-d v2 → 128-d to match checkpoint)
                    padded = np.zeros(self.rppg_dim, dtype=np.float32)
                    padded[:len(loaded)] = loaded
                    rppg_feat = torch.from_numpy(padded)

        # Load pre-extracted blink features from cache if available
        # cache_dir/<manip>/<video_name>/blink_feat.npy  (16-d)
        blink_feat = torch.zeros(16)
        if self.blink_cache_dir is not None:
            vpath_b = Path(vdir)
            blink_path = self.blink_cache_dir / vpath_b.parent.name / vpath_b.name / "blink_feat.npy"
            if blink_path.exists():
                loaded_b = np.load(str(blink_path)).astype(np.float32)
                if len(loaded_b) == 16:
                    blink_feat = torch.from_numpy(loaded_b)

        # Frame differences: I(t) - I(t-1) — captures temporal motion artifacts
        frame_diffs = clip_tensor[1:] - clip_tensor[:-1]     # (T-1, 3, H, W)
        # Pad to length T by repeating last diff
        frame_diffs = torch.cat([frame_diffs, frame_diffs[-1:]], dim=0)  # (T, 3, H, W)

        return {
            "frames": clip_tensor,               # (T, 3, H, W)
            "frame_diffs": frame_diffs,           # (T, 3, H, W) — temporal motion
            "label": torch.tensor(float(label), dtype=torch.float32),
            "rppg_feat": rppg_feat,              # (rppg_dim,) — sync features or zeros
            "blink_feat": blink_feat,              # (16,) — blink stats or zeros
        }


# ─── Backbone weight loading ──────────────────────────────────────────────────

def load_baseline_backbone(model: PhysioNet, baseline_ckpt_path: str, device: torch.device):
    """
    Load EfficientNet-B4 backbone weights from a SimpleClassifier checkpoint
    (train_baseline.py) into PhysioNet's frame_encoder.
    """
    ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # SimpleClassifier has: backbone.* and head.*
    # PhysioNet frame_encoder has: frame_encoder.encoder.*
    backbone_weights = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    missing, unexpected = model.frame_encoder.encoder.load_state_dict(
        backbone_weights, strict=False
    )
    print(f"  Backbone weights loaded: {len(backbone_weights)} tensors")
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


# ─── Training utils ──────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ─── Main training ────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    logger = None
    if TRACKIO_AVAILABLE:
        try:
            logger = ExperimentLogger(
                project="p3_physio_deepfake",
                run_name=args.run_name or "physio_png",
                config=vars(args),
                local_log_dir=args.log_dir,
            )
        except Exception as e:
            print(f"[WARN] Trackio init failed: {e}")
    if logger is None:
        print(f"[INFO] Logging to CSV only: {args.log_dir}/{args.run_name}_metrics.csv")

    # ─── Data ────────────────────────────────────────────────────────────
    print("\nScanning dataset...")
    video_dirs, labels, src_ids = scan_video_folders(args.ff_root)
    n_total = len(video_dirs)
    print(f"Total: {n_total} videos, real={labels.count(0)}, fake={labels.count(1)}")

    if n_total == 0:
        print("ERROR: No data found. Check --ff_root path.")
        return

    # Identity-aware split
    id_to_indices: Dict[str, List[int]] = {}
    for i, sid in enumerate(src_ids):
        id_to_indices.setdefault(sid, []).append(i)

    unique_ids = sorted(id_to_indices.keys())
    # Fixed seed for split — val set must be identical across all runs
    rng = random.Random(42)
    rng.shuffle(unique_ids)
    n_ids = len(unique_ids)
    n_train_ids = int(n_ids * 0.8)
    n_val_ids   = int(n_ids * 0.1)

    train_ids_set = set(unique_ids[:n_train_ids])
    val_ids_set   = set(unique_ids[n_train_ids:n_train_ids + n_val_ids])

    train_idx, val_idx, test_idx = [], [], []
    for sid, indices in id_to_indices.items():
        if sid in train_ids_set:
            train_idx.extend(indices)
        elif sid in val_ids_set:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    random.shuffle(train_idx)
    random.shuffle(val_idx)

    train_dirs   = [video_dirs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_dirs     = [video_dirs[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_dirs    = [video_dirs[i] for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    print(f"Split: {n_train_ids}/{n_val_ids}/{n_ids-n_train_ids-n_val_ids} source IDs")

    rppg_cache = args.rppg_cache if hasattr(args, "rppg_cache") else None
    rppg_dim = args.rppg_dim if hasattr(args, "rppg_dim") else 128
    rppg_ver = args.rppg_version if hasattr(args, "rppg_version") else 1
    blink_cache = args.blink_cache if hasattr(args, "blink_cache") else None
    train_ds = PNGClipDataset(train_dirs, train_labels, args.clip_len, args.img_size,
                               augment=True, clips_per_video=args.clips_per_video,
                               max_videos=args.max_train_videos,
                               rppg_cache_dir=rppg_cache,
                               rppg_dim=rppg_dim, rppg_version=rppg_ver,
                               blink_cache_dir=blink_cache)
    val_ds   = PNGClipDataset(val_dirs,   val_labels,   args.clip_len, args.img_size,
                               rppg_cache_dir=rppg_cache,
                               rppg_dim=rppg_dim, rppg_version=rppg_ver,
                               blink_cache_dir=blink_cache)
    test_ds  = PNGClipDataset(test_dirs,  test_labels,  args.clip_len, args.img_size,
                               rppg_cache_dir=rppg_cache,
                               rppg_dim=rppg_dim, rppg_version=rppg_ver,
                               blink_cache_dir=blink_cache)

    # Balanced sampler — use actual dataset labels (may be capped)
    tl = np.array(train_ds.labels)
    n_real = int((tl == 0).sum())
    n_fake = int((tl == 1).sum())
    per_video_w = np.where(tl == 0, 1.0 / (n_real + 1), 1.0 / (n_fake + 1))
    weights = np.tile(per_video_w, args.clips_per_video)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} clips ({n_real} real vids, {n_fake} fake vids), "
          f"{len(train_dl)} steps/epoch")
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ─── rPPG cache diagnostic ────────────────────────────────────────────
    if rppg_cache:
        feat_file = "rppg_v2_feat.npy" if rppg_ver == 2 else "rppg_feat.npy"
        n_with_feat = 0
        feat_norms = []
        pcc_values = []
        for vdir in train_ds.video_dirs[:200]:  # sample first 200 train videos
            vpath = Path(vdir)
            cache_feat = Path(rppg_cache) / vpath.parent.name / vpath.name / feat_file
            if cache_feat.exists():
                feat = np.load(str(cache_feat))
                if len(feat) == rppg_dim:
                    n_with_feat += 1
                    feat_norms.append(float(np.linalg.norm(feat)))
                    if rppg_ver == 2 and len(feat) >= 7:
                        pcc_values.append(float(feat[6]))  # pcc_raw is index 6
        pct = 100 * n_with_feat / min(200, len(train_ds.video_dirs))
        mean_norm = float(np.mean(feat_norms)) if feat_norms else 0.0
        print(f"rPPG v{rppg_ver} cache: {pct:.0f}% of sampled train videos have features "
              f"(mean L2-norm={mean_norm:.3f})")
        if rppg_ver == 2 and pcc_values:
            print(f"  Mean PCC (train sample): {np.mean(pcc_values):.3f} ± {np.std(pcc_values):.3f}")
        if pct < 50:
            print("  [WARN] Less than 50% of videos have rPPG features — cache path may be wrong")

    # ─── Blink cache diagnostic ───────────────────────────────────────────
    if blink_cache:
        n_blink_found = 0
        n_blink_zero = 0
        blink_norms = []
        for vdir in train_ds.video_dirs[:200]:
            vpath = Path(vdir)
            bp = Path(blink_cache) / vpath.parent.name / vpath.name / "blink_feat.npy"
            if bp.exists():
                feat = np.load(str(bp)).astype(np.float32)
                if len(feat) == 16:
                    n_blink_found += 1
                    norm = float(np.linalg.norm(feat))
                    blink_norms.append(norm)
                    if np.allclose(feat, 0):
                        n_blink_zero += 1
        pct_b = 100 * n_blink_found / min(200, len(train_ds.video_dirs))
        mean_norm_b = float(np.mean(blink_norms)) if blink_norms else 0.0
        print(f"Blink cache: {pct_b:.0f}% of sampled train videos have features "
              f"(mean L2-norm={mean_norm_b:.3f}, zeros={n_blink_zero})")
        if pct_b < 50:
            print("  [WARN] Less than 50% of videos have blink features — cache path may be wrong")
    else:
        print("Blink cache: NOT configured (blink features will be zeros)")

    # ─── Model ───────────────────────────────────────────────────────────
    use_physio = (args.w_pulse > 0 or args.w_blink > 0 or args.use_rppg_fusion)
    use_motion = getattr(args, 'use_motion', False)
    local_wt = getattr(args, 'backbone_weights', None)
    cfg = ModelConfig(
        backbone="efficientnet_b4",
        backbone_pretrained=(local_wt is None and args.baseline_ckpt is None and not (hasattr(args, 'resume_ckpt') and args.resume_ckpt)),
        backbone_local_weights=local_wt,
        temporal_model=args.temporal_model,
        temporal_layers=args.temporal_layers,
        temporal_dim=args.temporal_dim,
        clip_len=args.clip_len,
        img_size=args.img_size,
        dropout=args.dropout,
        use_pulse_head=(args.w_pulse > 0),
        use_blink_head=(args.w_blink > 0),
        use_physio_fusion=use_physio,
        temporal_pool="transformer" if args.temporal_model != "mean" else "mean",
        rppg_feature_dim=rppg_dim,
        use_motion_model=use_motion,
    )
    model = PhysioNet(cfg).to(device)

    # Load full PhysioNet checkpoint (resume_ckpt takes priority over baseline_ckpt)
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        print(f"\nResuming from PhysioNet checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Filter out tensors with shape mismatch (e.g., fusion layer when rppg_dim changed)
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped.append(f"{k}: ckpt {list(v.shape)} vs model {list(model_state[k].shape)}")
            else:
                filtered_state[k] = v
        if skipped:
            print(f"  Skipped {len(skipped)} size-mismatched keys (will init randomly):")
            for s in skipped:
                print(f"    {s}")
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded {len(filtered_state)}/{len(state)} tensors")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        prev_auc = ckpt.get("val_auc", 0.0)
        print(f"  Previous checkpoint val_auc={prev_auc:.4f} (epoch {ckpt.get('epoch', '?')})")
    # Load pretrained baseline backbone (only backbone weights from SimpleClassifier)
    elif args.baseline_ckpt and Path(args.baseline_ckpt).exists():
        print(f"\nLoading baseline backbone from: {args.baseline_ckpt}")
        load_baseline_backbone(model, args.baseline_ckpt, device)
    else:
        print("\nNo baseline checkpoint — using ImageNet pretrained backbone")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M params")

    # Freeze backbone initially
    if args.freeze_backbone_epochs > 0:
        model.freeze_backbone(True)
        print(f"Backbone FROZEN for first {args.freeze_backbone_epochs} epochs")

    # ─── Optimizer ───────────────────────────────────────────────────────
    backbone_params = list(model.frame_encoder.parameters())
    other_params    = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]

    # Only include backbone in optimizer if it will ever be unfrozen
    backbone_ever_unfrozen = (args.freeze_backbone_epochs < args.epochs)
    if backbone_ever_unfrozen and args.lr_backbone > 0:
        param_groups = [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params,    "lr": args.lr_head},
        ]
    else:
        # Backbone frozen for entire run — don't include in optimizer at all
        param_groups = [
            {"params": other_params, "lr": args.lr_head},
        ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # LR schedule:
    #   flat_then_cosine — keep full LR during frozen phase, cosine only after unfreeze
    #   cosine           — warmup then cosine over all epochs (original behavior)
    freeze_ep = args.freeze_backbone_epochs
    warmup_epochs = min(2, max(1, freeze_ep // 5)) if freeze_ep > 0 else min(2, args.epochs // 5)
    if args.lr_schedule == "flat_then_cosine" and freeze_ep > 0:
        finetune_epochs = max(1, args.epochs - freeze_ep)
        def lr_lambda(epoch):
            # Phase 1 (frozen): warmup then flat
            if epoch < freeze_ep:
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0  # flat at full LR
            # Phase 2 (unfrozen): cosine decay over finetune_epochs
            progress = (epoch - freeze_ep) / finetune_epochs
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Original cosine over all epochs
        warmup_epochs = min(2, args.epochs // 5)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    # ─── Loss ────────────────────────────────────────────────────────────
    label_smooth = getattr(args, 'label_smoothing', 0.0)
    if label_smooth > 0:
        print(f"Label smoothing: {label_smooth} (targets: {label_smooth/2:.3f} / {1-label_smooth/2:.3f})")
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    best_epoch = 0
    ema_val_auc = None   # EMA of val_auc — smoother signal for checkpoint saving
    start_time = time.time()

    # ─── Training loop ───────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after freeze period
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            model.freeze_backbone(False)
            print(f"  >> Backbone UNFROZEN at epoch {epoch}")

        model.train()
        losses, all_preds, all_targets = [], [], []

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        _blink_checked = False
        for batch_idx, batch in enumerate(pbar):
            frames   = batch["frames"].to(device, non_blocking=True)       # (B, T, 3, H, W)
            labels_b = batch["label"].to(device, non_blocking=True)
            rppg     = batch["rppg_feat"].to(device, non_blocking=True)    # (B, rppg_dim)
            blink    = batch["blink_feat"].to(device, non_blocking=True)   # (B, 16)
            diffs    = batch["frame_diffs"].to(device, non_blocking=True) if use_motion else None

            # First-batch sanity check: are blink features non-zero?
            if epoch == 1 and not _blink_checked:
                _blink_checked = True
                blink_nonzero = (blink.abs().sum(dim=1) > 0).float().mean().item()
                rppg_nonzero = (rppg.abs().sum(dim=1) > 0).float().mean().item()
                print(f"  [SANITY] Batch 0: blink non-zero: {blink_nonzero*100:.0f}%, "
                      f"rppg non-zero: {rppg_nonzero*100:.0f}%, "
                      f"blink L2-norm: {blink.norm(dim=1).mean().item():.3f}")

            optimizer.zero_grad(set_to_none=True)

            # Apply label smoothing: 0→ε/2, 1→1-ε/2
            targets = labels_b
            if label_smooth > 0:
                targets = targets * (1 - label_smooth) + label_smooth / 2

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                outputs = model(frames, rppg if use_physio else None,
                                        blink if use_physio else None,
                                        diffs)
                logits = outputs["logit"]
                loss = criterion(logits, targets)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            with torch.no_grad():
                probs = torch.sigmoid(logits.float().clamp(-20, 20)).cpu().numpy()
                probs = np.nan_to_num(probs, nan=0.5)
                all_preds.extend(probs.tolist())
                all_targets.extend(labels_b.cpu().numpy().tolist())

        scheduler.step()

        preds_arr = np.array(all_preds)
        n_nan = np.isnan(preds_arr).sum()
        if n_nan > 0:
            print(f"  [WARN] {n_nan} NaN predictions in train (replaced with 0.5)")
            preds_arr = np.nan_to_num(preds_arr, nan=0.5)
            all_preds = preds_arr.tolist()
        train_auc = roc_auc_score(all_targets, all_preds)
        print(f"  [Train] pred_mean={preds_arr.mean():.3f} pred_std={preds_arr.std():.3f} "
              f"frac>0.5={(preds_arr > 0.5).mean():.3f}")

        # ─── Val ─────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Val", leave=False):
                frames   = batch["frames"].to(device, non_blocking=True)
                rppg     = batch["rppg_feat"].to(device, non_blocking=True)
                blink    = batch["blink_feat"].to(device, non_blocking=True)
                diffs    = batch["frame_diffs"].to(device, non_blocking=True) if use_motion else None
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    outputs = model(frames, rppg if use_physio else None,
                                            blink if use_physio else None,
                                            diffs)
                probs = torch.sigmoid(outputs["logit"].float().clamp(-20, 20)).cpu().numpy()
                probs = np.nan_to_num(probs, nan=0.5)
                val_preds.extend(probs.tolist())
                val_targets.extend(batch["label"].numpy().tolist())

        val_preds_arr   = np.nan_to_num(np.array(val_preds), nan=0.5)
        val_targets_arr = np.array(val_targets)
        val_auc = roc_auc_score(val_targets_arr, val_preds_arr)
        val_eer = compute_eer(val_preds_arr, val_targets_arr)

        # EMA smoothing (α=0.3) — reduces checkpoint noise from 600-sample val set
        ema_alpha = 0.3
        ema_val_auc = val_auc if ema_val_auc is None else (ema_alpha * val_auc + (1 - ema_alpha) * ema_val_auc)

        avg_loss    = float(np.mean(losses))
        epoch_time  = time.time() - epoch_start
        total_time  = time.time() - start_time

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"train_auc={train_auc:.4f} | "
              f"val_auc={val_auc:.4f} (ema={ema_val_auc:.4f}) val_eer={val_eer:.4f} | "
              f"pred_std={val_preds_arr.std():.3f} | "
              f"time={epoch_time:.0f}s total={total_time/60:.1f}min")

        if logger:
            try:
                logger.log({
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/auc": train_auc,
                    "val/auc": val_auc,
                    "val/eer": val_eer,
                    "lr_backbone": optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                    "lr_head":     optimizer.param_groups[-1]["lr"],
                }, step=epoch)
            except Exception:
                pass

        if ema_val_auc > best_auc:
            best_auc = ema_val_auc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_eer": val_eer,
                "cfg": cfg,
                "args": vars(args),
            }, out_dir / f"{args.run_name}_best.pt")
            print(f"  >> New best AUC={val_auc:.4f}")

        patience = getattr(args, 'patience', 0)
        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"  >> Early stopping: no improvement for {patience} epochs (best at epoch {best_epoch})")
            break

        if total_time > 10.5 * 3600:
            print(f"  >> Time limit ({total_time/3600:.1f}h), stopping")
            break

    # ─── Test ────────────────────────────────────────────────────────────
    ckpt_path = out_dir / f"{args.run_name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, val_auc={ckpt['val_auc']:.4f})")
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Test", leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            rppg   = batch["rppg_feat"].to(device, non_blocking=True)
            blink  = batch["blink_feat"].to(device, non_blocking=True)
            diffs  = batch["frame_diffs"].to(device, non_blocking=True) if use_motion else None
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                outputs = model(frames, rppg if use_physio else None,
                                        blink if use_physio else None,
                                        diffs)
            probs = torch.sigmoid(outputs["logit"].float().clamp(-20, 20)).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.5)
            test_preds.extend(probs.tolist())
            test_targets.extend(batch["label"].numpy().tolist())

    test_auc = roc_auc_score(test_targets, np.nan_to_num(test_preds, nan=0.5))
    test_eer = compute_eer(np.array(test_preds), np.array(test_targets))
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"FINAL TEST:  AUC={test_auc:.4f}  EER={test_eer:.4f}")
    print(f"Best val:    AUC={best_auc:.4f}")
    print(f"Total time:  {total_time / 60:.1f} min")
    print("=" * 60)

    if logger:
        try:
            logger.log_summary({
                "test/auc": test_auc,
                "test/eer": test_eer,
                "best_val_auc": best_auc,
                "total_time_min": total_time / 60,
            })
            logger.finish()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="PhysioNet training on pre-extracted PNGs")
    p.add_argument("--ff_root", required=True,
                   help="Path to FaceForensics++_c23_processed")
    p.add_argument("--baseline_ckpt", default=None,
                   help="Path to baseline_best.pt to init backbone weights only")
    p.add_argument("--resume_ckpt", default=None,
                   help="Path to a previous PhysioNet checkpoint to resume all weights from")
    p.add_argument("--backbone_weights", default=None,
                   help="Path to local .pth file for backbone (skips HF download)")
    p.add_argument("--out_dir",  default="./checkpoints")
    p.add_argument("--log_dir",  default="./logs")
    p.add_argument("--run_name", default="physio_png_v1")

    # Architecture
    p.add_argument("--temporal_model",  default="transformer",
                   choices=["transformer", "lstm", "mean"])
    p.add_argument("--temporal_layers", type=int, default=2)
    p.add_argument("--temporal_dim",    type=int, default=512)
    p.add_argument("--clip_len",        type=int, default=16)
    p.add_argument("--img_size",        type=int, default=224)
    p.add_argument("--dropout",         type=float, default=0.3)

    # Loss weights
    p.add_argument("--w_pulse",       type=float, default=0.0)
    p.add_argument("--w_blink",       type=float, default=0.0)
    p.add_argument("--w_contrastive", type=float, default=0.0)

    # Training
    p.add_argument("--epochs",                 type=int,   default=20)
    p.add_argument("--batch_size",             type=int,   default=6)
    p.add_argument("--clips_per_video",        type=int,   default=1,
                   help="Clips sampled per video per epoch")
    p.add_argument("--max_train_videos",       type=int,   default=None,
                   help="Cap training videos per epoch (None=use all)")
    p.add_argument("--lr_backbone",            type=float, default=1e-5)
    p.add_argument("--lr_head",                type=float, default=1e-4)
    p.add_argument("--weight_decay",           type=float, default=1e-4)
    p.add_argument("--freeze_backbone_epochs", type=int,   default=3)
    p.add_argument("--lr_schedule", default="flat_then_cosine",
                   choices=["cosine", "flat_then_cosine"])
    p.add_argument("--use_rppg_fusion", action="store_true", default=False,
                   help="Fuse pre-extracted rPPG features into classifier")
    p.add_argument("--rppg_cache", default=None,
                   help="Path to rPPG cache dir (output of extract_rppg_png.py or v2)")
    p.add_argument("--blink_cache", default=None,
                   help="Path to blink cache dir (output of extract_blinks_mp4.py)")
    p.add_argument("--rppg_dim", type=int, default=128,
                   help="rPPG feature dimension: 128 for v1 (FFT), 12 for v2 (sync)")
    p.add_argument("--rppg_version", type=int, default=1, choices=[1, 2],
                   help="rPPG version: 1=128-d FFT (old), 2=12-d L/R cheek sync (new)")
    p.add_argument("--use_motion", action="store_true", default=False,
                   help="Add frame-difference motion branch")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing factor (e.g., 0.1). Targets become ε/2 and 1-ε/2")
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping patience (0=disabled). Stop after N epochs without improvement.")
    p.add_argument("--num_workers",            type=int,   default=2)
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
