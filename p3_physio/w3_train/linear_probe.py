"""
Linear probe evaluation of frozen backbone + per-manipulation breakdown.

Loads v13 backbone, freezes everything, trains a single linear layer.
Reports overall AUC + per-manipulation AUC + optional TTA.

This reveals the true feature quality ceiling without any fine-tuning noise.

Usage:
    python w3_train/linear_probe.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --backbone_weights /kaggle/input/.../tf_efficientnet_b4_aa-818f208c.pth \
        --epochs 15 --batch_size 8 --lr 1e-3 \
        --tta_clips 5 \
        --run_name linear_probe_v1
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
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

try:
    import trackio
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

def scan_video_folders(ff_root: str) -> Tuple[List[str], List[int], List[str], List[str]]:
    """Returns (video_dirs, labels, src_ids, manip_names)."""
    ff_root = Path(ff_root)
    video_dirs, labels, src_ids, manip_names = [], [], [], []

    for manip, label in FF_MANIPULATION_TYPES.items():
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            continue
        subdirs = sorted([d for d in manip_dir.iterdir() if d.is_dir()])
        valid = 0
        for sd in subdirs:
            has_frames = any(sd.glob("*.png")) or any(sd.glob("*.jpg"))
            if not has_frames:
                continue
            video_dirs.append(str(sd))
            labels.append(label)
            src_ids.append(sd.name.split("_")[0])
            manip_names.append(manip)
            valid += 1
        print(f"  {manip}: {valid} video folders")

    return video_dirs, labels, src_ids, manip_names


class ProbeDataset(Dataset):
    """Loads a clip of T consecutive PNG frames. Supports multiple clips per video for TTA."""

    def __init__(self, video_dirs, labels, manips, clip_len=16, img_size=224,
                 augment=False, n_clips=1):
        self.video_dirs = video_dirs
        self.labels = labels
        self.manips = manips
        self.clip_len = clip_len
        self.img_size = img_size
        self.augment = augment
        self.n_clips = n_clips  # >1 for TTA

        self.frame_lists = []
        for vd in video_dirs:
            frames = sorted([
                f for f in os.listdir(vd)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.frame_lists.append(frames)

    def __len__(self):
        return len(self.video_dirs) * self.n_clips

    def __getitem__(self, idx):
        video_idx = idx // self.n_clips
        clip_idx = idx % self.n_clips

        vdir = self.video_dirs[video_idx]
        label = self.labels[video_idx]
        manip = self.manips[video_idx]
        all_frames = self.frame_lists[video_idx]
        n = len(all_frames)

        if n == 0:
            clip = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            max_start = max(0, n - self.clip_len)
            if self.n_clips > 1:
                # TTA: spread clips evenly across the video
                if self.n_clips == 1:
                    start = max_start // 2
                else:
                    step = max_start / max(1, self.n_clips - 1)
                    start = int(clip_idx * step)
            elif self.augment:
                lo = int(max_start * 0.1)
                hi = max(lo, int(max_start * 0.9))
                start = random.randint(lo, hi)
            else:
                start = max_start // 2

            indices = [(start + i) % n for i in range(self.clip_len)]
            imgs = []
            for fi in indices:
                fpath = os.path.join(vdir, all_frames[fi])
                img = cv2.imread(fpath)
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                imgs.append(img)
            clip = np.stack(imgs, axis=0).astype(np.float32) / 255.0

        # Augmentations (training only)
        if self.augment:
            if random.random() > 0.5:
                clip = clip[:, :, ::-1, :].copy()  # horizontal flip
            if random.random() > 0.5:
                for c in range(3):
                    shift = random.uniform(-0.05, 0.05)
                    clip[:, :, :, c] = np.clip(clip[:, :, :, c] + shift, 0, 1)

        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip_tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float()

        return {
            "frames": clip_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "manip": manip,
            "video_idx": video_idx,
        }


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Logger
    logger = None
    if TRACKIO_AVAILABLE:
        try:
            trackio.init(project="p3_physio_deepfake")
            logger = trackio.Run(name=args.run_name)
            print(f"[Trackio] Initialized run '{args.run_name}'")
        except Exception as e:
            print(f"[Trackio] Init failed: {e}")

    # ─── Data ────────────────────────────────────────────────────────────
    print("\nScanning dataset...")
    video_dirs, labels, src_ids, manip_names = scan_video_folders(args.ff_root)
    n_total = len(video_dirs)
    print(f"Total: {n_total} videos, real={labels.count(0)}, fake={labels.count(1)}")

    # Identity-aware split (same as train_physio_png.py)
    id_to_indices = {}
    for i, sid in enumerate(src_ids):
        id_to_indices.setdefault(sid, []).append(i)

    unique_ids = sorted(id_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_ids)
    n_ids = len(unique_ids)
    n_train_ids = int(n_ids * 0.8)
    n_val_ids = int(n_ids * 0.1)

    train_ids_set = set(unique_ids[:n_train_ids])
    val_ids_set = set(unique_ids[n_train_ids:n_train_ids + n_val_ids])

    train_idx, val_idx, test_idx = [], [], []
    for sid, indices in id_to_indices.items():
        if sid in train_ids_set:
            train_idx.extend(indices)
        elif sid in val_ids_set:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    def subset(indices):
        return ([video_dirs[i] for i in indices],
                [labels[i] for i in indices],
                [manip_names[i] for i in indices])

    train_dirs, train_labels, train_manips = subset(train_idx)
    val_dirs, val_labels, val_manips = subset(val_idx)
    test_dirs, test_labels, test_manips = subset(test_idx)

    print(f"Split: {n_train_ids}/{n_val_ids}/{n_ids-n_train_ids-n_val_ids} source IDs")
    print(f"Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)}")

    train_ds = ProbeDataset(train_dirs, train_labels, train_manips,
                            args.clip_len, args.img_size, augment=True)
    val_ds = ProbeDataset(val_dirs, val_labels, val_manips,
                          args.clip_len, args.img_size)
    test_ds = ProbeDataset(test_dirs, test_labels, test_manips,
                           args.clip_len, args.img_size)
    # TTA version for final eval
    test_tta_ds = ProbeDataset(test_dirs, test_labels, test_manips,
                               args.clip_len, args.img_size, n_clips=args.tta_clips)

    # Balanced sampler
    tl = np.array(train_labels)
    n_real = int((tl == 0).sum())
    n_fake = int((tl == 1).sum())
    per_video_w = np.where(tl == 0, 1.0 / (n_real + 1), 1.0 / (n_fake + 1))
    sampler = WeightedRandomSampler(per_video_w, len(per_video_w), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    test_tta_dl = DataLoader(test_tta_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ─── Model ───────────────────────────────────────────────────────────
    # Build PhysioNet but we only use its backbone
    cfg = ModelConfig(
        backbone="efficientnet_b4",
        backbone_pretrained=False,
        backbone_local_weights=args.backbone_weights,
        temporal_model="mean",
        temporal_dim=0,
        clip_len=args.clip_len,
        img_size=args.img_size,
        dropout=0.0,
        use_physio_fusion=False,
        use_pulse_head=False,
        use_blink_head=False,
        use_motion_model=False,
    )
    model = PhysioNet(cfg).to(device)

    # Load checkpoint (only backbone weights matter)
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        print(f"\nLoading backbone from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Only load frame_encoder weights
        backbone_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded {len(backbone_state)} backbone tensors")
        print(f"  Missing: {len(missing)} (expected — non-backbone keys)")

    # Freeze entire model
    for param in model.parameters():
        param.requires_grad = False

    # Extract backbone feature dim
    feat_dim = model.frame_encoder.out_dim  # 1792 for efficientnet_b4
    print(f"Backbone feature dim: {feat_dim}")

    # Linear probe head: features → logit
    probe_head = nn.Linear(feat_dim, 1).to(device)
    print(f"Probe head: {sum(p.numel() for p in probe_head.parameters())} params")

    # Also test: features + blink/rppg → logit
    if args.blink_cache or args.rppg_cache:
        physio_dim = 0
        if args.rppg_cache:
            physio_dim += args.rppg_dim
        if args.blink_cache:
            physio_dim += 16
        probe_head_physio = nn.Sequential(
            nn.BatchNorm1d(feat_dim + physio_dim),
            nn.Linear(feat_dim + physio_dim, 1),
        ).to(device)
        print(f"Probe head (+ physio): {sum(p.numel() for p in probe_head_physio.parameters())} params "
              f"(backbone {feat_dim} + physio {physio_dim})")
    else:
        probe_head_physio = None

    optimizer = torch.optim.AdamW(probe_head.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ─── Feature extraction helper ───────────────────────────────────────

    @torch.no_grad()
    def extract_features(dataloader, desc="Extract"):
        """Extract backbone features for all clips."""
        model.eval()
        all_feats, all_labels, all_manips, all_vidx = [], [], [], []
        for batch in tqdm(dataloader, desc=desc, leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                frame_feats = model.frame_encoder(frames)  # (B, T, D)
                pooled = frame_feats.mean(dim=1)  # (B, D) — mean pool over time
            all_feats.append(pooled.float().cpu())
            all_labels.extend(batch["label"].numpy().tolist())
            all_manips.extend(batch["manip"])
            all_vidx.extend(batch["video_idx"].numpy().tolist())
        return torch.cat(all_feats, dim=0), np.array(all_labels), all_manips, np.array(all_vidx)

    # ─── Pre-extract features (huge speedup) ─────────────────────────────
    print("\nExtracting features (one-time)...")
    t0 = time.time()
    train_feats, train_labels_arr, train_manips_arr, _ = extract_features(train_dl, "Train features")
    val_feats, val_labels_arr, val_manips_arr, _ = extract_features(val_dl, "Val features")
    test_feats, test_labels_arr, test_manips_arr, _ = extract_features(test_dl, "Test features")
    # Clean NaN/Inf in features
    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        n_nan = torch.isnan(feats).sum().item()
        if n_nan > 0:
            print(f"  [WARN] {name}: {n_nan} NaN values in features — replacing with 0")
            feats.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Feature extraction: {time.time()-t0:.1f}s")
    print(f"Train: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")

    # ─── TTA features ────────────────────────────────────────────────────
    if args.tta_clips > 1:
        print(f"\nExtracting TTA features ({args.tta_clips} clips per video)...")
        test_tta_feats, test_tta_labels, test_tta_manips, test_tta_vidx = \
            extract_features(test_tta_dl, "TTA features")
        # Average features per video
        unique_vidx = sorted(set(test_tta_vidx.tolist()))
        tta_feats_avg = []
        tta_labels_avg = []
        tta_manips_avg = []
        for vi in unique_vidx:
            mask = test_tta_vidx == vi
            tta_feats_avg.append(test_tta_feats[mask].mean(dim=0))
            tta_labels_avg.append(test_tta_labels[mask][0])
            tta_manips_avg.append(np.array(test_tta_manips)[mask][0])
        test_tta_feats_avg = torch.stack(tta_feats_avg)
        test_tta_feats_avg.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        test_tta_labels_avg = np.array(tta_labels_avg)
        test_tta_manips_avg = tta_manips_avg
        print(f"TTA averaged: {test_tta_feats_avg.shape}")

    # ─── Train linear probe on pre-extracted features ─────────────────────
    print(f"\nTraining linear probe ({args.epochs} epochs)...")
    # Simple feature dataset
    train_feat_tensor = train_feats.to(device)
    train_label_tensor = torch.tensor(train_labels_arr, dtype=torch.float32, device=device)
    val_feat_tensor = val_feats.to(device)
    test_feat_tensor = test_feats.to(device)

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        probe_head.train()
        # Mini-batch training on features
        perm = torch.randperm(len(train_feat_tensor))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), args.probe_batch_size):
            idx = perm[i:i + args.probe_batch_size]
            feats_b = train_feat_tensor[idx]
            labels_b = train_label_tensor[idx]

            logits = probe_head(feats_b).squeeze(-1)
            loss = criterion(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Val
        probe_head.eval()
        with torch.no_grad():
            val_logits = probe_head(val_feat_tensor).squeeze(-1)
            val_probs = torch.sigmoid(val_logits.clamp(-20, 20)).cpu().numpy()
            val_probs = np.nan_to_num(val_probs, nan=0.5)
            val_auc = roc_auc_score(val_labels_arr, val_probs)
            val_eer = compute_eer(val_probs, val_labels_arr)

            train_logits = probe_head(train_feat_tensor).squeeze(-1)
            train_probs = torch.sigmoid(train_logits.clamp(-20, 20)).cpu().numpy()
            train_probs = np.nan_to_num(train_probs, nan=0.5)
            train_auc = roc_auc_score(train_labels_arr, train_probs)

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | train_auc={train_auc:.4f} | "
              f"val_auc={val_auc:.4f} val_eer={val_eer:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in probe_head.state_dict().items()}
            print(f"  >> New best val AUC={val_auc:.4f}")

        if logger:
            try:
                logger.log({"epoch": epoch, "train/loss": avg_loss,
                            "train/auc": train_auc, "val/auc": val_auc, "val/eer": val_eer}, step=epoch)
            except Exception:
                pass

    # ─── Final evaluation ─────────────────────────────────────────────────
    probe_head.load_state_dict(best_state)
    probe_head.eval()

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    def eval_split(feats, labels, manips, name, use_tta_feats=None, tta_labels=None, tta_manips=None):
        with torch.no_grad():
            logits = probe_head(feats.to(device)).squeeze(-1)
            probs = torch.sigmoid(logits.clamp(-20, 20)).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.5)

        auc = roc_auc_score(labels, probs)
        eer = compute_eer(probs, labels)
        print(f"\n  {name}: AUC={auc:.4f}  EER={eer:.4f}  (n={len(labels)})")

        # Per-manipulation breakdown
        manip_results = {}
        for manip in sorted(set(manips)):
            mask = np.array([m == manip for m in manips])
            if mask.sum() < 5:
                continue
            m_labels = labels[mask]
            m_probs = probs[mask]
            # For per-manip AUC, we need both classes. For fake manips, compare against all reals in this split.
            if manip == "original":
                manip_results[manip] = {"n": int(mask.sum()), "label": "real"}
                continue
            # Get real samples in this split
            real_mask = np.array([m == "original" for m in manips])
            combined_labels = np.concatenate([labels[real_mask], m_labels])
            combined_probs = np.concatenate([probs[real_mask], m_probs])
            if len(set(combined_labels)) < 2:
                continue
            m_auc = roc_auc_score(combined_labels, combined_probs)
            m_eer = compute_eer(combined_probs, combined_labels)
            manip_results[manip] = {"auc": m_auc, "eer": m_eer, "n": int(mask.sum())}
            print(f"    {manip:20s}: AUC={m_auc:.4f}  EER={m_eer:.4f}  (n={mask.sum()})")

        # TTA evaluation
        if use_tta_feats is not None:
            with torch.no_grad():
                tta_logits = probe_head(use_tta_feats.to(device)).squeeze(-1)
                tta_probs = torch.sigmoid(tta_logits).cpu().numpy()
            tta_auc = roc_auc_score(tta_labels, tta_probs)
            tta_eer = compute_eer(tta_probs, tta_labels)
            print(f"\n  {name} (TTA {args.tta_clips} clips): AUC={tta_auc:.4f}  EER={tta_eer:.4f}")

            # TTA per-manipulation
            for manip in sorted(set(tta_manips)):
                mask = np.array([m == manip for m in tta_manips])
                if mask.sum() < 5 or manip == "original":
                    continue
                real_mask = np.array([m == "original" for m in tta_manips])
                combined_labels = np.concatenate([tta_labels[real_mask], tta_labels[mask]])
                combined_probs = np.concatenate([tta_probs[real_mask], tta_probs[mask]])
                if len(set(combined_labels)) < 2:
                    continue
                m_auc = roc_auc_score(combined_labels, combined_probs)
                print(f"    {manip:20s}: AUC={m_auc:.4f}  (TTA)")

        return auc, eer, manip_results

    val_auc, val_eer, val_manip = eval_split(val_feats, val_labels_arr, val_manips_arr, "VAL")

    if args.tta_clips > 1:
        test_auc, test_eer, test_manip = eval_split(
            test_feats, test_labels_arr, test_manips_arr, "TEST",
            test_tta_feats_avg, test_tta_labels_avg, test_tta_manips_avg)
    else:
        test_auc, test_eer, test_manip = eval_split(
            test_feats, test_labels_arr, test_manips_arr, "TEST")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"  Best val AUC: {best_val_auc:.4f}")
    print(f"  Test AUC:     {test_auc:.4f}  EER: {test_eer:.4f}")
    print(f"{'='*70}")

    # Save results
    import json
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "run_name": args.run_name,
        "val_auc": float(best_val_auc),
        "test_auc": float(test_auc),
        "test_eer": float(test_eer),
        "per_manip_test": {k: v for k, v in test_manip.items()},
        "args": vars(args),
    }
    with open(out_dir / f"{args.run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_dir / args.run_name}_results.json")

    if logger:
        try:
            logger.log({"final/val_auc": best_val_auc, "final/test_auc": test_auc,
                         "final/test_eer": test_eer}, step=args.epochs + 1)
            logger.finish()
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Linear probe on frozen backbone")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", default=None, help="PhysioNet checkpoint (loads backbone only)")
    p.add_argument("--backbone_weights", default=None, help="Local .pth for backbone init")
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--rppg_dim", type=int, default=12)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--run_name", default="linear_probe_v1")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--probe_batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tta_clips", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true", default=True)
    main(p.parse_args())
