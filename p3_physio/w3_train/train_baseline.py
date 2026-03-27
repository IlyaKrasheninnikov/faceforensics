"""
W3 Baseline: Single-frame deepfake classifier on pre-extracted PNGs.

Loads from FaceForensics++_c23_processed which has structure:
    ff_root/
        original/000/0000.png, 0001.png, ...
        Deepfakes/000_003/0000.png, ...
        Face2Face/000_003/0000.png, ...
        FaceSwap/000_003/0000.png, ...
        NeuralTextures/000_003/0000.png, ...
        FaceShifter/000_003/0000.png, ...

Each "video folder" = one video. We pick 1 random PNG per video per epoch.
No cv2.VideoCapture overhead — pure image loading, 10-50x faster.

Usage (Kaggle):
    python w3_train/train_baseline.py \
        --ff_root /kaggle/input/ff-c23-processed/FaceForensics++_c23_processed \
        --out_dir /kaggle/working/checkpoints \
        --epochs 10 --batch_size 32 --num_workers 2 \
        --run_name baseline_png_v1
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("timm required: pip install timm")

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    raise ImportError("sklearn required: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from w1_setup.trackio_init import ExperimentLogger
    TRACKIO_AVAILABLE = True
except Exception:
    TRACKIO_AVAILABLE = False


# ─── Dataset: pre-extracted PNGs ─────────────────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
}


def scan_video_folders(ff_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan FF++ processed directory. Each subfolder under a manipulation type
    is one "video" containing extracted PNGs.

    Returns:
        video_dirs: list of paths to video folders
        labels: list of labels (0=real, 1=fake)
        src_ids: list of source identity IDs for identity-aware splitting
    """
    ff_root = Path(ff_root)
    video_dirs, labels, src_ids = [], [], []

    for manip, label in FF_MANIPULATION_TYPES.items():
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            print(f"  {manip}: NOT FOUND at {manip_dir}")
            continue

        # Each subfolder is one video
        subdirs = sorted([d for d in manip_dir.iterdir() if d.is_dir()])
        if not subdirs:
            # Maybe PNGs are directly in manip_dir (flat structure)?
            pngs = sorted(manip_dir.glob("*.png"))
            if pngs:
                video_dirs.append(str(manip_dir))
                labels.append(label)
                src_ids.append(manip)
                print(f"  {manip}: 1 folder ({len(pngs)} frames)")
            else:
                print(f"  {manip}: no subfolders or PNGs found")
            continue

        n_frames_total = 0
        for sd in subdirs:
            # Check it actually has PNGs
            has_frames = any(sd.glob("*.png")) or any(sd.glob("*.jpg"))
            if not has_frames:
                continue
            video_dirs.append(str(sd))
            labels.append(label)
            # Source ID: "000_003" → "000" (original person)
            src_ids.append(sd.name.split("_")[0])

        print(f"  {manip}: {len([d for d in subdirs])} video folders")

    return video_dirs, labels, src_ids


class PNGFrameDataset(Dataset):
    """
    Each sample = 1 random frame from a video folder.
    Loads PNG directly — no video decoding overhead.
    """

    def __init__(self, video_dirs: List[str], labels: List[int], img_size: int = 224):
        self.video_dirs = video_dirs
        self.labels = labels
        self.img_size = img_size
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Pre-scan frame lists for speed (avoid repeated os.listdir)
        self.frame_lists = []
        for vd in video_dirs:
            frames = sorted([
                f for f in os.listdir(vd)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.frame_lists.append(frames)

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        label = self.labels[idx]
        frames = self.frame_lists[idx]

        if not frames:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            # Pick a random frame from the middle 80% (skip intro/outro)
            n = len(frames)
            lo = int(n * 0.1)
            hi = max(lo + 1, int(n * 0.9))
            chosen = random.randint(lo, hi - 1)
            fpath = os.path.join(vdir, frames[chosen])

            img = cv2.imread(fpath)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0

        # ImageNet normalize
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return {
            "frame": img,
            "label": torch.tensor(float(label), dtype=torch.float32),
        }


# ─── Model ───────────────────────────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features  # 1792 for efficientnet_b4
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)  # (B,)


# ─── Training ────────────────────────────────────────────────────────────────

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    logger = None
    if TRACKIO_AVAILABLE:
        try:
            logger = ExperimentLogger(
                project="p3_physio_deepfake",
                run_name=args.run_name or "baseline_png",
                config=vars(args),
                local_log_dir=args.log_dir,
            )
        except Exception as e:
            print(f"[WARN] Trackio init failed: {e}")
    if logger is None:
        print("[INFO] Logging to stdout only")

    # ─── Data ────────────────────────────────────────────────────────────
    print("\nScanning dataset...")
    video_dirs, labels, src_ids = scan_video_folders(args.ff_root)
    n_total = len(video_dirs)
    n_real = labels.count(0)
    n_fake = labels.count(1)
    print(f"Total: {n_total} videos, real={n_real}, fake={n_fake}")

    if n_total == 0:
        print("ERROR: No data found! Check --ff_root path.")
        return

    # ── Identity-aware split ──────────────────────────────────────────
    id_to_indices: Dict[str, List[int]] = {}
    for i, sid in enumerate(src_ids):
        id_to_indices.setdefault(sid, []).append(i)

    unique_ids = sorted(id_to_indices.keys())
    random.shuffle(unique_ids)

    n_ids = len(unique_ids)
    n_train_ids = int(n_ids * 0.8)
    n_val_ids = int(n_ids * 0.1)

    train_ids = set(unique_ids[:n_train_ids])
    val_ids = set(unique_ids[n_train_ids:n_train_ids + n_val_ids])

    train_idx, val_idx, test_idx = [], [], []
    for sid, indices in id_to_indices.items():
        if sid in train_ids:
            train_idx.extend(indices)
        elif sid in val_ids:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    train_dirs = [video_dirs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_dirs = [video_dirs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_dirs = [video_dirs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Identity-aware split: {n_train_ids}/{n_val_ids}/{n_ids - n_train_ids - n_val_ids} "
          f"source IDs (no identity overlap)")

    train_ds = PNGFrameDataset(train_dirs, train_labels, args.img_size)
    val_ds = PNGFrameDataset(val_dirs, val_labels, args.img_size)
    test_ds = PNGFrameDataset(test_dirs, test_labels, args.img_size)

    # Balanced sampler
    train_labels_arr = np.array(train_labels)
    n_real_train = int((train_labels_arr == 0).sum())
    n_fake_train = int((train_labels_arr == 1).sum())
    weights = np.where(train_labels_arr == 0, 1.0 / (n_real_train + 1), 1.0 / (n_fake_train + 1))
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} videos (real={n_real_train}, fake={n_fake_train}), "
          f"{len(train_dl)} steps/epoch")
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ─── Model ───────────────────────────────────────────────────────────
    model = SimpleClassifier(args.backbone, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M params")

    # ─── Optimizer: lower LR for backbone, higher for head ──────────────
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    start_time = time.time()

    # ─── Train ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        all_preds, all_targets = [], []

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            frames = batch["frame"].to(device, non_blocking=True)
            labels_b = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames)
                loss = criterion(logits, labels_b)

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
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_targets.extend(labels_b.cpu().numpy().tolist())

        scheduler.step()

        # Train metrics
        preds_arr = np.array(all_preds)
        pred_mean, pred_std = preds_arr.mean(), preds_arr.std()
        train_auc = roc_auc_score(all_targets, all_preds)

        print(f"  [Train] pred_mean={pred_mean:.3f} pred_std={pred_std:.3f} "
              f"frac>0.5={( preds_arr > 0.5).mean():.3f}")

        # ─── Validation ──────────────────────────────────────────────────
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Val", leave=False):
                frames = batch["frame"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    logits = model(frames)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(batch["label"].numpy().tolist())

        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)
        val_auc = roc_auc_score(val_targets_arr, val_preds_arr)
        val_eer = compute_eer(val_preds_arr, val_targets_arr)
        val_pred_std = val_preds_arr.std()

        avg_loss = np.mean(losses)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"train_auc={train_auc:.4f} | "
              f"val_auc={val_auc:.4f} val_eer={val_eer:.4f} | "
              f"pred_std={val_pred_std:.3f} | "
              f"time={epoch_time:.0f}s total={total_time/60:.1f}min")

        if logger:
            try:
                logger.log({
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/auc": train_auc,
                    "train/pred_mean": float(pred_mean),
                    "train/pred_std": float(pred_std),
                    "val/auc": val_auc,
                    "val/eer": val_eer,
                    "val/pred_std": float(val_pred_std),
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_head": optimizer.param_groups[1]["lr"],
                }, step=epoch)
            except Exception:
                pass

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_eer": val_eer,
                "args": vars(args),
            }, out_dir / "baseline_best.pt")
            print(f"  >> New best AUC={val_auc:.4f}")

        # Safety: stop if approaching Kaggle 12h limit (leave 30min buffer)
        if total_time > 10.5 * 3600:
            print(f"  >> Time limit approaching ({total_time/3600:.1f}h), stopping early")
            break

    # ─── Final test ──────────────────────────────────────────────────────
    ckpt_path = out_dir / "baseline_best.pt"
    if ckpt_path.exists():
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"\nLoaded best checkpoint (epoch {best_ckpt['epoch']}, val_auc={best_ckpt['val_auc']:.4f})")
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Test", leave=False):
            frames = batch["frame"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            test_preds.extend(probs.tolist())
            test_targets.extend(batch["label"].numpy().tolist())

    test_auc = roc_auc_score(test_targets, test_preds)
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
    p = argparse.ArgumentParser(description="P3 Baseline: frame-level deepfake classifier (PNG)")
    p.add_argument("--ff_root", required=True,
                   help="Path to FaceForensics++_c23_processed with subfolders: original, Deepfakes, etc.")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--run_name", default=None)

    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
