"""
W3 Baseline: Simple frame-level deepfake classifier.

PURPOSE: Diagnose whether the data supports learning AT ALL.
If this gets AUC > 0.70 in 2-3 epochs, the data is fine and the problem
is in the PhysioNet architecture. If it doesn't learn either, the data
pipeline or labels are broken.

Architecture: EfficientNet-B4 (pretrained) → single linear → BCE
Input: 1 random frame per video (not a clip)
No temporal model, no rPPG, no blink, no multi-task loss.

Usage:
    python w3_train/train_baseline.py \
        --ff_root /kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23 \
        --out_dir /kaggle/working/checkpoints \
        --epochs 5 \
        --batch_size 32 \
        --num_workers 2 \
        --run_name baseline_v1
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
from w1_setup.trackio_init import ExperimentLogger


# ─── Dataset: single random frame per video ─────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
    "DeepFakeDetection": 1,
}


def build_video_list(ff_root: str) -> Tuple[List[str], List[int]]:
    """Scan FF++ and return (paths, labels)."""
    ff_root = Path(ff_root)
    paths, labels = [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        candidates = [ff_root / manip, ff_root / manip / "c23" / "videos"]
        for d in candidates:
            if d.exists():
                vids = sorted(d.glob("*.mp4"))
                if vids:
                    paths.extend([str(v) for v in vids])
                    labels.extend([label] * len(vids))
                    print(f"  {manip}: {len(vids)} videos ({d})")
                    break
    return paths, labels


class FrameDataset(Dataset):
    """Loads ONE random frame per video. Ultra-simple, fast."""

    def __init__(self, video_paths: List[str], labels: List[int], img_size: int = 224):
        self.video_paths = video_paths
        self.labels = labels
        self.img_size = img_size
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        frame = self._load_random_frame(path)
        if frame is None:
            frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        # Normalize: [0,1] → ImageNet stats
        frame = (frame - self.mean) / self.std
        # (H, W, 3) → (3, H, W)
        frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous()

        return {
            "frame": frame,
            "label": torch.tensor(float(label), dtype=torch.float32),
        }

    def _load_random_frame(self, path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        # Pick a frame from the middle 80% of the video (avoid black intro/outro)
        lo = int(total * 0.1)
        hi = max(lo + 1, int(total * 0.9))
        target = random.randint(lo, hi - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        return frame.astype(np.float32) / 255.0


# ─── Model: EfficientNet-B4 + 1 linear head ─────────────────────────────────

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

    logger = ExperimentLogger(
        project="p3_physio_deepfake",
        run_name=args.run_name or "baseline",
        config=vars(args),
        local_log_dir=args.log_dir,
    )

    # ─── Data ────────────────────────────────────────────────────────────
    paths, labels = build_video_list(args.ff_root)
    print(f"\nTotal: {len(paths)} videos, real={labels.count(0)}, fake={labels.count(1)}")

    # Shuffle + split
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths, labels = zip(*combined)
    paths, labels = list(paths), list(labels)

    n = len(paths)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_paths, train_labels = paths[:n_train], labels[:n_train]
    val_paths, val_labels = paths[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_paths, test_labels = paths[n_train+n_val:], labels[n_train+n_val:]

    train_ds = FrameDataset(train_paths, train_labels, args.img_size)
    val_ds = FrameDataset(val_paths, val_labels, args.img_size)
    test_ds = FrameDataset(test_paths, test_labels, args.img_size)

    # Balanced sampler
    train_labels_arr = np.array(train_labels)
    n_real = (train_labels_arr == 0).sum()
    n_fake = (train_labels_arr == 1).sum()
    weights = np.where(train_labels_arr == 0, 1.0 / (n_real + 1), 1.0 / (n_fake + 1))
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    real_ratio = n_real / max(n_fake, 1)
    fake_ratio = n_fake / max(n_real, 1)
    print(f"Train: {len(train_ds)} (real={n_real}, fake={n_fake}, ratio={fake_ratio:.1f})")
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ─── Model ───────────────────────────────────────────────────────────
    model = SimpleClassifier(args.backbone, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params")

    # ─── Optimizer: lower LR for backbone, higher for head ──────────────
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    # ─── Loss: simple BCE, no multi-task ────────────────────────────────
    # pos_weight compensates for balanced sampler seeing equal real/fake
    # but original data has more fakes — we want to slightly favor catching reals
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    # ─── Train ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        all_preds, all_targets = [], []

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            frames = batch["frame"].to(device)
            labels_b = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
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

            with torch.no_grad():
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_targets.extend(labels_b.cpu().numpy().tolist())

        scheduler.step()

        # Collapse check
        preds_arr = np.array(all_preds)
        pred_mean, pred_std = preds_arr.mean(), preds_arr.std()
        print(f"  [Train] pred_mean={pred_mean:.3f} pred_std={pred_std:.3f} "
              f"frac_fake={( preds_arr > 0.5).mean():.3f}")

        train_auc = roc_auc_score(all_targets, all_preds)

        # ─── Validation ──────────────────────────────────────────────────
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Val", leave=False):
                frames = batch["frame"].to(device)
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    logits = model(frames)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(batch["label"].numpy().tolist())

        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)
        val_auc = roc_auc_score(val_targets_arr, val_preds_arr)
        val_eer = compute_eer(val_preds_arr, val_targets_arr)

        # Val collapse check
        val_pred_std = val_preds_arr.std()
        val_frac_fake = (val_preds_arr > 0.5).mean()

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"train_auc={train_auc:.4f} | "
              f"val_auc={val_auc:.4f} val_eer={val_eer:.4f} | "
              f"val_pred_std={val_pred_std:.3f} val_frac_fake={val_frac_fake:.3f}")

        logger.log({
            "epoch": epoch,
            "train/loss": avg_loss,
            "train/auc": train_auc,
            "train/pred_mean": float(pred_mean),
            "train/pred_std": float(pred_std),
            "val/auc": val_auc,
            "val/eer": val_eer,
            "val/pred_std": float(val_pred_std),
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
            }, out_dir / "baseline_best.pt")
            print(f"  >> New best AUC={val_auc:.4f}")

    # ─── Final test ──────────────────────────────────────────────────────
    best_ckpt = torch.load(out_dir / "baseline_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Test", leave=False):
            frames = batch["frame"].to(device)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(frames)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            test_preds.extend(probs.tolist())
            test_targets.extend(batch["label"].numpy().tolist())

    test_auc = roc_auc_score(test_targets, test_preds)
    test_eer = compute_eer(np.array(test_preds), np.array(test_targets))

    print("\n" + "=" * 50)
    print(f"FINAL TEST: AUC={test_auc:.4f}  EER={test_eer:.4f}")
    print(f"Best val AUC={best_auc:.4f} (epoch {best_ckpt['epoch']})")
    print("=" * 50)

    logger.log_summary({"test/auc": test_auc, "test/eer": test_eer, "best_val_auc": best_auc})
    logger.finish()


def parse_args():
    p = argparse.ArgumentParser(description="P3 Baseline: frame-level deepfake classifier")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--run_name", default=None)

    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
