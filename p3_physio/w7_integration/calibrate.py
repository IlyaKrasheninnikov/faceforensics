"""
W7: Platt scaling calibration for the linear probe.

Trains a logistic calibrator on the validation set so that output probabilities
match empirical accuracy (ECE → 0). Uses the same frozen backbone + probe setup.

Usage:
    python w7_integration/calibrate.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/calibration
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
except ImportError:
    raise ImportError("sklearn required: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FF_MANIPULATION_TYPES = {
    "original": 0, "Deepfakes": 1, "Face2Face": 1,
    "FaceSwap": 1, "NeuralTextures": 1, "FaceShifter": 1,
}


# ─── Dataset (reused) ───────────────────────────────────────────────────────

def scan_video_folders(ff_root):
    ff_root = Path(ff_root)
    video_dirs, labels, src_ids = [], [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            continue
        for sd in sorted(d for d in manip_dir.iterdir() if d.is_dir()):
            if not any(sd.glob("*.png")) and not any(sd.glob("*.jpg")):
                continue
            video_dirs.append(str(sd))
            labels.append(label)
            src_ids.append(sd.name.split("_")[0])
    return video_dirs, labels, src_ids


class ClipDataset(Dataset):
    def __init__(self, video_dirs, labels, clip_len=16, img_size=224):
        self.video_dirs = video_dirs
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.frame_lists = []
        for vd in video_dirs:
            frames = sorted(f for f in os.listdir(vd) if f.endswith(('.png', '.jpg', '.jpeg')))
            self.frame_lists.append(frames)

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        label = self.labels[idx]
        all_frames = self.frame_lists[idx]
        n = len(all_frames)
        if n == 0:
            clip = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            max_start = max(0, n - self.clip_len)
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
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip_tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        return {"frames": clip_tensor, "label": torch.tensor(label, dtype=torch.float32)}


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(probs) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def full_metrics(probs, labels):
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "eer": compute_eer(probs, labels),
        "ap": float(average_precision_score(labels, probs)),
        "ece": compute_ece(probs, labels),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build backbone
    cfg = ModelConfig(
        backbone="efficientnet_b4", backbone_pretrained=False,
        temporal_model="mean", temporal_dim=0, clip_len=args.clip_len,
        img_size=args.img_size, dropout=0.0,
        use_physio_fusion=False, use_pulse_head=False,
        use_blink_head=False, use_motion_model=False,
    )
    model = PhysioNet(cfg).to(device)

    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        print(f"Loading backbone from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded {len(backbone_state)} backbone tensors")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Scan & split
    print("\nScanning dataset...")
    video_dirs, labels, src_ids = scan_video_folders(args.ff_root)

    id_to_indices = {}
    for i, sid in enumerate(src_ids):
        id_to_indices.setdefault(sid, []).append(i)
    unique_ids = sorted(id_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_ids)
    n_train = int(len(unique_ids) * 0.8)
    n_val = int(len(unique_ids) * 0.1)
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    def get_subset(id_set):
        idx = [i for i, s in enumerate(src_ids) if s in id_set]
        return [video_dirs[i] for i in idx], [labels[i] for i in idx]

    train_dirs, train_labels = get_subset(train_ids)
    val_dirs, val_labels = get_subset(val_ids)
    test_dirs, test_labels = get_subset(test_ids)
    print(f"Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)}")

    # Extract features
    @torch.no_grad()
    def extract(dirs, lbls, desc=""):
        ds = ClipDataset(dirs, lbls, args.clip_len, args.img_size)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        all_f, all_l = [], []
        for batch in tqdm(dl, desc=desc, leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ff = model.frame_encoder(frames)
                pooled = ff.mean(dim=1)
            all_f.append(pooled.float().cpu())
            all_l.extend(batch["label"].numpy().tolist())
        feats = torch.cat(all_f, dim=0)
        feats.nan_to_num_(nan=0.0)
        return feats, np.array(all_l)

    print("\nExtracting features...")
    t0 = time.time()
    train_feats, train_labels_arr = extract(train_dirs, train_labels, "Train")
    val_feats, val_labels_arr = extract(val_dirs, val_labels, "Val")
    test_feats, test_labels_arr = extract(test_dirs, test_labels, "Test")
    print(f"Extraction: {time.time()-t0:.1f}s")

    # Train linear probe
    print("\nTraining linear probe...")
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_f_d = train_feats.to(device)
    train_l_d = torch.tensor(train_labels_arr, dtype=torch.float32, device=device)
    val_f_d = val_feats.to(device)

    best_auc, best_state = 0.0, None
    for epoch in range(1, 16):
        probe.train()
        perm = torch.randperm(len(train_f_d))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            logits = probe(train_f_d[idx]).squeeze(-1)
            loss = criterion(logits, train_l_d[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        probe.eval()
        with torch.no_grad():
            vp = torch.sigmoid(probe(val_f_d).squeeze(-1).clamp(-20, 20)).cpu().numpy()
            vp = np.nan_to_num(vp, nan=0.5)
            va = roc_auc_score(val_labels_arr, vp)
        if va > best_auc:
            best_auc = va
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    print(f"  Probe val AUC: {best_auc:.4f}")

    # Get raw logits for val and test
    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_f_d).squeeze(-1).cpu().numpy()
        test_logits = probe(test_feats.to(device)).squeeze(-1).cpu().numpy()

    # Uncalibrated probabilities
    val_probs_raw = 1 / (1 + np.exp(-np.clip(val_logits, -20, 20)))
    test_probs_raw = 1 / (1 + np.exp(-np.clip(test_logits, -20, 20)))

    print("\n--- BEFORE calibration ---")
    val_metrics_raw = full_metrics(val_probs_raw, val_labels_arr)
    test_metrics_raw = full_metrics(test_probs_raw, test_labels_arr)
    print(f"  Val:  AUC={val_metrics_raw['auc']:.4f}  EER={val_metrics_raw['eer']:.4f}  "
          f"ECE={val_metrics_raw['ece']:.4f}")
    print(f"  Test: AUC={test_metrics_raw['auc']:.4f}  EER={test_metrics_raw['eer']:.4f}  "
          f"ECE={test_metrics_raw['ece']:.4f}")

    # ─── Platt scaling ───────────────────────────────────────────────────────
    # Fit logistic regression on val logits → val labels
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(val_logits.reshape(-1, 1), val_labels_arr)

    val_probs_cal = platt.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    test_probs_cal = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]

    print("\n--- AFTER Platt scaling ---")
    val_metrics_cal = full_metrics(val_probs_cal, val_labels_arr)
    test_metrics_cal = full_metrics(test_probs_cal, test_labels_arr)
    print(f"  Val:  AUC={val_metrics_cal['auc']:.4f}  EER={val_metrics_cal['eer']:.4f}  "
          f"ECE={val_metrics_cal['ece']:.4f}")
    print(f"  Test: AUC={test_metrics_cal['auc']:.4f}  EER={test_metrics_cal['eer']:.4f}  "
          f"ECE={test_metrics_cal['ece']:.4f}")

    # ─── Reliability diagram data ────────────────────────────────────────────
    n_bins = 10
    for name, probs, lbls in [("val_raw", val_probs_raw, val_labels_arr),
                               ("val_cal", val_probs_cal, val_labels_arr),
                               ("test_raw", test_probs_raw, test_labels_arr),
                               ("test_cal", test_probs_cal, test_labels_arr)]:
        fraction_pos, mean_pred = calibration_curve(lbls, probs, n_bins=n_bins, strategy="uniform")
        print(f"\n  Reliability ({name}):")
        for mp, fp in zip(mean_pred, fraction_pos):
            bar = "#" * int(fp * 40)
            print(f"    pred={mp:.2f} → actual={fp:.2f}  {bar}")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<12s} {'Val raw':>10s} {'Val cal':>10s} {'Test raw':>10s} {'Test cal':>10s}")
    print("-" * 55)
    for metric in ["auc", "eer", "ece", "ap"]:
        print(f"{metric:<12s} {val_metrics_raw[metric]:>10.4f} {val_metrics_cal[metric]:>10.4f} "
              f"{test_metrics_raw[metric]:>10.4f} {test_metrics_cal[metric]:>10.4f}")
    print("=" * 55)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "val_raw": val_metrics_raw, "val_calibrated": val_metrics_cal,
        "test_raw": test_metrics_raw, "test_calibrated": test_metrics_cal,
        "platt_coef": float(platt.coef_[0][0]),
        "platt_intercept": float(platt.intercept_[0]),
    }
    with open(out_dir / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_dir / 'calibration_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W7: Platt scaling calibration")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./calibration")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
