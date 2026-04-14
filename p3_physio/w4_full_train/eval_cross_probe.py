"""
W4: Cross-dataset evaluation via linear probe.

Uses the same frozen backbone + linear probe approach as the ablation study.
Trains the probe on FF++ c23, then evaluates on CelebDF-v2 and/or DFDC.

This tests generalization: do the backbone features + bio-signals transfer?

Datasets supported:
  - CelebDF-v2: expects structure CelebDF-v2/{Celeb-real, Celeb-synthesis, YouTube-real}/frames/...
  - DFDC: expects structure dfdc/frames/{real, fake}/...

Usage:
    python w4_full_train/eval_cross_probe.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../CelebDF-v2 \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --rppg_cache /kaggle/input/.../rppg_v2_300 \
        --blink_cache /kaggle/input/.../blink \
        --out_dir /kaggle/working/cross_eval
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
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


# ─── Dataset scanning ───────────────────────────────────────────────────────

def scan_ff_folders(ff_root: str):
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


def scan_celebdf_folders(celebdf_root: str):
    """Scan CelebDF-v2 frame folders (diwakarsehgal/celebdfv2 Kaggle dataset).

    Expected structure:
        crop/{Train,Test}/{real,fake}/{video_id}/ *.png
    """
    root = Path(celebdf_root)
    video_dirs, labels = [], []

    for split in ["Test", "Train"]:
        for label_name, label in [("real", 0), ("fake", 1)]:
            ldir = root / split / label_name
            if not ldir.exists():
                # Also try: crop/Test/real etc.
                ldir = root / "crop" / split / label_name
            if not ldir.exists():
                continue
            subdirs = sorted(d for d in ldir.iterdir() if d.is_dir())
            count = 0
            for sd in subdirs:
                if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                    video_dirs.append(str(sd))
                    labels.append(label)
                    count += 1
            print(f"  CelebDF {split}/{label_name}: {count} video folders")

    return video_dirs, labels


def scan_dfdc_faces(dfdc_root: str):
    """Scan DFDC face images (itamargr/dfdc-faces-of-the-train-sample).

    Structure: {train,validation}/{real,fake}/ individual face images.
    Since images are individual faces (not grouped by video), we group them
    by video ID prefix (e.g., 'aapnvogymq' from 'aapnvogymq_0_0.png').

    Returns video_dirs (pseudo-dirs = grouped file lists), labels.
    """
    root = Path(dfdc_root)
    video_dirs, labels = [], []

    for split in ["validation", "train"]:
        for label_name, label in [("real", 0), ("fake", 1)]:
            ldir = root / split / label_name
            if not ldir.exists():
                continue
            # Group files by video ID prefix
            vid_to_files = {}
            for f in ldir.iterdir():
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    # filename like: aapnvogymq_0_0.png → video_id = aapnvogymq
                    vid_id = f.stem.rsplit('_', 2)[0] if f.stem.count('_') >= 2 else f.stem.split('_')[0]
                    vid_to_files.setdefault(vid_id, []).append(str(f))

            for vid_id, files in sorted(vid_to_files.items()):
                video_dirs.append(files)  # list of file paths instead of dir
                labels.append(label)
            print(f"  DFDC {split}/{label_name}: {len(vid_to_files)} video groups")

    return video_dirs, labels


def scan_dfdc_folders(dfdc_root: str):
    """Scan DFDC frame folders (generic structure).

    Tries multiple structures:
    1. diwakarsehgal format: {split}/{real,fake}/{video_id}/frames
    2. itamargr format: {split}/{real,fake}/individual_images
    3. Generic: {real,fake}/{video_id}/frames
    """
    root = Path(dfdc_root)
    video_dirs, labels = [], []

    # Try generic real/fake folder structure
    for label_name, label in [("real", 0), ("fake", 1)]:
        for candidate in [root / label_name, root / "frames" / label_name]:
            if not candidate.exists():
                continue
            for sd in sorted(d for d in candidate.iterdir() if d.is_dir()):
                if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                    video_dirs.append(str(sd))
                    labels.append(label)

    return video_dirs, labels


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """Handles two formats:
    - video_dirs[i] = str (directory path) → list frames from that dir
    - video_dirs[i] = list[str] (file paths) → use those files directly (DFDC faces)
    """
    def __init__(self, video_dirs, labels, clip_len=16, img_size=224):
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.frame_paths = []  # list of lists of absolute file paths
        for vd in video_dirs:
            if isinstance(vd, list):
                # Already a list of file paths (DFDC faces format)
                self.frame_paths.append(sorted(vd))
            else:
                # Directory path — list files
                frames = sorted(
                    os.path.join(vd, f)
                    for f in os.listdir(vd)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                )
                self.frame_paths.append(frames)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        all_frames = self.frame_paths[idx]
        n = len(all_frames)

        if n == 0:
            clip = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            max_start = max(0, n - self.clip_len)
            start = max_start // 2
            indices = [(start + i) % n for i in range(self.clip_len)]
            imgs = []
            for fi in indices:
                fpath = all_frames[fi]
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
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
    return float(ece)


# ─── Feature extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, device, desc="Extract"):
    all_feats, all_labels = [], []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            frame_feats = model.frame_encoder(frames)
            pooled = frame_feats.mean(dim=1)
        all_feats.append(pooled.float().cpu())
        all_labels.extend(batch["label"].numpy().tolist())
    feats = torch.cat(all_feats, dim=0)
    feats.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return feats, np.array(all_labels)


# ─── Linear probe ───────────────────────────────────────────────────────────

def train_probe(train_feats, train_labels, val_feats, val_labels,
                epochs=15, lr=1e-3, batch_size=256, device="cpu"):
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_feats_d = train_feats.to(device)
    train_labels_d = torch.tensor(train_labels, dtype=torch.float32, device=device)
    val_feats_d = val_feats.to(device)

    best_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(train_feats_d))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(train_feats_d[idx]).squeeze(-1)
            loss = criterion(logits, train_labels_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats_d).squeeze(-1)
            val_probs = torch.sigmoid(val_logits.clamp(-20, 20)).cpu().numpy()
            val_probs = np.nan_to_num(val_probs, nan=0.5)
            if len(set(val_labels)) >= 2:
                val_auc = roc_auc_score(val_labels, val_probs)
            else:
                val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state:
        probe.load_state_dict(best_state)
    return probe, best_auc


def evaluate_probe(probe, feats, labels, device="cpu"):
    probe.eval()
    with torch.no_grad():
        logits = probe(feats.to(device)).squeeze(-1)
        probs = torch.sigmoid(logits.clamp(-20, 20)).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)

    labels_arr = np.array(labels)
    if len(set(labels_arr)) < 2:
        return {"auc": 0.5, "eer": 0.5, "ap": 0.5, "ece": 0.5, "n": len(labels_arr)}

    return {
        "auc": roc_auc_score(labels_arr, probs),
        "eer": compute_eer(probs, labels_arr),
        "ap": average_precision_score(labels_arr, probs),
        "ece": compute_ece(probs, labels_arr),
        "n": len(labels_arr),
        "n_real": int((labels_arr == 0).sum()),
        "n_fake": int((labels_arr == 1).sum()),
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

    # ─── FF++ train/val split (for training the probe) ───────────────────────
    print("\nScanning FF++ dataset...")
    ff_dirs, ff_labels, ff_src_ids = scan_ff_folders(args.ff_root)
    print(f"  FF++ total: {len(ff_dirs)} videos")

    # Identity-based split
    id_to_indices = {}
    for i, sid in enumerate(ff_src_ids):
        id_to_indices.setdefault(sid, []).append(i)
    unique_ids = sorted(id_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_ids)
    n_train = int(len(unique_ids) * 0.8)
    n_val = int(len(unique_ids) * 0.1)
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])

    train_idx = [i for i, s in enumerate(ff_src_ids) if s in train_ids]
    val_idx = [i for i, s in enumerate(ff_src_ids) if s in val_ids]

    train_dirs = [ff_dirs[i] for i in train_idx]
    train_labels = [ff_labels[i] for i in train_idx]
    val_dirs = [ff_dirs[i] for i in val_idx]
    val_labels = [ff_labels[i] for i in val_idx]
    print(f"  Train: {len(train_dirs)} | Val: {len(val_dirs)}")

    # Extract FF++ features
    print("\nExtracting FF++ features...")
    t0 = time.time()
    train_ds = ClipDataset(train_dirs, train_labels, args.clip_len, args.img_size)
    val_ds = ClipDataset(val_dirs, val_labels, args.clip_len, args.img_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    train_feats, train_labels_arr = extract_features(model, train_dl, device, "FF++ Train")
    val_feats, val_labels_arr = extract_features(model, val_dl, device, "FF++ Val")
    print(f"  FF++ extraction: {time.time()-t0:.1f}s, feat dim: {train_feats.shape[1]}")

    # Train probe on FF++
    print("\nTraining linear probe on FF++...")
    probe, val_auc = train_probe(train_feats, train_labels_arr, val_feats, val_labels_arr,
                                 epochs=args.probe_epochs, lr=args.lr, device=device)
    print(f"  FF++ val AUC: {val_auc:.4f}")

    all_results = {"ff_val": {"auc": val_auc}}

    # ─── Cross-dataset evaluation ────────────────────────────────────────────
    cross_datasets = {}

    if args.celebdf_root:
        cd_dirs, cd_labels = scan_celebdf_folders(args.celebdf_root)
        if cd_dirs:
            cross_datasets["CelebDF-v2"] = (cd_dirs, cd_labels)

    if args.dfdc_root:
        dfdc_dirs, dfdc_labels = scan_dfdc_folders(args.dfdc_root)
        if dfdc_dirs:
            cross_datasets["DFDC"] = (dfdc_dirs, dfdc_labels)

    if args.dfdc_faces_root:
        dfdc_f_dirs, dfdc_f_labels = scan_dfdc_faces(args.dfdc_faces_root)
        if dfdc_f_dirs:
            cross_datasets["DFDC-faces"] = (dfdc_f_dirs, dfdc_f_labels)

    for ds_name, (ds_dirs, ds_labels) in cross_datasets.items():
        print(f"\n{'='*60}")
        print(f"Cross-dataset: {ds_name}")
        print(f"  Videos: {len(ds_dirs)} (real={ds_labels.count(0) if isinstance(ds_labels, list) else (np.array(ds_labels)==0).sum()}, "
              f"fake={ds_labels.count(1) if isinstance(ds_labels, list) else (np.array(ds_labels)==1).sum()})")

        ds = ClipDataset(ds_dirs, ds_labels, args.clip_len, args.img_size)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        print(f"  Extracting features...")
        t0 = time.time()
        feats, labels_arr = extract_features(model, dl, device, ds_name)
        print(f"  Extraction: {time.time()-t0:.1f}s")

        metrics = evaluate_probe(probe, feats, labels_arr, device)
        print(f"  AUC={metrics['auc']:.4f}  EER={metrics['eer']:.4f}  "
              f"AP={metrics['ap']:.4f}  ECE={metrics['ece']:.4f}")
        all_results[ds_name] = metrics

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20s} {'N':>6s} {'AUC':>8s} {'EER':>8s} {'AP':>8s} {'ECE':>8s}")
    print("-" * 60)
    for ds_name, m in all_results.items():
        if isinstance(m, dict) and "n" in m:
            print(f"{ds_name:<20s} {m['n']:>6d} {m['auc']:>8.4f} {m['eer']:>8.4f} "
                  f"{m['ap']:>8.4f} {m['ece']:>8.4f}")
        elif isinstance(m, dict) and "auc" in m:
            print(f"{ds_name:<20s} {'—':>6s} {m['auc']:>8.4f}")
    print("=" * 60)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cross_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'cross_eval_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W4: Cross-dataset eval via linear probe")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_root", default=None, help="DFDC with frame folders (real/fake/video_id/)")
    p.add_argument("--dfdc_faces_root", default=None, help="DFDC faces dataset (itamargr format: train/real/*.png)")
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--out_dir", default="./cross_eval")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--probe_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
