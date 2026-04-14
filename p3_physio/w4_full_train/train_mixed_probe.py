"""
W4+: Mixed-dataset probe training.

Instead of training the linear probe on FF++ only and testing on CelebDF/DFDC,
this script pools features from ALL datasets and trains a single probe.

The key hypothesis: if we expose the probe to diverse manipulation types at
training time, cross-dataset generalization improves significantly.

Protocol:
  1. Extract backbone features from FF++, CelebDF, DFDC (same frozen backbone)
  2. Pool all features, split 80/20 train/test (stratified by dataset)
  3. Train one linear probe on the mixed training set
  4. Report per-dataset AUC on the held-out test portions

Usage:
    python w4_full_train/train_mixed_probe.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../crop \
        --dfdc_faces_root /kaggle/input/.../dfdc-faces \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/mixed_probe
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
except ImportError:
    raise ImportError("sklearn required")

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FF_MANIPULATION_TYPES = {
    "original": 0, "Deepfakes": 1, "Face2Face": 1,
    "FaceSwap": 1, "NeuralTextures": 1, "FaceShifter": 1,
}


def scan_ff(ff_root):
    ff_root = Path(ff_root)
    dirs, labels = [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        mdir = ff_root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                dirs.append(str(sd))
                labels.append(label)
    return dirs, labels


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
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    vid_id = f.stem.rsplit('_', 2)[0] if f.stem.count('_') >= 2 else f.stem.split('_')[0]
                    vid_to_files.setdefault(vid_id, []).append(str(f))
            for vid_id, files in sorted(vid_to_files.items()):
                dirs.append(files)
                labels.append(label)
    return dirs, labels


class ClipDataset(Dataset):
    def __init__(self, video_dirs, labels, clip_len=16, img_size=224):
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.frame_paths = []
        for vd in video_dirs:
            if isinstance(vd, list):
                self.frame_paths.append(sorted(vd))
            else:
                self.frame_paths.append(sorted(
                    os.path.join(vd, f) for f in os.listdir(vd)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                ))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        frames = self.frame_paths[idx]
        n = len(frames)
        if n == 0:
            clip = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            start = max(0, n - self.clip_len) // 2
            indices = [(start + i) % n for i in range(self.clip_len)]
            imgs = []
            for fi in indices:
                img = cv2.imread(frames[fi])
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                imgs.append(img)
            clip = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "frames": torch.from_numpy(clip).permute(0, 3, 1, 2).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


@torch.no_grad()
def extract(model, dl, device, desc=""):
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


def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


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
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        bb = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(bb, strict=False)
        print(f"Loaded {len(bb)} backbone tensors")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # ─── Collect all datasets ────────────────────────────────────────────────
    dataset_feats = {}  # name → (feats, labels)

    # FF++
    print("\nScanning FF++...")
    ff_dirs, ff_labels = scan_ff(args.ff_root)
    print(f"  FF++: {len(ff_dirs)} videos")
    ff_dl = DataLoader(ClipDataset(ff_dirs, ff_labels, args.clip_len, args.img_size),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ff_feats, ff_labels_arr = extract(model, ff_dl, device, "FF++")
    dataset_feats["FF++"] = (ff_feats, ff_labels_arr)

    # CelebDF
    if args.celebdf_root:
        cd_dirs, cd_labels = scan_celebdf(args.celebdf_root)
        if cd_dirs:
            print(f"\n  CelebDF: {len(cd_dirs)} videos")
            cd_dl = DataLoader(ClipDataset(cd_dirs, cd_labels, args.clip_len, args.img_size),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            cd_feats, cd_labels_arr = extract(model, cd_dl, device, "CelebDF")
            dataset_feats["CelebDF"] = (cd_feats, cd_labels_arr)

    # DFDC
    if args.dfdc_faces_root:
        df_dirs, df_labels = scan_dfdc_faces(args.dfdc_faces_root)
        if df_dirs:
            print(f"\n  DFDC: {len(df_dirs)} video groups")
            df_dl = DataLoader(ClipDataset(df_dirs, df_labels, args.clip_len, args.img_size),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            df_feats, df_labels_arr = extract(model, df_dl, device, "DFDC")
            dataset_feats["DFDC"] = (df_feats, df_labels_arr)

    # ─── Pool and split ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MIXED-DATASET PROBE TRAINING")
    print(f"{'='*60}")

    # For each dataset: 80% train, 20% test
    rng = random.Random(42)
    train_feats_list, train_labels_list = [], []
    test_splits = {}  # name → (feats, labels)

    for ds_name, (feats, labels) in dataset_feats.items():
        n = len(labels)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * 0.8)
        tr_idx = indices[:n_train]
        te_idx = indices[n_train:]

        train_feats_list.append(feats[tr_idx])
        train_labels_list.append(labels[tr_idx])
        test_splits[ds_name] = (feats[te_idx], labels[te_idx])

        n_real_tr = int((labels[tr_idx] == 0).sum())
        n_real_te = int((labels[te_idx] == 0).sum())
        print(f"  {ds_name}: train={n_train} (real={n_real_tr}), test={n-n_train} (real={n_real_te})")

    all_train_feats = torch.cat(train_feats_list, dim=0)
    all_train_labels = np.concatenate(train_labels_list, axis=0)
    print(f"\n  Total train: {len(all_train_labels)} "
          f"(real={int((all_train_labels==0).sum())}, fake={int((all_train_labels==1).sum())})")

    # ─── Train mixed probe ───────────────────────────────────────────────────
    print("\nTraining mixed-dataset probe...")
    dim = all_train_feats.shape[1]
    probe = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    tfd = all_train_feats.to(device)
    tld = torch.tensor(all_train_labels, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    for ep in range(1, 21):
        probe.train()
        perm = torch.randperm(len(tfd))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = crit(probe(tfd[idx]).squeeze(-1), tld[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    print(f"  Best train loss: {best_loss:.4f}")

    # ─── Evaluate per dataset ────────────────────────────────────────────────
    # Also train FF++-only probe for comparison
    print("\nTraining FF++-only probe (baseline)...")
    ff_only_feats = dataset_feats["FF++"][0]
    ff_only_labels = dataset_feats["FF++"][1]
    n_ff = len(ff_only_labels)
    ff_idx = list(range(n_ff))
    rng2 = random.Random(42)
    rng2.shuffle(ff_idx)
    n_ff_tr = int(n_ff * 0.8)
    ff_tr_feats = ff_only_feats[ff_idx[:n_ff_tr]]
    ff_tr_labels = ff_only_labels[ff_idx[:n_ff_tr]]

    probe_ff = nn.Linear(dim, 1).to(device)
    opt_ff = torch.optim.AdamW(probe_ff.parameters(), lr=1e-3, weight_decay=1e-3)
    ff_tfd = ff_tr_feats.to(device)
    ff_tld = torch.tensor(ff_tr_labels, dtype=torch.float32, device=device)
    best_ff_loss, best_ff_state = float("inf"), None
    for ep in range(1, 21):
        probe_ff.train()
        perm = torch.randperm(len(ff_tfd))
        el = 0.0; nb = 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = crit(probe_ff(ff_tfd[idx]).squeeze(-1), ff_tld[idx])
            opt_ff.zero_grad(); loss.backward(); opt_ff.step()
            el += loss.item(); nb += 1
        if el/nb < best_ff_loss:
            best_ff_loss = el/nb
            best_ff_state = {k: v.clone() for k, v in probe_ff.state_dict().items()}
    probe_ff.load_state_dict(best_ff_state)

    # Evaluate both probes on all test splits
    all_results = {}
    print(f"\n{'='*60}")
    print(f"{'Dataset':<15s} {'FF-probe AUC':>13s} {'Mixed AUC':>11s} {'Delta':>8s} {'FF EER':>8s} {'Mix EER':>8s}")
    print("-" * 60)

    for ds_name, (te_feats, te_labels) in test_splits.items():
        if len(set(te_labels)) < 2:
            continue
        probe.eval(); probe_ff.eval()
        with torch.no_grad():
            mixed_probs = torch.sigmoid(probe(te_feats.to(device)).squeeze(-1).clamp(-20,20)).cpu().numpy()
            ff_probs = torch.sigmoid(probe_ff(te_feats.to(device)).squeeze(-1).clamp(-20,20)).cpu().numpy()
        mixed_probs = np.nan_to_num(mixed_probs, nan=0.5)
        ff_probs = np.nan_to_num(ff_probs, nan=0.5)

        mixed_auc = roc_auc_score(te_labels, mixed_probs)
        ff_auc = roc_auc_score(te_labels, ff_probs)
        mixed_eer = compute_eer(mixed_probs, te_labels)
        ff_eer = compute_eer(ff_probs, te_labels)
        delta = mixed_auc - ff_auc

        print(f"{ds_name:<15s} {ff_auc:>13.4f} {mixed_auc:>11.4f} {delta:>+8.4f} {ff_eer:>8.4f} {mixed_eer:>8.4f}")

        all_results[ds_name] = {
            "ff_only_auc": ff_auc, "mixed_auc": mixed_auc,
            "delta_auc": delta,
            "ff_only_eer": ff_eer, "mixed_eer": mixed_eer,
            "n_test": len(te_labels),
        }

    print("=" * 60)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mixed_probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'mixed_probe_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W4+: Mixed-dataset probe training")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./mixed_probe")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
