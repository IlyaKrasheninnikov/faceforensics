"""
W8: Operating point analysis — TPR@FPR curves for forensic deployment.

For a forensic system, what matters is: "at 1% false alarm rate, how many
fakes do we catch?" This script computes TPR@FPR tables across all datasets.

Also computes:
  - Precision-Recall at various thresholds
  - Optimal threshold (Youden's J) for each dataset
  - Decision boundary analysis

Usage:
    python w8_eval/operating_point.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../crop \
        --dfdc_faces_root /kaggle/input/.../dfdc-faces \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/operating_point
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
    from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                                 average_precision_score, confusion_matrix)
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


# ─── Scanners (reused from w4) ──────────────────────────────────────────────

def scan_ff(ff_root):
    ff_root = Path(ff_root)
    dirs, labels, sids = [], [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        mdir = ff_root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                dirs.append(str(sd))
                labels.append(label)
                sids.append(sd.name.split("_")[0])
    return dirs, labels, sids


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
                frames = sorted(
                    os.path.join(vd, f) for f in os.listdir(vd)
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
                img = cv2.imread(all_frames[fi])
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                imgs.append(img)
            clip = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        return {"frames": tensor, "label": torch.tensor(label, dtype=torch.float32)}


@torch.no_grad()
def extract_features(model, dataloader, device, desc=""):
    all_f, all_l = [], []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ff = model.frame_encoder(frames)
            pooled = ff.mean(dim=1)
        all_f.append(pooled.float().cpu())
        all_l.extend(batch["label"].numpy().tolist())
    feats = torch.cat(all_f, dim=0)
    feats.nan_to_num_(nan=0.0)
    return feats, np.array(all_l)


def tpr_at_fpr(labels, scores, target_fprs=[0.01, 0.05, 0.10, 0.20]):
    """Compute TPR at specific FPR thresholds."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    results = {}
    for target in target_fprs:
        idx = np.searchsorted(fpr, target, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        results[f"TPR@FPR={target:.0%}"] = float(tpr[idx])
        results[f"thresh@FPR={target:.0%}"] = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return results


def optimal_threshold(labels, scores):
    """Youden's J statistic — optimal operating point."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    idx = np.argmax(j)
    return {
        "optimal_threshold": float(thresholds[idx]) if idx < len(thresholds) else 0.5,
        "tpr_at_optimal": float(tpr[idx]),
        "fpr_at_optimal": float(fpr[idx]),
        "youdens_j": float(j[idx]),
    }


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

    # FF++ train/val/test split
    print("\nScanning FF++...")
    ff_dirs, ff_labels, ff_sids = scan_ff(args.ff_root)
    id_map = {}
    for i, s in enumerate(ff_sids):
        id_map.setdefault(s, []).append(i)
    uids = sorted(id_map.keys())
    rng = random.Random(42)
    rng.shuffle(uids)
    n_tr = int(len(uids) * 0.8)
    n_va = int(len(uids) * 0.1)
    tr_ids = set(uids[:n_tr])
    va_ids = set(uids[n_tr:n_tr+n_va])
    te_ids = set(uids[n_tr+n_va:])

    def subset(id_set):
        idx = [i for i, s in enumerate(ff_sids) if s in id_set]
        return [ff_dirs[i] for i in idx], [ff_labels[i] for i in idx]

    tr_d, tr_l = subset(tr_ids)
    va_d, va_l = subset(va_ids)
    te_d, te_l = subset(te_ids)

    # Extract
    print("\nExtracting FF++ features...")
    t0 = time.time()
    tr_dl = DataLoader(ClipDataset(tr_d, tr_l, args.clip_len, args.img_size),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(ClipDataset(va_d, va_l, args.clip_len, args.img_size),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(ClipDataset(te_d, te_l, args.clip_len, args.img_size),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    tr_feats, tr_labels = extract_features(model, tr_dl, device, "FF++ train")
    va_feats, va_labels = extract_features(model, va_dl, device, "FF++ val")
    te_feats, te_labels = extract_features(model, te_dl, device, "FF++ test")
    print(f"  Extraction: {time.time()-t0:.1f}s")

    # Train probe
    print("\nTraining probe...")
    dim = tr_feats.shape[1]
    probe = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    tfd = tr_feats.to(device)
    tld = torch.tensor(tr_labels, dtype=torch.float32, device=device)
    vfd = va_feats.to(device)
    best_auc, best_st = 0.0, None
    for ep in range(1, 16):
        probe.train()
        perm = torch.randperm(len(tfd))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = crit(probe(tfd[idx]).squeeze(-1), tld[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            vp = torch.sigmoid(probe(vfd).squeeze(-1).clamp(-20,20)).cpu().numpy()
            va = roc_auc_score(va_labels, np.nan_to_num(vp, nan=0.5))
        if va > best_auc:
            best_auc = va
            best_st = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_st)
    print(f"  Val AUC: {best_auc:.4f}")

    # Evaluate all datasets
    datasets = {}

    # FF++ test
    probe.eval()
    with torch.no_grad():
        ff_probs = torch.sigmoid(probe(te_feats.to(device)).squeeze(-1).clamp(-20,20)).cpu().numpy()
    ff_probs = np.nan_to_num(ff_probs, nan=0.5)
    datasets["FF++ test"] = (te_labels, ff_probs)

    # CelebDF
    if args.celebdf_root:
        cd_dirs, cd_labels = scan_celebdf(args.celebdf_root)
        if cd_dirs:
            print(f"\nCelebDF: {len(cd_dirs)} videos")
            cd_dl = DataLoader(ClipDataset(cd_dirs, cd_labels, args.clip_len, args.img_size),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            cd_feats, cd_labels_arr = extract_features(model, cd_dl, device, "CelebDF")
            with torch.no_grad():
                cd_probs = torch.sigmoid(probe(cd_feats.to(device)).squeeze(-1).clamp(-20,20)).cpu().numpy()
            cd_probs = np.nan_to_num(cd_probs, nan=0.5)
            datasets["CelebDF-v2"] = (cd_labels_arr, cd_probs)

    # DFDC
    if args.dfdc_faces_root:
        df_dirs, df_labels = scan_dfdc_faces(args.dfdc_faces_root)
        if df_dirs:
            print(f"\nDFDC: {len(df_dirs)} video groups")
            df_dl = DataLoader(ClipDataset(df_dirs, df_labels, args.clip_len, args.img_size),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            df_feats, df_labels_arr = extract_features(model, df_dl, device, "DFDC")
            with torch.no_grad():
                df_probs = torch.sigmoid(probe(df_feats.to(device)).squeeze(-1).clamp(-20,20)).cpu().numpy()
            df_probs = np.nan_to_num(df_probs, nan=0.5)
            datasets["DFDC"] = (df_labels_arr, df_probs)

    # ─── Analysis ────────────────────────────────────────────────────────────
    all_results = {}
    print(f"\n{'='*70}")
    print("OPERATING POINT ANALYSIS")
    print(f"{'='*70}")

    for ds_name, (labels, probs) in datasets.items():
        print(f"\n--- {ds_name} (n={len(labels)}, real={int((labels==0).sum())}, fake={int((labels==1).sum())}) ---")

        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
        tpr_fpr = tpr_at_fpr(labels, probs)
        opt = optimal_threshold(labels, probs)

        print(f"  AUC={auc:.4f}  AP={ap:.4f}")
        print(f"  Operating points:")
        for k, v in tpr_fpr.items():
            if k.startswith("TPR"):
                print(f"    {k}: {v:.4f}")
        print(f"  Optimal (Youden): threshold={opt['optimal_threshold']:.3f}, "
              f"TPR={opt['tpr_at_optimal']:.4f}, FPR={opt['fpr_at_optimal']:.4f}")

        # Confusion matrix at optimal threshold
        preds = (probs >= opt['optimal_threshold']).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        print(f"  Confusion @ optimal: TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Precision={tp/(tp+fp+1e-8):.4f}  Recall={tp/(tp+fn+1e-8):.4f}")

        all_results[ds_name] = {
            "auc": auc, "ap": ap,
            **tpr_fpr, **opt,
            "confusion_optimal": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        }

    # ─── Cross-dataset TPR@FPR comparison table ─────────────────────────────
    print(f"\n{'='*70}")
    print("TPR @ FPR COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Dataset':<15s} {'AUC':>6s}"
    for fpr in ["1%", "5%", "10%", "20%"]:
        header += f"  {'TPR@'+fpr:>10s}"
    print(header)
    print("-" * 65)
    for ds_name, r in all_results.items():
        row = f"{ds_name:<15s} {r['auc']:>6.3f}"
        for fpr_key in ["TPR@FPR=1%", "TPR@FPR=5%", "TPR@FPR=10%", "TPR@FPR=20%"]:
            row += f"  {r.get(fpr_key, 0):>10.3f}"
        print(row)
    print("=" * 65)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "operating_point_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'operating_point_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W8: Operating point analysis")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./operating_point")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
