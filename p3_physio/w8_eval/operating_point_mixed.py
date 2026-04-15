"""
Experiment D: Operating point analysis for the MIXED-dataset probe.

Same TPR@FPR analysis as operating_point.py, but trains the probe on
pooled FF++/CelebDF/DFDC features instead of FF++ only.

This shows how the forensic deployment profile improves when using
diverse training data — particularly TPR@FPR=1% which is the key
metric for real-world forensic screening.

Usage:
    python w8_eval/operating_point_mixed.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../crop \
        --dfdc_faces_root /kaggle/input/.../dfdc-faces \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/operating_point_mixed
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


# ─── Scanners (same as train_mixed_probe.py) ─────────────────────────────────

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


def tpr_at_fpr(labels, scores, target_fprs=[0.01, 0.05, 0.10, 0.20]):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    results = {}
    for target in target_fprs:
        idx = np.searchsorted(fpr, target, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        results[f"TPR@FPR={target:.0%}"] = float(tpr[idx])
        results[f"thresh@FPR={target:.0%}"] = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return results


def optimal_threshold(labels, scores):
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

    # ─── Extract all datasets ────────────────────────────────────────────
    dataset_feats = {}

    print("\nScanning FF++...")
    ff_dirs, ff_labels = scan_ff(args.ff_root)
    print(f"  FF++: {len(ff_dirs)} videos")
    ff_dl = DataLoader(ClipDataset(ff_dirs, ff_labels, args.clip_len, args.img_size),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    ff_feats, ff_la = extract(model, ff_dl, device, "FF++")
    dataset_feats["FF++"] = (ff_feats, ff_la)

    if args.celebdf_root:
        cd_dirs, cd_labels = scan_celebdf(args.celebdf_root)
        if cd_dirs:
            print(f"\n  CelebDF: {len(cd_dirs)} videos")
            cd_dl = DataLoader(ClipDataset(cd_dirs, cd_labels, args.clip_len, args.img_size),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
            cd_feats, cd_la = extract(model, cd_dl, device, "CelebDF")
            dataset_feats["CelebDF"] = (cd_feats, cd_la)

    if args.dfdc_faces_root:
        df_dirs, df_labels = scan_dfdc_faces(args.dfdc_faces_root)
        if df_dirs:
            print(f"\n  DFDC: {len(df_dirs)} video groups")
            df_dl = DataLoader(ClipDataset(df_dirs, df_labels, args.clip_len, args.img_size),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
            df_feats, df_la = extract(model, df_dl, device, "DFDC")
            dataset_feats["DFDC"] = (df_feats, df_la)

    # ─── Pool: 80% train, 20% test per dataset ──────────────────────────
    rng = random.Random(42)
    train_feats_list, train_labels_list = [], []
    test_splits = {}

    for ds_name, (feats, labels) in dataset_feats.items():
        n = len(labels)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * 0.8)
        tr_idx, te_idx = indices[:n_train], indices[n_train:]
        train_feats_list.append(feats[tr_idx])
        train_labels_list.append(labels[tr_idx])
        test_splits[ds_name] = (feats[te_idx], labels[te_idx])
        print(f"  {ds_name}: train={n_train}, test={n-n_train}")

    all_train_feats = torch.cat(train_feats_list, dim=0)
    all_train_labels = np.concatenate(train_labels_list, axis=0)
    print(f"\n  Total train: {len(all_train_labels)}")

    # ─── Train mixed probe ───────────────────────────────────────────────
    print("\nTraining mixed-dataset probe...")
    dim = all_train_feats.shape[1]
    probe = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    tfd = all_train_feats.to(device)
    tld = torch.tensor(all_train_labels, dtype=torch.float32, device=device)

    best_loss, best_state = float("inf"), None
    for ep in range(1, 21):
        probe.train()
        perm = torch.randperm(len(tfd))
        epoch_loss, nb = 0.0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = crit(probe(tfd[idx]).squeeze(-1), tld[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); nb += 1
        avg = epoch_loss / nb
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    print(f"  Best train loss: {best_loss:.4f}")

    # ─── Also train FF-only probe for comparison ────────────────────────
    print("Training FF-only probe (baseline)...")
    ff_only_f, ff_only_l = dataset_feats["FF++"]
    n_ff = len(ff_only_l)
    ff_idx = list(range(n_ff))
    rng2 = random.Random(42)
    rng2.shuffle(ff_idx)
    n_ff_tr = int(n_ff * 0.8)
    ff_tr_f = ff_only_f[ff_idx[:n_ff_tr]]
    ff_tr_l = ff_only_l[ff_idx[:n_ff_tr]]

    probe_ff = nn.Linear(dim, 1).to(device)
    opt_ff = torch.optim.AdamW(probe_ff.parameters(), lr=1e-3, weight_decay=1e-3)
    ff_tfd = ff_tr_f.to(device)
    ff_tld = torch.tensor(ff_tr_l, dtype=torch.float32, device=device)
    best_ff_loss, best_ff_state = float("inf"), None
    for ep in range(1, 21):
        probe_ff.train()
        perm = torch.randperm(len(ff_tfd))
        el, nb = 0.0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = crit(probe_ff(ff_tfd[idx]).squeeze(-1), ff_tld[idx])
            opt_ff.zero_grad(); loss.backward(); opt_ff.step()
            el += loss.item(); nb += 1
        if el/nb < best_ff_loss:
            best_ff_loss = el/nb
            best_ff_state = {k: v.clone() for k, v in probe_ff.state_dict().items()}
    probe_ff.load_state_dict(best_ff_state)

    # ─── Operating point analysis for BOTH probes ────────────────────────
    all_results = {}

    print(f"\n{'='*80}")
    print("OPERATING POINT ANALYSIS: Mixed Probe vs FF-Only Probe")
    print(f"{'='*80}")

    for ds_name, (te_feats, te_labels) in test_splits.items():
        if len(set(te_labels)) < 2:
            continue

        print(f"\n--- {ds_name} (n={len(te_labels)}, "
              f"real={int((te_labels==0).sum())}, fake={int((te_labels==1).sum())}) ---")

        probe.eval(); probe_ff.eval()
        with torch.no_grad():
            mixed_probs = torch.sigmoid(
                probe(te_feats.to(device)).squeeze(-1).clamp(-20, 20)
            ).cpu().numpy()
            ff_probs = torch.sigmoid(
                probe_ff(te_feats.to(device)).squeeze(-1).clamp(-20, 20)
            ).cpu().numpy()
        mixed_probs = np.nan_to_num(mixed_probs, nan=0.5)
        ff_probs = np.nan_to_num(ff_probs, nan=0.5)

        ds_results = {}
        for probe_name, probs in [("ff_only", ff_probs), ("mixed", mixed_probs)]:
            auc = roc_auc_score(te_labels, probs)
            ap = average_precision_score(te_labels, probs)
            eer = compute_eer(probs, te_labels)
            tpr_fpr = tpr_at_fpr(te_labels, probs)
            opt = optimal_threshold(te_labels, probs)

            # Confusion matrix at optimal threshold
            preds = (probs >= opt['optimal_threshold']).astype(int)
            tn, fp, fn, tp = confusion_matrix(te_labels, preds).ravel()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            print(f"\n  [{probe_name.upper()}]")
            print(f"    AUC={auc:.4f}  AP={ap:.4f}  EER={eer:.4f}")
            for k, v in tpr_fpr.items():
                if k.startswith("TPR"):
                    print(f"    {k}: {v:.4f}")
            print(f"    Optimal: thresh={opt['optimal_threshold']:.3f}, "
                  f"TPR={opt['tpr_at_optimal']:.4f}, FPR={opt['fpr_at_optimal']:.4f}")
            print(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
            print(f"    Precision={precision:.4f}  Recall={recall:.4f}")

            ds_results[probe_name] = {
                "auc": auc, "ap": ap, "eer": eer,
                **tpr_fpr, **opt,
                "precision": precision, "recall": recall,
                "confusion_optimal": {"tp": int(tp), "fp": int(fp),
                                       "fn": int(fn), "tn": int(tn)},
            }

        all_results[ds_name] = ds_results

    # ─── Comparison table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TPR@FPR COMPARISON: Mixed vs FF-Only")
    print(f"{'='*80}")

    fpr_keys = ["TPR@FPR=1%", "TPR@FPR=5%", "TPR@FPR=10%", "TPR@FPR=20%"]

    print(f"\n{'Dataset':<12s} {'Probe':<8s} {'AUC':>6s}", end="")
    for k in fpr_keys:
        print(f"  {k:>12s}", end="")
    print()
    print("-" * 80)

    for ds_name, ds_results in all_results.items():
        for probe_name in ["ff_only", "mixed"]:
            r = ds_results[probe_name]
            row = f"{ds_name:<12s} {probe_name:<8s} {r['auc']:>6.3f}"
            for k in fpr_keys:
                row += f"  {r.get(k, 0):>12.4f}"
            print(row)
        # Delta row
        ff_r = ds_results["ff_only"]
        mx_r = ds_results["mixed"]
        row = f"{'':12s} {'delta':<8s} {mx_r['auc']-ff_r['auc']:>+6.3f}"
        for k in fpr_keys:
            delta = mx_r.get(k, 0) - ff_r.get(k, 0)
            row += f"  {delta:>+12.4f}"
        print(row)
        print()

    print("=" * 80)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "operating_point_mixed_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'operating_point_mixed_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment D: Operating points for mixed probe")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./operating_point_mixed")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
