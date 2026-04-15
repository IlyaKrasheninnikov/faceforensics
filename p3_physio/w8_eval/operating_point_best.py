"""
Experiment F: Operating point analysis for the BEST model (mixed + rPPG).

Combines:
  - Mixed-dataset probe training (pooled FF++/CelebDF/DFDC)
  - rPPG features concatenated to backbone (the +2.3% CelebDF boost)
  - Full TPR@FPR deployment analysis

This gives the final forensic deployment profile for the strongest model.

Usage:
    python w8_eval/operating_point_best.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../crop \
        --dfdc_faces_root /kaggle/input/.../dfdc-faces \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --rppg_cache /kaggle/input/.../rppg_v2_300 \
        --blink_cache /kaggle/input/.../blink \
        --out_dir /kaggle/working/operating_point_best
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


# ─── Scanners ─────────────────────────────────────────────────────────────────

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
    def __init__(self, video_dirs, labels, clip_len=16, img_size=224,
                 rppg_cache=None, rppg_dim=12, blink_cache=None):
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.rppg_cache = Path(rppg_cache) if rppg_cache else None
        self.rppg_dim = rppg_dim
        self.blink_cache = Path(blink_cache) if blink_cache else None
        self.video_dirs_raw = video_dirs

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

        # rPPG — only for FF++ directory-based entries
        rppg_feat = np.zeros(self.rppg_dim, dtype=np.float32)
        vd_raw = self.video_dirs_raw[idx]
        if self.rppg_cache is not None and isinstance(vd_raw, str):
            vpath = Path(vd_raw)
            cache_feat = self.rppg_cache / vpath.parent.name / vpath.name / "rppg_v2_feat.npy"
            if cache_feat.exists():
                loaded = np.load(str(cache_feat)).astype(np.float32)
                if len(loaded) <= self.rppg_dim:
                    rppg_feat[:len(loaded)] = loaded

        # Blink — only for FF++
        blink_feat = np.zeros(16, dtype=np.float32)
        if self.blink_cache is not None and isinstance(vd_raw, str):
            vpath_b = Path(vd_raw)
            bp = self.blink_cache / vpath_b.parent.name / vpath_b.name / "blink_feat.npy"
            if bp.exists():
                loaded_b = np.load(str(bp)).astype(np.float32)
                if len(loaded_b) == 16:
                    blink_feat = loaded_b

        return {
            "frames": torch.from_numpy(clip).permute(0, 3, 1, 2).float(),
            "label": torch.tensor(label, dtype=torch.float32),
            "rppg_feat": torch.from_numpy(rppg_feat),
            "blink_feat": torch.from_numpy(blink_feat),
        }


@torch.no_grad()
def extract(model, dl, device, desc=""):
    all_bb, all_rppg, all_blink, all_l = [], [], [], []
    for batch in tqdm(dl, desc=desc, leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ff = model.frame_encoder(frames)
            pooled = ff.mean(dim=1)
        all_bb.append(pooled.float().cpu())
        all_rppg.append(batch["rppg_feat"])
        all_blink.append(batch["blink_feat"])
        all_l.extend(batch["label"].numpy().tolist())
    bb = torch.cat(all_bb, dim=0)
    bb.nan_to_num_(nan=0.0)
    rppg = torch.cat(all_rppg, dim=0)
    blink = torch.cat(all_blink, dim=0)
    return bb, rppg, blink, np.array(all_l)


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


def train_probe(feats, labels, device, epochs=20, lr=1e-3, bs=256):
    dim = feats.shape[1]
    probe = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    tfd = feats.to(device)
    tld = torch.tensor(labels, dtype=torch.float32, device=device)
    best_loss, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(tfd))
        el, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            loss = crit(probe(tfd[idx]).squeeze(-1), tld[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        if el/nb < best_loss:
            best_loss = el/nb
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe, best_loss


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
    dataset_feats = {}  # name → (bb, rppg, blink, labels)

    print("\nScanning FF++...")
    ff_dirs, ff_labels = scan_ff(args.ff_root)
    print(f"  FF++: {len(ff_dirs)} videos")
    ff_dl = DataLoader(
        ClipDataset(ff_dirs, ff_labels, args.clip_len, args.img_size,
                    args.rppg_cache, args.rppg_dim, args.blink_cache),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    ff_bb, ff_rppg, ff_blink, ff_la = extract(model, ff_dl, device, "FF++")
    dataset_feats["FF++"] = (ff_bb, ff_rppg, ff_blink, ff_la)

    rppg_hit = (ff_rppg.abs().sum(1) > 0).float().mean().item() * 100
    print(f"  FF++ rPPG non-zero: {rppg_hit:.0f}%")

    if args.celebdf_root:
        cd_dirs, cd_labels = scan_celebdf(args.celebdf_root)
        if cd_dirs:
            print(f"\n  CelebDF: {len(cd_dirs)} videos")
            cd_dl = DataLoader(
                ClipDataset(cd_dirs, cd_labels, args.clip_len, args.img_size),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            cd_bb, cd_rppg, cd_blink, cd_la = extract(model, cd_dl, device, "CelebDF")
            dataset_feats["CelebDF"] = (cd_bb, cd_rppg, cd_blink, cd_la)

    if args.dfdc_faces_root:
        df_dirs, df_labels = scan_dfdc_faces(args.dfdc_faces_root)
        if df_dirs:
            print(f"\n  DFDC: {len(df_dirs)} video groups")
            df_dl = DataLoader(
                ClipDataset(df_dirs, df_labels, args.clip_len, args.img_size),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            df_bb, df_rppg, df_blink, df_la = extract(model, df_dl, device, "DFDC")
            dataset_feats["DFDC"] = (df_bb, df_rppg, df_blink, df_la)

    # ─── Pool and split ──────────────────────────────────────────────────
    rng = random.Random(42)
    train_data = {"bb": [], "rppg": [], "blink": [], "labels": []}
    test_splits = {}

    for ds_name, (bb, rppg, blink, labels) in dataset_feats.items():
        n = len(labels)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * 0.8)
        tr_idx, te_idx = indices[:n_train], indices[n_train:]
        train_data["bb"].append(bb[tr_idx])
        train_data["rppg"].append(rppg[tr_idx])
        train_data["blink"].append(blink[tr_idx])
        train_data["labels"].append(labels[tr_idx])
        test_splits[ds_name] = (bb[te_idx], rppg[te_idx], blink[te_idx], labels[te_idx])
        print(f"  {ds_name}: train={n_train}, test={n-n_train}")

    all_bb = torch.cat(train_data["bb"], dim=0)
    all_rppg = torch.cat(train_data["rppg"], dim=0)
    all_blink = torch.cat(train_data["blink"], dim=0)
    all_labels = np.concatenate(train_data["labels"], axis=0)

    # ─── Train 4 model variants ──────────────────────────────────────────
    # We'll do operating point analysis for all 4 to find the true best
    variants = {
        "backbone_only": lambda bb, rppg, blink: bb,
        "backbone+rppg": lambda bb, rppg, blink: torch.cat([bb, rppg], dim=1),
        "backbone+blink": lambda bb, rppg, blink: torch.cat([bb, blink], dim=1),
        "full_fusion": lambda bb, rppg, blink: torch.cat([bb, rppg, blink], dim=1),
    }

    all_results = {}

    print(f"\n{'='*80}")
    print("EXPERIMENT F: Best Model Operating Points (Mixed + Bio-Signals)")
    print(f"{'='*80}")

    for vname, feat_fn in variants.items():
        train_feats = feat_fn(all_bb, all_rppg, all_blink)
        print(f"\n{'─'*70}")
        print(f"Variant: {vname} (dim={train_feats.shape[1]})")
        print(f"{'─'*70}")

        probe, best_loss = train_probe(train_feats, all_labels, device)
        print(f"  Train loss: {best_loss:.4f}")

        variant_results = {}

        for ds_name, (te_bb, te_rppg, te_blink, te_labels) in test_splits.items():
            if len(set(te_labels)) < 2:
                continue

            te_feats = feat_fn(te_bb, te_rppg, te_blink)
            probe.eval()
            with torch.no_grad():
                logits = probe(te_feats.to(device)).squeeze(-1).clamp(-20, 20)
                probs = torch.sigmoid(logits).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.5)

            auc = roc_auc_score(te_labels, probs)
            ap = average_precision_score(te_labels, probs)
            eer = compute_eer(probs, te_labels)
            tpr_fpr_res = tpr_at_fpr(te_labels, probs)
            opt = optimal_threshold(te_labels, probs)

            preds = (probs >= opt['optimal_threshold']).astype(int)
            tn, fp, fn, tp = confusion_matrix(te_labels, preds).ravel()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            print(f"\n  [{ds_name}] n={len(te_labels)}, real={int((te_labels==0).sum())}, fake={int((te_labels==1).sum())}")
            print(f"    AUC={auc:.4f}  AP={ap:.4f}  EER={eer:.4f}")
            for k, v in tpr_fpr_res.items():
                if k.startswith("TPR"):
                    print(f"    {k}: {v:.4f}")
            print(f"    Optimal: thresh={opt['optimal_threshold']:.3f}, "
                  f"TPR={opt['tpr_at_optimal']:.4f}, FPR={opt['fpr_at_optimal']:.4f}")
            print(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
            print(f"    Precision={precision:.4f}  Recall={recall:.4f}")

            variant_results[ds_name] = {
                "auc": auc, "ap": ap, "eer": eer,
                **tpr_fpr_res, **opt,
                "precision": precision, "recall": recall,
                "confusion_optimal": {"tp": int(tp), "fp": int(fp),
                                       "fn": int(fn), "tn": int(tn)},
            }

        all_results[vname] = variant_results

    # ─── Grand comparison table ──────────────────────────────────────────
    print(f"\n{'='*90}")
    print("GRAND COMPARISON: All Variants x All Datasets")
    print(f"{'='*90}")

    fpr_keys = ["TPR@FPR=1%", "TPR@FPR=5%", "TPR@FPR=10%", "TPR@FPR=20%"]

    for ds_name in test_splits.keys():
        print(f"\n  {ds_name}:")
        print(f"  {'Variant':<25s} {'AUC':>6s} {'EER':>6s}", end="")
        for k in fpr_keys:
            print(f"  {k:>12s}", end="")
        print()
        print(f"  {'-'*85}")

        best_auc = max(all_results[v].get(ds_name, {}).get("auc", 0) for v in all_results)

        for vname in variants:
            r = all_results[vname].get(ds_name, {})
            if not r:
                continue
            marker = " <-- BEST" if r["auc"] == best_auc else ""
            row = f"  {vname:<25s} {r['auc']:>6.4f} {r['eer']:>6.4f}"
            for k in fpr_keys:
                row += f"  {r.get(k, 0):>12.4f}"
            print(row + marker)

    print(f"\n{'='*90}")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "operating_point_best_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'operating_point_best_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment F: Best model operating points")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--rppg_dim", type=int, default=12)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--out_dir", default="./operating_point_best")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
