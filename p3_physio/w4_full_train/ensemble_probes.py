"""
Experiment E: Ensemble of FF-only + Mixed probe.

Simple logit averaging: train both probes independently, average their
raw logits at inference time. Diverse training strategies often produce
complementary errors — the ensemble may beat either probe alone.

Reports AUC, EER, TPR@FPR for each dataset, plus delta vs each individual probe.

Usage:
    python w4_full_train/ensemble_probes.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../crop \
        --dfdc_faces_root /kaggle/input/.../dfdc-faces \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/ensemble
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
    from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                                 confusion_matrix)
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
    return results


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
    return probe


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

    # ─── Split: 80% train, 20% test per dataset ─────────────────────────
    rng = random.Random(42)
    mixed_train_f, mixed_train_l = [], []
    ff_train_f, ff_train_l = None, None
    test_splits = {}

    for ds_name, (feats, labels) in dataset_feats.items():
        n = len(labels)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * 0.8)
        tr_idx, te_idx = indices[:n_train], indices[n_train:]

        mixed_train_f.append(feats[tr_idx])
        mixed_train_l.append(labels[tr_idx])
        test_splits[ds_name] = (feats[te_idx], labels[te_idx])

        if ds_name == "FF++":
            ff_train_f = feats[tr_idx]
            ff_train_l = labels[tr_idx]

        print(f"  {ds_name}: train={n_train}, test={n-n_train}")

    all_mixed_f = torch.cat(mixed_train_f, dim=0)
    all_mixed_l = np.concatenate(mixed_train_l, axis=0)
    print(f"\n  Mixed train total: {len(all_mixed_l)}")

    # ─── Train both probes ───────────────────────────────────────────────
    print("\nTraining FF-only probe...")
    probe_ff = train_probe(ff_train_f, ff_train_l, device)
    print("Training mixed probe...")
    probe_mixed = train_probe(all_mixed_f, all_mixed_l, device)
    print("Done.\n")

    # ─── Evaluate: individual + ensemble ─────────────────────────────────
    all_results = {}

    print(f"{'='*80}")
    print("EXPERIMENT E: Ensemble (FF-only + Mixed Probe)")
    print(f"{'='*80}")

    for ds_name, (te_feats, te_labels) in test_splits.items():
        if len(set(te_labels)) < 2:
            continue

        print(f"\n--- {ds_name} (n={len(te_labels)}) ---")

        probe_ff.eval(); probe_mixed.eval()
        with torch.no_grad():
            ff_logits = probe_ff(te_feats.to(device)).squeeze(-1).clamp(-20, 20)
            mx_logits = probe_mixed(te_feats.to(device)).squeeze(-1).clamp(-20, 20)

            ff_probs = torch.sigmoid(ff_logits).cpu().numpy()
            mx_probs = torch.sigmoid(mx_logits).cpu().numpy()

            # Ensemble: average logits, then sigmoid
            ens_logits = (ff_logits + mx_logits) / 2.0
            ens_probs = torch.sigmoid(ens_logits).cpu().numpy()

        ff_probs = np.nan_to_num(ff_probs, nan=0.5)
        mx_probs = np.nan_to_num(mx_probs, nan=0.5)
        ens_probs = np.nan_to_num(ens_probs, nan=0.5)

        ds_results = {}
        for name, probs in [("ff_only", ff_probs), ("mixed", mx_probs), ("ensemble", ens_probs)]:
            auc = roc_auc_score(te_labels, probs)
            eer = compute_eer(probs, te_labels)
            ap = average_precision_score(te_labels, probs)
            tpr_fpr = tpr_at_fpr(te_labels, probs)

            print(f"  {name:>10s}: AUC={auc:.4f}  EER={eer:.4f}  AP={ap:.4f}  "
                  f"TPR@1%={tpr_fpr['TPR@FPR=1%']:.4f}  TPR@5%={tpr_fpr['TPR@FPR=5%']:.4f}")

            ds_results[name] = {
                "auc": auc, "eer": eer, "ap": ap,
                **tpr_fpr,
            }

        # Deltas
        print(f"\n  Ensemble vs FF-only:  AUC {ds_results['ensemble']['auc']-ds_results['ff_only']['auc']:+.4f}")
        print(f"  Ensemble vs Mixed:   AUC {ds_results['ensemble']['auc']-ds_results['mixed']['auc']:+.4f}")

        all_results[ds_name] = ds_results

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<12s} {'Probe':<10s} {'AUC':>6s} {'EER':>6s} {'TPR@1%':>8s} {'TPR@5%':>8s} {'TPR@10%':>9s}")
    print("-" * 70)

    for ds_name, ds_r in all_results.items():
        for pname in ["ff_only", "mixed", "ensemble"]:
            r = ds_r[pname]
            marker = " ***" if pname == "ensemble" else ""
            print(f"{ds_name:<12s} {pname:<10s} {r['auc']:>6.3f} {r['eer']:>6.3f} "
                  f"{r.get('TPR@FPR=1%', 0):>8.4f} {r.get('TPR@FPR=5%', 0):>8.4f} "
                  f"{r.get('TPR@FPR=10%', 0):>9.4f}{marker}")
        print()
    print("=" * 70)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ensemble_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'ensemble_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment E: Ensemble of FF-only + Mixed probes")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./ensemble")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
