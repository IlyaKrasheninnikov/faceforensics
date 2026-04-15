"""
Experiment C: Mixed-dataset probe WITH bio-signal features.

Tests whether rPPG and blink features help MORE when the probe is
cross-dataset aware (trained on pooled FF++/CelebDF/DFDC data).

The W5 ablation showed bio-signals are weak on FF++-only probe.
But the mixed probe changes the baseline — if blink/rPPG push
cross-dataset AUC higher, that proves physiological features
provide complementary generalization.

Runs 4 variants on the mixed training set:
  1. backbone_only (baseline = existing mixed probe)
  2. backbone + blink
  3. backbone + rppg
  4. backbone + rppg + blink (full fusion)

Usage:
    python w4_full_train/train_mixed_probe_biosig.py \
        --ff_root /kaggle/input/.../frames \
        --celebdf_root /kaggle/input/.../crop \
        --dfdc_faces_root /kaggle/input/.../dfdc-faces \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --rppg_cache /kaggle/input/.../rppg_v2_300 \
        --blink_cache /kaggle/input/.../blink \
        --out_dir /kaggle/working/mixed_probe_biosig
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


# ─── Scanners ────────────────────────────────────────────────────────────────

def scan_ff(ff_root):
    ff_root = Path(ff_root)
    dirs, labels, manips = [], [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        mdir = ff_root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                dirs.append(str(sd))
                labels.append(label)
                manips.append(manip)
    return dirs, labels, manips


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


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    def __init__(self, video_dirs, labels, clip_len=16, img_size=224,
                 rppg_cache=None, rppg_dim=12, blink_cache=None):
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.rppg_cache = Path(rppg_cache) if rppg_cache else None
        self.rppg_dim = rppg_dim
        self.blink_cache = Path(blink_cache) if blink_cache else None

        self.frame_paths = []
        self.video_dirs_raw = video_dirs
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

        # rPPG features — only available for FF++ (cache structured by manip/video_id)
        rppg_feat = np.zeros(self.rppg_dim, dtype=np.float32)
        vd_raw = self.video_dirs_raw[idx]
        if self.rppg_cache is not None and isinstance(vd_raw, str):
            vpath = Path(vd_raw)
            cache_feat = self.rppg_cache / vpath.parent.name / vpath.name / "rppg_v2_feat.npy"
            if cache_feat.exists():
                loaded = np.load(str(cache_feat)).astype(np.float32)
                if len(loaded) <= self.rppg_dim:
                    rppg_feat[:len(loaded)] = loaded

        # Blink features — only available for FF++
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


# ─── Feature extraction ─────────────────────────────────────────────────────

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


# ─── Probe training ─────────────────────────────────────────────────────────

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
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            loss = crit(probe(tfd[idx]).squeeze(-1), tld[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        avg = epoch_loss / n_batches
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe, best_loss


def eval_probe(probe, feats, labels, device):
    probe.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            probe(feats.to(device)).squeeze(-1).clamp(-20, 20)
        ).cpu().numpy()
    probs = np.nan_to_num(probs, nan=0.5)
    auc = roc_auc_score(labels, probs)
    eer = compute_eer(probs, labels)
    return auc, eer


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
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        bb = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(bb, strict=False)
        print(f"Loaded {len(bb)} backbone tensors")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # ─── Collect all datasets ────────────────────────────────────────────
    dataset_feats = {}  # name → (bb, rppg, blink, labels)

    # FF++
    print("\nScanning FF++...")
    ff_dirs, ff_labels, ff_manips = scan_ff(args.ff_root)
    print(f"  FF++: {len(ff_dirs)} videos")
    ff_dl = DataLoader(
        ClipDataset(ff_dirs, ff_labels, args.clip_len, args.img_size,
                    args.rppg_cache, args.rppg_dim, args.blink_cache),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    ff_bb, ff_rppg, ff_blink, ff_la = extract(model, ff_dl, device, "FF++")
    dataset_feats["FF++"] = (ff_bb, ff_rppg, ff_blink, ff_la)

    # Report cache hit rates for FF++
    rppg_hit = (ff_rppg.abs().sum(1) > 0).float().mean().item() * 100
    blink_hit = (ff_blink.abs().sum(1) > 0).float().mean().item() * 100
    print(f"  FF++ rPPG non-zero: {rppg_hit:.0f}%, Blink non-zero: {blink_hit:.0f}%")

    # CelebDF
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

    # DFDC
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

    # ─── Pool and split (80/20 per dataset) ──────────────────────────────
    rng = random.Random(42)
    train_data = {"bb": [], "rppg": [], "blink": [], "labels": []}
    test_splits = {}  # name → (bb, rppg, blink, labels)

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

        n_real_tr = int((labels[tr_idx] == 0).sum())
        print(f"  {ds_name}: train={n_train} (real={n_real_tr}), test={n-n_train}")

    all_bb = torch.cat(train_data["bb"], dim=0)
    all_rppg = torch.cat(train_data["rppg"], dim=0)
    all_blink = torch.cat(train_data["blink"], dim=0)
    all_labels = np.concatenate(train_data["labels"], axis=0)
    print(f"\n  Total train: {len(all_labels)} samples")

    # ─── Define variants ─────────────────────────────────────────────────
    rppg_dim = all_rppg.shape[1]
    blink_dim = all_blink.shape[1]
    bb_dim = all_bb.shape[1]

    variants = {
        "mixed_backbone_only": {
            "train": all_bb,
            "test_fn": lambda bb, rppg, blink: bb,
            "desc": f"Backbone only ({bb_dim}-d)",
        },
        "mixed_backbone+blink": {
            "train": torch.cat([all_bb, all_blink], dim=1),
            "test_fn": lambda bb, rppg, blink: torch.cat([bb, blink], dim=1),
            "desc": f"Backbone + Blink ({bb_dim}+{blink_dim})",
        },
        "mixed_backbone+rppg": {
            "train": torch.cat([all_bb, all_rppg], dim=1),
            "test_fn": lambda bb, rppg, blink: torch.cat([bb, rppg], dim=1),
            "desc": f"Backbone + rPPG ({bb_dim}+{rppg_dim})",
        },
        "mixed_backbone+rppg+blink": {
            "train": torch.cat([all_bb, all_rppg, all_blink], dim=1),
            "test_fn": lambda bb, rppg, blink: torch.cat([bb, rppg, blink], dim=1),
            "desc": f"Full fusion ({bb_dim}+{rppg_dim}+{blink_dim})",
        },
    }

    # ─── Train and evaluate ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXPERIMENT C: Mixed Probe + Bio-Signal Ablation")
    print(f"{'='*70}")

    all_results = {}

    for vname, vdata in variants.items():
        print(f"\n--- {vname}: {vdata['desc']} ---")
        probe, best_loss = train_probe(vdata["train"], all_labels, device)
        print(f"  Train loss: {best_loss:.4f}")

        vresults = {"description": vdata["desc"], "per_dataset": {}}

        for ds_name, (te_bb, te_rppg, te_blink, te_labels) in test_splits.items():
            if len(set(te_labels)) < 2:
                continue
            te_feats = vdata["test_fn"](te_bb, te_rppg, te_blink)
            auc, eer = eval_probe(probe, te_feats, te_labels, device)
            vresults["per_dataset"][ds_name] = {"auc": auc, "eer": eer, "n": len(te_labels)}
            print(f"  {ds_name:>10s}: AUC={auc:.4f}  EER={eer:.4f}  (n={len(te_labels)})")

        all_results[vname] = vresults

    # ─── Summary comparison ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: Does adding bio-signals help the mixed probe?")
    print(f"{'='*70}")

    ds_names = list(test_splits.keys())
    header = f"{'Variant':<30s}" + "".join(f"  {d:>12s}" for d in ds_names)
    print(header)
    print("-" * len(header))

    baseline_aucs = {}
    for vname in ["mixed_backbone_only", "mixed_backbone+blink",
                   "mixed_backbone+rppg", "mixed_backbone+rppg+blink"]:
        r = all_results.get(vname, {})
        row = f"{vname:<30s}"
        for ds in ds_names:
            auc = r.get("per_dataset", {}).get(ds, {}).get("auc", 0)
            if vname == "mixed_backbone_only":
                baseline_aucs[ds] = auc
                row += f"  {auc:>12.4f}"
            else:
                delta = auc - baseline_aucs.get(ds, 0)
                row += f"  {auc:.4f}({delta:+.3f})"
        print(row)
    print("=" * len(header))

    # ─── Save ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mixed_biosig_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'mixed_biosig_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment C: Mixed probe + bio-signals")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--rppg_dim", type=int, default=12)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--out_dir", default="./mixed_probe_biosig")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
