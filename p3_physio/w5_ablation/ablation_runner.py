"""
W5: Ablation study — linear probe on frozen backbone with all signal combinations.

Extracts backbone features once (~20 min), then runs 6 ablation variants as linear
probes on different feature concatenations (seconds each). Per-manipulation breakdown
for every variant. Also includes FakeCatcher-style SVM baseline on rPPG features only.

Variants:
  1. backbone_only      — EfficientNet-B4 features (1792-d)
  2. backbone+rppg      — backbone + rPPG features (1792 + rppg_dim)
  3. backbone+blink     — backbone + blink features (1792 + 16)
  4. backbone+rppg+blink — backbone + rPPG + blink (1792 + rppg_dim + 16)
  5. rppg_only          — rPPG features alone (rppg_dim)
  6. blink_only         — blink features alone (16)
  7. fakecatcher_svm    — SVM on rPPG features (FakeCatcher reproduction)

Usage:
    python w5_ablation/ablation_runner.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --rppg_cache /kaggle/input/.../rppg_v2_300 \
        --blink_cache /kaggle/input/.../blink \
        --rppg_version 2 --rppg_dim 12 \
        --out_dir /kaggle/working/ablation
"""

import argparse
import json
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("sklearn required: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig

try:
    import trackio
    TRACKIO_AVAILABLE = True
except Exception:
    TRACKIO_AVAILABLE = False


# ─── Constants ───────────────────────────────────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Dataset ─────────────────────────────────────────────────────────────────

def scan_video_folders(ff_root: str):
    ff_root = Path(ff_root)
    video_dirs, labels, src_ids, manip_names = [], [], [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            continue
        subdirs = sorted([d for d in manip_dir.iterdir() if d.is_dir()])
        valid = 0
        for sd in subdirs:
            has_frames = any(sd.glob("*.png")) or any(sd.glob("*.jpg"))
            if not has_frames:
                continue
            video_dirs.append(str(sd))
            labels.append(label)
            src_ids.append(sd.name.split("_")[0])
            manip_names.append(manip)
            valid += 1
        print(f"  {manip}: {valid} video folders")
    return video_dirs, labels, src_ids, manip_names


class ClipDataset(Dataset):
    def __init__(self, video_dirs, labels, manips, clip_len=16, img_size=224,
                 rppg_cache=None, rppg_dim=12, rppg_version=2,
                 blink_cache=None):
        self.video_dirs = video_dirs
        self.labels = labels
        self.manips = manips
        self.clip_len = clip_len
        self.img_size = img_size
        self.rppg_cache = Path(rppg_cache) if rppg_cache else None
        self.rppg_dim = rppg_dim
        self.rppg_version = rppg_version
        self.blink_cache = Path(blink_cache) if blink_cache else None

        self.frame_lists = []
        for vd in video_dirs:
            frames = sorted([f for f in os.listdir(vd) if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.frame_lists.append(frames)

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        label = self.labels[idx]
        manip = self.manips[idx]
        all_frames = self.frame_lists[idx]
        n = len(all_frames)

        if n == 0:
            clip = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            max_start = max(0, n - self.clip_len)
            start = max_start // 2  # center clip
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

        # Load rPPG features
        rppg_feat = np.zeros(self.rppg_dim, dtype=np.float32)
        if self.rppg_cache is not None:
            vpath = Path(vdir)
            fname = "rppg_v2_feat.npy" if self.rppg_version == 2 else "rppg_feat.npy"
            cache_feat = self.rppg_cache / vpath.parent.name / vpath.name / fname
            if cache_feat.exists():
                loaded = np.load(str(cache_feat)).astype(np.float32)
                if len(loaded) <= self.rppg_dim:
                    rppg_feat[:len(loaded)] = loaded

        # Load blink features
        blink_feat = np.zeros(16, dtype=np.float32)
        if self.blink_cache is not None:
            vpath_b = Path(vdir)
            bp = self.blink_cache / vpath_b.parent.name / vpath_b.name / "blink_feat.npy"
            if bp.exists():
                loaded_b = np.load(str(bp)).astype(np.float32)
                if len(loaded_b) == 16:
                    blink_feat = loaded_b

        return {
            "frames": clip_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "manip": manip,
            "rppg_feat": torch.from_numpy(rppg_feat),
            "blink_feat": torch.from_numpy(blink_feat),
        }


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


def eval_per_manipulation(probs, labels, manips):
    """Compute per-manipulation AUC (each fake type vs all reals in split)."""
    results = {}
    real_mask = np.array([m == "original" for m in manips])
    real_labels = labels[real_mask]
    real_probs = probs[real_mask]

    for manip_name in sorted(set(manips)):
        if manip_name == "original":
            results[manip_name] = {"n": int(real_mask.sum()), "type": "real"}
            continue
        fake_mask = np.array([m == manip_name for m in manips])
        if fake_mask.sum() < 5:
            continue
        combined_labels = np.concatenate([real_labels, labels[fake_mask]])
        combined_probs = np.concatenate([real_probs, probs[fake_mask]])
        if len(set(combined_labels)) < 2:
            continue
        m_auc = roc_auc_score(combined_labels, combined_probs)
        m_eer = compute_eer(combined_probs, combined_labels)
        results[manip_name] = {"auc": m_auc, "eer": m_eer, "n": int(fake_mask.sum())}
    return results


# ─── Linear Probe ────────────────────────────────────────────────────────────

def train_linear_probe(train_feats, train_labels, val_feats, val_labels,
                       epochs=15, lr=1e-3, batch_size=256, device="cpu"):
    """Train a linear probe and return best model state + metrics."""
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_feats_d = train_feats.to(device)
    train_labels_d = torch.tensor(train_labels, dtype=torch.float32, device=device)
    val_feats_d = val_feats.to(device)

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(train_feats_d))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(train_feats_d[idx]).squeeze(-1)
            loss = criterion(logits, train_labels_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats_d).squeeze(-1)
            val_probs = torch.sigmoid(val_logits.clamp(-20, 20)).cpu().numpy()
            val_probs = np.nan_to_num(val_probs, nan=0.5)
            val_auc = roc_auc_score(val_labels, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    # Load best and return
    probe.load_state_dict(best_state)
    return probe, best_val_auc


def evaluate_probe(probe, feats, labels, manips, device="cpu"):
    """Evaluate a trained probe, return metrics + per-manipulation breakdown."""
    probe.eval()
    with torch.no_grad():
        logits = probe(feats.to(device)).squeeze(-1)
        probs = torch.sigmoid(logits.clamp(-20, 20)).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)

    labels_arr = np.array(labels)
    auc = roc_auc_score(labels_arr, probs)
    eer = compute_eer(probs, labels_arr)
    ap = average_precision_score(labels_arr, probs)
    ece = compute_ece(probs, labels_arr)
    per_manip = eval_per_manipulation(probs, labels_arr, manips)

    return {"auc": auc, "eer": eer, "ap": ap, "ece": ece, "per_manip": per_manip}


# ─── FakeCatcher SVM baseline ───────────────────────────────────────────────

def fakecatcher_svm(train_rppg, train_labels, test_rppg, test_labels, test_manips):
    """FakeCatcher-style: SVM on rPPG features only."""
    X_train = train_rppg.numpy()
    X_test = test_rppg.numpy()
    # Replace NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(probability=True, kernel="rbf", random_state=42)
    svm.fit(X_train, train_labels)
    scores = svm.predict_proba(X_test)[:, 1]

    labels_arr = np.array(test_labels)
    auc = roc_auc_score(labels_arr, scores)
    eer = compute_eer(scores, labels_arr)
    ap = average_precision_score(labels_arr, scores)
    ece = compute_ece(scores, labels_arr)
    per_manip = eval_per_manipulation(scores, labels_arr, test_manips)

    return {"auc": auc, "eer": eer, "ap": ap, "ece": ece, "per_manip": per_manip}


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Data ────────────────────────────────────────────────────────────
    print("\nScanning dataset...")
    video_dirs, labels, src_ids, manip_names = scan_video_folders(args.ff_root)
    n_total = len(video_dirs)
    print(f"Total: {n_total} videos, real={labels.count(0)}, fake={labels.count(1)}")

    # Identity-aware split (identical to train_physio_png.py)
    id_to_indices = {}
    for i, sid in enumerate(src_ids):
        id_to_indices.setdefault(sid, []).append(i)
    unique_ids = sorted(id_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_ids)
    n_ids = len(unique_ids)
    n_train_ids = int(n_ids * 0.8)
    n_val_ids = int(n_ids * 0.1)
    train_ids_set = set(unique_ids[:n_train_ids])
    val_ids_set = set(unique_ids[n_train_ids:n_train_ids + n_val_ids])

    train_idx, val_idx, test_idx = [], [], []
    for sid, indices in id_to_indices.items():
        if sid in train_ids_set:
            train_idx.extend(indices)
        elif sid in val_ids_set:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    def subset(indices):
        return ([video_dirs[i] for i in indices],
                [labels[i] for i in indices],
                [manip_names[i] for i in indices])

    train_dirs, train_labels, train_manips = subset(train_idx)
    val_dirs, val_labels, val_manips = subset(val_idx)
    test_dirs, test_labels, test_manips = subset(test_idx)
    print(f"Split: {n_train_ids}/{n_val_ids}/{n_ids-n_train_ids-n_val_ids} source IDs")
    print(f"Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)}")

    train_ds = ClipDataset(train_dirs, train_labels, train_manips,
                           args.clip_len, args.img_size,
                           args.rppg_cache, args.rppg_dim, args.rppg_version,
                           args.blink_cache)
    val_ds = ClipDataset(val_dirs, val_labels, val_manips,
                         args.clip_len, args.img_size,
                         args.rppg_cache, args.rppg_dim, args.rppg_version,
                         args.blink_cache)
    test_ds = ClipDataset(test_dirs, test_labels, test_manips,
                          args.clip_len, args.img_size,
                          args.rppg_cache, args.rppg_dim, args.rppg_version,
                          args.blink_cache)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # ─── Build backbone & extract features ────────────────────────────────
    cfg = ModelConfig(
        backbone="efficientnet_b4",
        backbone_pretrained=False,
        temporal_model="mean", temporal_dim=0,
        clip_len=args.clip_len, img_size=args.img_size,
        dropout=0.0,
        use_physio_fusion=False, use_pulse_head=False,
        use_blink_head=False, use_motion_model=False,
    )
    model = PhysioNet(cfg).to(device)

    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        print(f"\nLoading backbone from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded {len(backbone_state)} backbone tensors")
    elif args.backbone_weights and Path(args.backbone_weights).exists():
        print(f"\nLoading backbone from local weights: {args.backbone_weights}")
        # handled by ModelConfig.backbone_local_weights
    else:
        print("\n[WARN] No backbone checkpoint — using random init")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Extract all features
    @torch.no_grad()
    def extract_all(dataloader, desc="Extract"):
        all_backbone, all_rppg, all_blink = [], [], []
        all_labels, all_manips = [], []
        for batch in tqdm(dataloader, desc=desc, leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                frame_feats = model.frame_encoder(frames)
                pooled = frame_feats.mean(dim=1)
            all_backbone.append(pooled.float().cpu())
            all_rppg.append(batch["rppg_feat"])
            all_blink.append(batch["blink_feat"])
            all_labels.extend(batch["label"].numpy().tolist())
            all_manips.extend(batch["manip"])
        backbone = torch.cat(all_backbone, dim=0)
        backbone.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        rppg = torch.cat(all_rppg, dim=0)
        blink = torch.cat(all_blink, dim=0)
        return backbone, rppg, blink, np.array(all_labels), all_manips

    print("\nExtracting features (one-time)...")
    t0 = time.time()
    train_bb, train_rppg, train_blink, train_labels_arr, train_manips_list = extract_all(train_dl, "Train")
    val_bb, val_rppg, val_blink, val_labels_arr, val_manips_list = extract_all(val_dl, "Val")
    test_bb, test_rppg, test_blink, test_labels_arr, test_manips_list = extract_all(test_dl, "Test")
    print(f"Feature extraction: {time.time()-t0:.1f}s")
    print(f"  Backbone: {train_bb.shape[1]}-d, rPPG: {train_rppg.shape[1]}-d, Blink: {train_blink.shape[1]}-d")

    # Cache diagnostics
    rppg_nonzero_pct = (train_rppg.abs().sum(1) > 0).float().mean().item() * 100
    blink_nonzero_pct = (train_blink.abs().sum(1) > 0).float().mean().item() * 100
    print(f"  rPPG non-zero: {rppg_nonzero_pct:.0f}%, Blink non-zero: {blink_nonzero_pct:.0f}%")

    # ─── Define ablation variants ─────────────────────────────────────────
    variants = {
        "1_backbone_only": {
            "train": train_bb, "val": val_bb, "test": test_bb,
            "desc": "EfficientNet-B4 backbone features (1792-d)",
        },
        "2_backbone+rppg": {
            "train": torch.cat([train_bb, train_rppg], dim=1),
            "val": torch.cat([val_bb, val_rppg], dim=1),
            "test": torch.cat([test_bb, test_rppg], dim=1),
            "desc": f"Backbone + rPPG ({train_bb.shape[1]}+{train_rppg.shape[1]})",
        },
        "3_backbone+blink": {
            "train": torch.cat([train_bb, train_blink], dim=1),
            "val": torch.cat([val_bb, val_blink], dim=1),
            "test": torch.cat([test_bb, test_blink], dim=1),
            "desc": f"Backbone + Blink ({train_bb.shape[1]}+{train_blink.shape[1]})",
        },
        "4_backbone+rppg+blink": {
            "train": torch.cat([train_bb, train_rppg, train_blink], dim=1),
            "val": torch.cat([val_bb, val_rppg, val_blink], dim=1),
            "test": torch.cat([test_bb, test_rppg, test_blink], dim=1),
            "desc": f"Backbone + rPPG + Blink (full fusion)",
        },
        "5_rppg_only": {
            "train": train_rppg, "val": val_rppg, "test": test_rppg,
            "desc": f"rPPG features only ({train_rppg.shape[1]}-d)",
        },
        "6_blink_only": {
            "train": train_blink, "val": val_blink, "test": test_blink,
            "desc": f"Blink features only ({train_blink.shape[1]}-d)",
        },
    }

    # ─── Run ablations ────────────────────────────────────────────────────
    all_results = {}
    print(f"\n{'='*80}")
    print("ABLATION STUDY — Linear Probe on Frozen Backbone")
    print(f"{'='*80}")

    for vname, vdata in variants.items():
        print(f"\n--- {vname}: {vdata['desc']} ---")
        feat_dim = vdata["train"].shape[1]
        print(f"  Feature dim: {feat_dim}")

        probe, best_val_auc = train_linear_probe(
            vdata["train"], train_labels_arr,
            vdata["val"], val_labels_arr,
            epochs=args.probe_epochs, lr=args.lr,
            batch_size=args.probe_batch_size, device=device,
        )
        print(f"  Best val AUC: {best_val_auc:.4f}")

        # Evaluate on val
        val_metrics = evaluate_probe(probe, vdata["val"], val_labels_arr, val_manips_list, device)
        print(f"  Val:  AUC={val_metrics['auc']:.4f}  EER={val_metrics['eer']:.4f}  "
              f"AP={val_metrics['ap']:.4f}  ECE={val_metrics['ece']:.4f}")

        # Evaluate on test
        test_metrics = evaluate_probe(probe, vdata["test"], test_labels_arr, test_manips_list, device)
        print(f"  Test: AUC={test_metrics['auc']:.4f}  EER={test_metrics['eer']:.4f}  "
              f"AP={test_metrics['ap']:.4f}  ECE={test_metrics['ece']:.4f}")

        # Per-manipulation
        print(f"  Per-manipulation (test):")
        for mname, mdata in sorted(test_metrics["per_manip"].items()):
            if "auc" in mdata:
                print(f"    {mname:20s}: AUC={mdata['auc']:.4f}  EER={mdata['eer']:.4f}  (n={mdata['n']})")

        all_results[vname] = {
            "description": vdata["desc"],
            "feat_dim": feat_dim,
            "val_auc": val_metrics["auc"], "val_eer": val_metrics["eer"],
            "val_ap": val_metrics["ap"], "val_ece": val_metrics["ece"],
            "test_auc": test_metrics["auc"], "test_eer": test_metrics["eer"],
            "test_ap": test_metrics["ap"], "test_ece": test_metrics["ece"],
            "test_per_manip": test_metrics["per_manip"],
        }

    # ─── FakeCatcher SVM baseline ─────────────────────────────────────────
    print(f"\n--- 7_fakecatcher_svm: SVM on rPPG features (FakeCatcher reproduction) ---")
    fc_metrics = fakecatcher_svm(
        train_rppg, train_labels_arr,
        test_rppg, test_labels_arr, test_manips_list,
    )
    print(f"  Test: AUC={fc_metrics['auc']:.4f}  EER={fc_metrics['eer']:.4f}  "
          f"AP={fc_metrics['ap']:.4f}  ECE={fc_metrics['ece']:.4f}")
    print(f"  Per-manipulation (test):")
    for mname, mdata in sorted(fc_metrics["per_manip"].items()):
        if "auc" in mdata:
            print(f"    {mname:20s}: AUC={mdata['auc']:.4f}  EER={mdata['eer']:.4f}  (n={mdata['n']})")

    all_results["7_fakecatcher_svm"] = {
        "description": "FakeCatcher-style SVM on rPPG features",
        "feat_dim": train_rppg.shape[1],
        "val_auc": 0.0, "val_eer": 0.0, "val_ap": 0.0, "val_ece": 0.0,
        "test_auc": fc_metrics["auc"], "test_eer": fc_metrics["eer"],
        "test_ap": fc_metrics["ap"], "test_ece": fc_metrics["ece"],
        "test_per_manip": fc_metrics["per_manip"],
    }

    # ─── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<30s} {'Feat':>5s} {'Val AUC':>8s} {'Test AUC':>9s} "
          f"{'Test EER':>9s} {'Test AP':>8s} {'Test ECE':>9s}")
    print("-" * 80)
    for vname in sorted(all_results.keys()):
        r = all_results[vname]
        print(f"{vname:<30s} {r['feat_dim']:>5d} {r['val_auc']:>8.4f} {r['test_auc']:>9.4f} "
              f"{r['test_eer']:>9.4f} {r['test_ap']:>8.4f} {r['test_ece']:>9.4f}")
    print("=" * 80)

    # Per-manipulation comparison table
    fake_manips = [m for m in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
                   if any(m in str(r.get("test_per_manip", {})) for r in all_results.values())]

    if fake_manips:
        print(f"\nPER-MANIPULATION AUC (Test)")
        header = f"{'Variant':<30s}" + "".join(f"{m:>16s}" for m in fake_manips)
        print(header)
        print("-" * len(header))
        for vname in sorted(all_results.keys()):
            pm = all_results[vname].get("test_per_manip", {})
            row = f"{vname:<30s}"
            for m in fake_manips:
                if m in pm and "auc" in pm[m]:
                    row += f"{pm[m]['auc']:>16.4f}"
                else:
                    row += f"{'—':>16s}"
            print(row)
        print("=" * len(header))

    # ─── Save results ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'ablation_results.json'}")

    # CSV summary
    csv_rows = []
    for vname, r in sorted(all_results.items()):
        row = {"variant": vname, "feat_dim": r["feat_dim"],
               "val_auc": r["val_auc"], "test_auc": r["test_auc"],
               "test_eer": r["test_eer"], "test_ap": r["test_ap"],
               "test_ece": r["test_ece"]}
        for m in fake_manips:
            pm = r.get("test_per_manip", {})
            row[f"auc_{m}"] = pm.get(m, {}).get("auc", None)
        csv_rows.append(row)

    # Write CSV manually (avoid pandas dependency)
    if csv_rows:
        keys = csv_rows[0].keys()
        with open(out_dir / "ablation_results.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for row in csv_rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"CSV saved: {out_dir / 'ablation_results.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W5 Ablation: linear probe on frozen backbone")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", default=None, help="PhysioNet checkpoint (backbone)")
    p.add_argument("--backbone_weights", default=None, help="Local .pth for backbone")
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--rppg_version", type=int, default=2)
    p.add_argument("--rppg_dim", type=int, default=12)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--out_dir", default="./ablation")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--probe_batch_size", type=int, default=256)
    p.add_argument("--probe_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
