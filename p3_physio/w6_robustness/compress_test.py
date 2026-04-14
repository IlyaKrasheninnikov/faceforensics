"""
W6: Robustness testing — evaluate model under compression and visual perturbations.

Tests the linear probe under:
  1. JPEG compression (quality 50, 30, 10)
  2. Gaussian blur (sigma 1, 2, 3)
  3. Gaussian noise (sigma 5, 10, 20)
  4. Downscale + upscale (factor 2, 4)

Applies perturbations to extracted frames at load time, re-extracts backbone features,
and evaluates the same probe trained on clean data.

Usage:
    python w6_robustness/compress_test.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/robustness
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
    raise ImportError("sklearn required: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FF_MANIPULATION_TYPES = {
    "original": 0, "Deepfakes": 1, "Face2Face": 1,
    "FaceSwap": 1, "NeuralTextures": 1, "FaceShifter": 1,
}


# ─── Perturbations ──────────────────────────────────────────────────────────

def apply_jpeg_compression(img_uint8, quality=50):
    """Simulate JPEG compression artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img_uint8, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_gaussian_blur(img_uint8, sigma=2.0):
    ksize = int(sigma * 4) | 1  # odd kernel
    return cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)


def apply_gaussian_noise(img_uint8, sigma=10):
    noise = np.random.randn(*img_uint8.shape).astype(np.float32) * sigma
    noisy = img_uint8.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_downscale(img_uint8, factor=2):
    h, w = img_uint8.shape[:2]
    small = cv2.resize(img_uint8, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS = {
    "clean": lambda img: img,
    "jpeg_q50": lambda img: apply_jpeg_compression(img, 50),
    "jpeg_q30": lambda img: apply_jpeg_compression(img, 30),
    "jpeg_q10": lambda img: apply_jpeg_compression(img, 10),
    "blur_s1": lambda img: apply_gaussian_blur(img, 1.0),
    "blur_s2": lambda img: apply_gaussian_blur(img, 2.0),
    "blur_s3": lambda img: apply_gaussian_blur(img, 3.0),
    "noise_s5": lambda img: apply_gaussian_noise(img, 5),
    "noise_s10": lambda img: apply_gaussian_noise(img, 10),
    "noise_s20": lambda img: apply_gaussian_noise(img, 20),
    "downscale_2x": lambda img: apply_downscale(img, 2),
    "downscale_4x": lambda img: apply_downscale(img, 4),
}


# ─── Dataset ─────────────────────────────────────────────────────────────────

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


class PerturbedClipDataset(Dataset):
    def __init__(self, video_dirs, labels, clip_len=16, img_size=224, perturb_fn=None):
        self.video_dirs = video_dirs
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.perturb_fn = perturb_fn or (lambda x: x)
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
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    # Apply perturbation in BGR uint8 space
                    img = self.perturb_fn(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    # Scan dataset
    print("\nScanning dataset...")
    video_dirs, labels, src_ids = scan_video_folders(args.ff_root)

    # Identity split
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

    train_idx = [i for i, s in enumerate(src_ids) if s in train_ids]
    val_idx = [i for i, s in enumerate(src_ids) if s in val_ids]
    test_idx = [i for i, s in enumerate(src_ids) if s in test_ids]

    train_dirs = [video_dirs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_dirs = [video_dirs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_dirs = [video_dirs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)}")

    # ─── Step 1: Extract clean features and train probe ──────────────────────
    print("\nExtracting CLEAN features (train + val)...")
    t0 = time.time()

    @torch.no_grad()
    def extract_feats(dirs, lbls, perturb_fn=None, desc=""):
        ds = PerturbedClipDataset(dirs, lbls, args.clip_len, args.img_size, perturb_fn)
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

    train_feats, train_labels_arr = extract_feats(train_dirs, train_labels, None, "Clean train")
    val_feats, val_labels_arr = extract_feats(val_dirs, val_labels, None, "Clean val")
    print(f"  Clean extraction: {time.time()-t0:.1f}s")

    # Train probe on clean data
    print("\nTraining probe on clean data...")
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-3)
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
            vp = torch.sigmoid(probe(val_f_d).squeeze(-1).clamp(-20,20)).cpu().numpy()
            vp = np.nan_to_num(vp, nan=0.5)
            va = roc_auc_score(val_labels_arr, vp)
        if va > best_auc:
            best_auc = va
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    print(f"  Clean val AUC: {best_auc:.4f}")

    # ─── Step 2: Evaluate under each perturbation ────────────────────────────
    all_results = {}
    print(f"\n{'='*70}")
    print("ROBUSTNESS EVALUATION")
    print(f"{'='*70}")

    for pname, pfn in PERTURBATIONS.items():
        print(f"\n--- {pname} ---")
        t0 = time.time()
        test_feats, test_labels_arr = extract_feats(test_dirs, test_labels, pfn, pname)
        ext_time = time.time() - t0

        probe.eval()
        with torch.no_grad():
            logits = probe(test_feats.to(device)).squeeze(-1)
            probs = torch.sigmoid(logits.clamp(-20, 20)).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.5)

        auc = roc_auc_score(test_labels_arr, probs)
        eer = compute_eer(probs, test_labels_arr)
        ap = average_precision_score(test_labels_arr, probs)
        ece = compute_ece(probs, test_labels_arr)

        all_results[pname] = {"auc": auc, "eer": eer, "ap": ap, "ece": ece}
        print(f"  AUC={auc:.4f}  EER={eer:.4f}  AP={ap:.4f}  ECE={ece:.4f}  ({ext_time:.1f}s)")

    # ─── Summary ─────────────────────────────────────────────────────────────
    clean_auc = all_results["clean"]["auc"]
    print(f"\n{'='*70}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Perturbation':<20s} {'AUC':>8s} {'Drop':>8s} {'EER':>8s} {'AP':>8s}")
    print("-" * 50)
    for pname in PERTURBATIONS:
        r = all_results[pname]
        drop = r["auc"] - clean_auc
        print(f"{pname:<20s} {r['auc']:>8.4f} {drop:>+8.4f} {r['eer']:>8.4f} {r['ap']:>8.4f}")
    print("=" * 50)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "robustness_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'robustness_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W6: Robustness evaluation")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./robustness")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
