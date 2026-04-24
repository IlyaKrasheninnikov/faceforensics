"""
E6: Modern-backbone pilot.  Replace EfficientNet-B4 with a frozen SSL
ViT backbone (CLIP or DINOv2) and re-run the same multi-seed probe.

Produces npz caches in the same schema as multiseed_and_stats.py
`extract`, so the downstream `probe` sub-command runs unchanged.

Why this matters:

  E3 shows the best CelebDF AUC with EfficientNet-B4 is 0.640 ± 0.005
  (mixed probe + rPPG, 5 seeds).  Is this ceiling set by the backbone or
  by the probe-level intervention?  Running the same probe on a modern
  SSL backbone (CLIP ViT-L/14 or DINOv2 ViT-B/14) answers that directly.

Usage on Kaggle:

  # CLIP ViT-L/14 (HuggingFace openai/clip-vit-large-patch14, 768-d)
  python w10_stats/extract_clip_backbone.py \\
      --backbone clip_vitl14 \\
      --ff_root              "$FF" \\
      --celebdf_root         "$CDF" \\
      --dfdc_faces_root      "$DFDC" \\
      --rppg_cache           "$RPPG" \\
      --blink_cache          "$BLINK" \\
      --celebdf_rppg_cache   "$CDF_RPPG" \\
      --celebdf_blink_cache  "$CDF_BLINK" \\
      --dfdc_rppg_cache      "$DFDC_RPPG" \\
      --dfdc_blink_cache     "$DFDC_BLINK" \\
      --cache_dir /kaggle/working/feat_cache_clip \\
      --clip_len 16 --batch_size 4 --num_workers 2

  # DINOv2 ViT-B/14 (facebook/dinov2-base, 768-d)
  python w10_stats/extract_clip_backbone.py \\
      --backbone dinov2_vitb14 \\
      ... (same flags) \\
      --cache_dir /kaggle/working/feat_cache_dinov2

Then:
  python w10_stats/multiseed_and_stats.py probe \\
      --cache_dir /kaggle/working/feat_cache_clip \\
      --out_dir   /kaggle/working/w10_stats_clip \\
      --seeds 0 1 42 1337 2024

The probe output can be compared directly to E3 (same seeds, same
rPPG/blink caches, same identity-aware FF++ split, same 80/20 CDF/DFDC
split — only the backbone differs).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ───────────────────────────────────────────────────────────────────────────
# Dataset scanners + cache_key functions — copied from multiseed_and_stats.py
# to keep this script standalone, but semantically identical.
# ───────────────────────────────────────────────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0, "Deepfakes": 1, "Face2Face": 1,
    "FaceSwap": 1, "NeuralTextures": 1, "FaceShifter": 1,
}


def scan_ff(ff_root):
    ff_root = Path(ff_root)
    dirs, labels, manips, src_ids = [], [], [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        mdir = ff_root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                dirs.append(str(sd))
                labels.append(label)
                manips.append(manip)
                src_ids.append(sd.name.split("_")[0])
    return dirs, labels, manips, src_ids


def scan_celebdf(root):
    root = Path(root)
    dirs, labels = [], []
    for split in ["Test", "Train"]:
        for lname in ["real", "fake"]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            for sd in sorted(d for d in ldir.iterdir() if d.is_dir()):
                if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                    dirs.append(str(sd))
                    labels.append(0 if lname == "real" else 1)
    return dirs, labels


def scan_dfdc_faces(root):
    root = Path(root)
    dirs, labels = [], []
    for split in ["validation", "train"]:
        for lname in ["real", "fake"]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            vid_to_files = {}
            for f in ldir.iterdir():
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    stem = f.stem
                    vid_id = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem.split("_")[0]
                    vid_to_files.setdefault(vid_id, []).append(str(f))
            for vid_id, files in sorted(vid_to_files.items()):
                dirs.append(files)
                labels.append(0 if lname == "real" else 1)
    return dirs, labels


def ff_cache_key(vd):
    vp = Path(vd)
    return vp.parent.name, vp.name


def celebdf_cache_key(vd):
    vp = Path(vd)
    lname = vp.parent.name
    split = vp.parent.parent.name
    return f"{split}_{lname}", vp.name


def dfdc_cache_key(vd):
    if not isinstance(vd, list) or not vd:
        return None, None
    first = Path(vd[0])
    lname = first.parent.name
    split = first.parent.parent.name
    stem = first.stem
    vid_id = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem.split("_")[0]
    return f"{split}_{lname}", vid_id


# ───────────────────────────────────────────────────────────────────────────
# Modern backbone loaders (CLIP + DINOv2)
# ───────────────────────────────────────────────────────────────────────────

def load_backbone(name, device):
    """
    Returns: (model, preprocess_fn, feature_dim)
      preprocess_fn: (img_uint8_HWC_rgb) -> tensor (1, 3, H, W) already on device
      model.forward(x) returns (B, feature_dim) pooled features, frozen.
    """
    import torch
    if name == "clip_vitl14":
        # openai/clip-vit-large-patch14 via HuggingFace transformers.
        # Input: 224x224, ImageNet-style normalization specific to CLIP.
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model_id = "openai/clip-vit-large-patch14"
        model = CLIPVisionModel.from_pretrained(model_id).to(device).eval()
        proc = CLIPImageProcessor.from_pretrained(model_id)
        feat_dim = model.config.hidden_size  # 1024 for ViT-L

        def preprocess_batch(imgs_hwc_rgb):
            # imgs_hwc_rgb: list of np.uint8 HxWx3 RGB
            # CLIPImageProcessor handles resize, crop, normalize
            pixel_values = proc(images=imgs_hwc_rgb, return_tensors="pt")["pixel_values"]
            return pixel_values.to(device, non_blocking=True)

        @torch.no_grad()
        def forward(x):
            out = model(pixel_values=x)
            # pooler_output is the [CLS] pooled feature, shape (B, hidden_size)
            return out.pooler_output.float()

        return forward, preprocess_batch, feat_dim

    elif name == "dinov2_vitb14":
        # facebook/dinov2-base via HuggingFace transformers.
        from transformers import AutoImageProcessor, AutoModel
        model_id = "facebook/dinov2-base"
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(model_id)
        feat_dim = model.config.hidden_size  # 768

        def preprocess_batch(imgs_hwc_rgb):
            pixel_values = proc(images=imgs_hwc_rgb, return_tensors="pt")["pixel_values"]
            return pixel_values.to(device, non_blocking=True)

        @torch.no_grad()
        def forward(x):
            out = model(pixel_values=x)
            # DINOv2: pooler_output is the [CLS] token — primary global feature
            return out.pooler_output.float()

        return forward, preprocess_batch, feat_dim

    else:
        raise ValueError(f"unknown backbone: {name}")


# ───────────────────────────────────────────────────────────────────────────
# Extraction
# ───────────────────────────────────────────────────────────────────────────

def do_extract(args):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}  backbone={args.backbone}")

    forward_fn, preprocess_batch, feat_dim = load_backbone(args.backbone, device)
    print(f"[extract] feature dim={feat_dim}")

    class ClipDataset(Dataset):
        """
        For each video folder / file group, uniformly sample clip_len frames.
        Returns the raw uint8 RGB frame list (we let the backbone's
        HF processor do its own resize/crop/normalize so we don't double-process).
        """
        def __init__(self, video_dirs, labels, rppg_cache, blink_cache, cache_key_fn):
            self.labels = labels
            self.video_dirs_raw = video_dirs
            self.rppg_cache = Path(rppg_cache) if rppg_cache else None
            self.blink_cache = Path(blink_cache) if blink_cache else None
            self.cache_key_fn = cache_key_fn
            self.frame_paths = []
            for vd in video_dirs:
                if isinstance(vd, list):
                    self.frame_paths.append(sorted(vd))
                else:
                    self.frame_paths.append(sorted(
                        os.path.join(vd, f) for f in os.listdir(vd)
                        if f.endswith((".png", ".jpg", ".jpeg"))
                    ))

        def __len__(self):
            return len(self.frame_paths)

        def __getitem__(self, idx):
            frames = self.frame_paths[idx]
            n = len(frames)
            imgs = []
            if n == 0:
                # 16 blank black frames
                imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(args.clip_len)]
            else:
                start = max(0, n - args.clip_len) // 2
                indices = [(start + i) % n for i in range(args.clip_len)]
                for fi in indices:
                    img = cv2.imread(frames[fi])
                    if img is None:
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(img)

            # Load rPPG / blink from their per-dataset caches
            vd = self.video_dirs_raw[idx]
            class_name, video_id = self.cache_key_fn(vd)

            rppg_feat = np.zeros(12, dtype=np.float32)
            if self.rppg_cache is not None and class_name and video_id:
                cp = self.rppg_cache / class_name / video_id / "rppg_v2_feat.npy"
                if cp.exists():
                    loaded = np.load(str(cp)).astype(np.float32)
                    if len(loaded) <= 12:
                        rppg_feat[: len(loaded)] = loaded

            blink_feat = np.zeros(16, dtype=np.float32)
            if self.blink_cache is not None and class_name and video_id:
                bp = self.blink_cache / class_name / video_id / "blink_feat.npy"
                if bp.exists():
                    loaded = np.load(str(bp)).astype(np.float32)
                    if len(loaded) == 16:
                        blink_feat = loaded

            return {
                "imgs": imgs,                      # list of np.uint8 HxWx3 RGB
                "label": float(self.labels[idx]),
                "rppg": rppg_feat,
                "blink": blink_feat,
            }

    def collate(batch):
        # imgs is a list of clip_len uint8 RGB frames; flatten across B
        return batch

    @torch.no_grad()
    def run_one(dirs, labels, tag, rppg_cache, blink_cache, cache_key_fn,
                manips=None, src_ids=None):
        if len(dirs) == 0:
            print(f"[extract] {tag}: EMPTY, skipping")
            return
        ds = ClipDataset(dirs, labels, rppg_cache, blink_cache, cache_key_fn)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False,
                        collate_fn=collate)
        bb_list, rppg_list, blink_list, label_list = [], [], [], []
        t0 = time.time()

        for batch in tqdm(dl, desc=tag, leave=False):
            # batch is a list of {imgs, label, rppg, blink}
            all_imgs = []
            for s in batch:
                all_imgs.extend(s["imgs"])
            # Forward all frames in one call → (B * clip_len, feat_dim)
            pixel_values = preprocess_batch(all_imgs)
            # Some HF processors are huge; chunk if pixel_values is big
            chunk = 32
            feats_chunks = []
            for i in range(0, pixel_values.shape[0], chunk):
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    feats_chunks.append(forward_fn(pixel_values[i:i + chunk]))
            feats = torch.cat(feats_chunks, dim=0)
            feats = feats.view(len(batch), args.clip_len, -1).mean(dim=1)  # (B, feat_dim)
            feats = feats.float().cpu().numpy()
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            bb_list.append(feats)
            rppg_list.append(np.stack([s["rppg"] for s in batch]))
            blink_list.append(np.stack([s["blink"] for s in batch]))
            label_list.append(np.array([s["label"] for s in batch], dtype=np.float32))

        bb = np.concatenate(bb_list, axis=0)
        rppg = np.concatenate(rppg_list, axis=0)
        blink = np.concatenate(blink_list, axis=0)
        lbls = np.concatenate(label_list, axis=0)

        out = {"backbone": bb, "rppg": rppg, "blink": blink, "labels": lbls}
        if manips is not None:
            out["manip"] = np.array(manips)
        if src_ids is not None:
            out["src_id"] = np.array(src_ids)

        rppg_hit = (np.abs(rppg).sum(1) > 0).mean() * 100
        blink_hit = (np.abs(blink).sum(1) > 0).mean() * 100
        print(f"[extract] {tag}: n={len(lbls)} real={int((lbls==0).sum())} "
              f"fake={int((lbls==1).sum())} feat_dim={bb.shape[1]} "
              f"rppg_hit={rppg_hit:.0f}% blink_hit={blink_hit:.0f}% "
              f"time={time.time()-t0:.1f}s")

        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(args.cache_dir) / f"{tag}.npz", **out)

    # FF++
    print("\n[extract] scanning FF++")
    ff_dirs, ff_labels, ff_manips, ff_src = scan_ff(args.ff_root)
    print(f"[extract] FF++: {len(ff_dirs)} videos")
    run_one(ff_dirs, ff_labels, "ff",
            args.rppg_cache, args.blink_cache, ff_cache_key,
            manips=ff_manips, src_ids=ff_src)

    # CelebDF
    if args.celebdf_root:
        print("\n[extract] scanning CelebDF")
        cd_dirs, cd_labels = scan_celebdf(args.celebdf_root)
        print(f"[extract] CelebDF: {len(cd_dirs)} videos")
        run_one(cd_dirs, cd_labels, "celebdf",
                args.celebdf_rppg_cache, args.celebdf_blink_cache, celebdf_cache_key)

    # DFDC
    if args.dfdc_faces_root:
        print("\n[extract] scanning DFDC")
        df_dirs, df_labels = scan_dfdc_faces(args.dfdc_faces_root)
        print(f"[extract] DFDC: {len(df_dirs)} groups")
        run_one(df_dirs, df_labels, "dfdc",
                args.dfdc_rppg_cache, args.dfdc_blink_cache, dfdc_cache_key)

    print(f"\n[extract] DONE. Caches in {args.cache_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="E6: frozen-modern-backbone feature extraction")
    p.add_argument("--backbone", choices=["clip_vitl14", "dinov2_vitb14"], required=True)
    p.add_argument("--ff_root", required=True)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--celebdf_rppg_cache", default=None)
    p.add_argument("--celebdf_blink_cache", default=None)
    p.add_argument("--dfdc_rppg_cache", default=None)
    p.add_argument("--dfdc_blink_cache", default=None)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4,
                   help="ViT-L batches are memory-heavy; 4 videos * 16 frames = 64 images per pass.")
    p.add_argument("--num_workers", type=int, default=2)
    return p


if __name__ == "__main__":
    do_extract(build_parser().parse_args())
