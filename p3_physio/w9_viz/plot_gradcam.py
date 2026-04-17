"""
W9: Grad-CAM visualization — what does the backbone attend to?

Produces heatmaps on real/fake face frames showing which spatial regions
drive the deepfake detection decision. Uses the actual trained linear probe
so that gradients reflect the classification boundary, not just feature norms.

Previous version (v1) used pooled.sum() as pseudo-logit → heatmap was on
background corners. This v2 trains a quick probe on backbone features, then
backprops through the probe logit for meaningful Grad-CAM.

Usage:
    python w9_viz/plot_gradcam.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/figures
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FF_MANIPULATION_TYPES = {
    "original": 0, "Deepfakes": 1, "Face2Face": 1,
    "FaceSwap": 1, "NeuralTextures": 1, "FaceShifter": 1,
}


class ProbeGradCAM:
    """
    Grad-CAM that routes through a trained linear probe.

    Instead of using a pseudo-score (sum of features), this hooks into
    the last convolutional layer and backprops from the actual probe
    logit. This ensures the heatmap highlights regions that drive the
    real/fake classification decision.
    """

    def __init__(self, encoder, probe, target_layer):
        self.encoder = encoder
        self.probe = probe
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class="fake"):
        """
        Generate Grad-CAM heatmap driven by probe classification.

        target_class: "fake" → gradients for fake prediction (positive logit)
                      "real" → gradients for real prediction (negative logit)
        """
        self.encoder.zero_grad()
        self.probe.zero_grad()

        # Forward through the full EfficientNet-B4 backbone
        features = self.encoder.forward_features(input_tensor)  # (B, C, H, W)
        pooled = features.mean(dim=(2, 3))  # (B, C) — global avg pool

        # Route through the actual trained probe
        logit = self.probe(pooled).squeeze(-1)  # (B,)

        # Backprop: for "fake" we want positive logit, for "real" negative
        if target_class == "real":
            score = -logit
        else:
            score = logit
        score.backward(torch.ones_like(score))

        # Grad-CAM: weight each channel by its gradient importance
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                           mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)  # (B, H, W)

        # Normalize per image to [0, 1]
        for i in range(cam.shape[0]):
            c = cam[i]
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-8:
                cam[i] = (c - c_min) / (c_max - c_min)
            else:
                cam[i] = 0

        return cam.cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()


def load_frame(fpath, img_size=224):
    img = cv2.imread(fpath)
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_display = img_resized.copy()

    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)
    return tensor, img_display


def overlay_cam(img, cam, alpha=0.45):
    """Overlay heatmap on image with reduced alpha for better face visibility."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * img)


class SingleFrameDataset(Dataset):
    """Load one frame per video for quick probe training."""
    def __init__(self, video_dirs, labels, img_size=224):
        self.labels = labels
        self.img_size = img_size
        self.frame_paths = []
        for vd in video_dirs:
            frames = sorted(f for f in os.listdir(vd) if f.endswith(('.png', '.jpg')))
            if frames:
                self.frame_paths.append(os.path.join(vd, frames[len(frames)//2]))
            else:
                self.frame_paths.append(None)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        fpath = self.frame_paths[idx]
        label = self.labels[idx]
        if fpath is None or not os.path.exists(fpath):
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            img = cv2.imread(fpath)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "frame": torch.from_numpy(img).permute(2, 0, 1).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def find_sample_frames(ff_root, n_per_class=4):
    """Find sample real and fake frames for visualization."""
    ff_root = Path(ff_root)
    samples = {"real": [], "fake": []}

    # Real — pick from the middle of the directory list for diversity
    orig_dir = ff_root / "original"
    if orig_dir.exists():
        subdirs = sorted(d for d in orig_dir.iterdir() if d.is_dir())
        # Pick from spread-out positions for variety
        step = max(1, len(subdirs) // (n_per_class + 1))
        for i in range(n_per_class):
            sd = subdirs[min((i + 1) * step, len(subdirs) - 1)]
            pngs = sorted(sd.glob("*.png"))
            if pngs:
                samples["real"].append(str(pngs[len(pngs)//2]))

    # Fake — one per manipulation type
    for manip in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            continue
        subdirs = sorted(d for d in manip_dir.iterdir() if d.is_dir())
        if subdirs:
            # Pick from middle of directory for more representative samples
            sd = subdirs[len(subdirs) // 3]
            pngs = sorted(sd.glob("*.png"))
            if pngs:
                samples["fake"].append((manip, str(pngs[len(pngs)//2])))

    return samples


def main(args):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    cfg = ModelConfig(
        backbone="efficientnet_b4", backbone_pretrained=False,
        temporal_model="mean", temporal_dim=0, clip_len=16,
        img_size=args.img_size, dropout=0.0,
        use_physio_fusion=False, use_pulse_head=False,
        use_blink_head=False, use_motion_model=False,
    )
    model = PhysioNet(cfg).to(device)

    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
        model.load_state_dict(backbone_state, strict=False)
        print(f"Loaded {len(backbone_state)} backbone tensors")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    encoder = model.frame_encoder.encoder

    # ─── Step 1: Quick-train a linear probe on backbone features ──────
    # We need this so Grad-CAM has a meaningful classification signal
    print("\nTraining quick probe for Grad-CAM guidance...")
    ff_root = Path(args.ff_root)
    video_dirs, labels = [], []
    for manip, label in FF_MANIPULATION_TYPES.items():
        mdir = ff_root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if any(sd.glob("*.png")) or any(sd.glob("*.jpg")):
                video_dirs.append(str(sd))
                labels.append(label)
    print(f"  Total videos: {len(video_dirs)}")

    # Use a subset for speed (500 samples is enough to train a probe)
    rng = random.Random(42)
    indices = list(range(len(video_dirs)))
    rng.shuffle(indices)
    n_probe = min(500, len(indices))
    probe_idx = indices[:n_probe]
    probe_dirs = [video_dirs[i] for i in probe_idx]
    probe_labels = [labels[i] for i in probe_idx]

    probe_ds = SingleFrameDataset(probe_dirs, probe_labels, args.img_size)
    probe_dl = DataLoader(probe_ds, batch_size=16, shuffle=False, num_workers=2)

    # Extract features
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(probe_dl, desc="Extracting features", leave=False):
            frames = batch["frame"].to(device)
            feats = encoder(frames)  # (B, 1792)
            all_feats.append(feats.cpu())
            all_labels.extend(batch["label"].numpy().tolist())
    all_feats = torch.cat(all_feats, dim=0)
    all_feats.nan_to_num_(nan=0.0)
    all_labels_arr = np.array(all_labels)

    # Train probe
    dim = all_feats.shape[1]
    probe = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    tfd = all_feats.to(device)
    tld = torch.tensor(all_labels_arr, dtype=torch.float32, device=device)
    for ep in range(15):
        probe.train()
        perm = torch.randperm(len(tfd))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = crit(probe(tfd[idx]).squeeze(-1), tld[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    print(f"  Probe trained (dim={dim})")

    # ─── Step 2: Find target conv layer ──────────────────────────────
    target_layer = None
    if hasattr(encoder, 'conv_head'):
        target_layer = encoder.conv_head
        print(f"  Target layer: conv_head")
    elif hasattr(encoder, 'blocks'):
        target_layer = encoder.blocks[-1]
        print(f"  Target layer: blocks[-1]")
    else:
        for name, module in encoder.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                layer_name = name
        if target_layer:
            print(f"  Target layer: {layer_name}")

    if target_layer is None:
        print("[ERROR] No suitable conv layer found")
        return

    gradcam = ProbeGradCAM(encoder, probe, target_layer)

    # ─── Step 3: Find samples and generate Grad-CAM ──────────────────
    print("\nFinding sample frames...")
    samples = find_sample_frames(args.ff_root)
    print(f"  Real: {len(samples['real'])}, Fake: {len(samples['fake'])}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_real = len(samples["real"])
    n_fake = len(samples["fake"])
    n_cols = max(n_real, n_fake)

    # Увеличиваем высоту фигуры для размещения заголовков
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 10))
    
    # Добавляем отступ для suptitle
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.3)

    # Real row — show what backbone sees as "real-like" regions
    for i, fpath in enumerate(samples["real"]):
        tensor, img_display = load_frame(fpath, args.img_size)
        if tensor is None:
            continue
        tensor = tensor.to(device).requires_grad_(True)
        cam, prob = gradcam.generate(tensor, target_class="fake")
        overlay = overlay_cam(img_display, cam[0])

        ax = axes[0, i] if n_cols > 1 else axes[0]
        ax.imshow(overlay)
        ax.set_title(f"Real #{i+1}\nP(fake)={prob[0]:.2f}",
                     fontweight="bold", color="#2196F3", fontsize=11, pad=10)
        ax.axis("off")

    # Fake row — show which regions trigger the fake detection
    for i, (manip, fpath) in enumerate(samples["fake"]):
        tensor, img_display = load_frame(fpath, args.img_size)
        if tensor is None:
            continue
        tensor = tensor.to(device).requires_grad_(True)
        cam, prob = gradcam.generate(tensor, target_class="fake")
        overlay = overlay_cam(img_display, cam[0])

        ax = axes[1, i] if n_cols > 1 else axes[1]
        ax.imshow(overlay)
        ax.set_title(f"Fake ({manip})\nP(fake)={prob[0]:.2f}",
                     fontweight="bold", color="#F44336", fontsize=11, pad=10)
        ax.axis("off")

    # Hide empty axes
    for row in range(2):
        n_used = n_real if row == 0 else n_fake
        for i in range(n_used, n_cols):
            ax = axes[row, i] if n_cols > 1 else axes[row]
            ax.axis("off")

    # Основной заголовок с явным позиционированием
    fig.suptitle("P3: Probe-Guided Grad-CAM — Regions Driving Fake Detection",
                 fontsize=15, fontweight="bold", y=0.96)
    
    # Подпись внизу
    fig.text(0.5, 0.02,
             "Heatmap shows spatial regions most influential for the deepfake classification decision.\n"
             "Guided by trained linear probe on frozen EfficientNet-B4 backbone features.",
             ha='center', fontsize=9, color='gray', style='italic')

    # Сохраняем с явным bbox_inches='tight' для обрезки пустых краев
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_gradcam.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"\nSaved: {out_dir / 'fig5_gradcam.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W9: Grad-CAM visualization (probe-guided)")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--img_size", type=int, default=224)
    main(p.parse_args())
