"""
W9: Grad-CAM visualization — what does the backbone attend to?

Produces heatmaps on real/fake face frames showing which regions the
EfficientNet-B4 backbone uses for classification. Expected: cheeks (pulse),
eyes (blink), boundary artifacts.

Usage:
    python w9_viz/plot_gradcam.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/figures
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GradCAM:
    """Grad-CAM for the last conv layer of EfficientNet-B4."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Generate Grad-CAM heatmap for the fake class."""
        self.model.zero_grad()
        # Forward through backbone
        features = self.model.frame_encoder.encoder.forward_features(input_tensor)
        # Global average pool to get logit-like score
        pooled = features.mean(dim=(2, 3))  # (B, C)
        # Use sum as pseudo-logit (higher = more fake-like after probe)
        score = pooled.sum(dim=1)
        score.backward(torch.ones_like(score))

        # Grad-CAM computation
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)  # (B, H, W)

        # Normalize per image
        for i in range(cam.shape[0]):
            c = cam[i]
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-8:
                cam[i] = (c - c_min) / (c_max - c_min)
            else:
                cam[i] = 0

        return cam.cpu().numpy()


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


def overlay_cam(img, cam, alpha=0.5):
    """Overlay heatmap on image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * img)


def find_sample_frames(ff_root, n_per_class=4):
    """Find sample real and fake frames for visualization."""
    ff_root = Path(ff_root)
    samples = {"real": [], "fake": []}

    # Real
    orig_dir = ff_root / "original"
    if orig_dir.exists():
        subdirs = sorted(d for d in orig_dir.iterdir() if d.is_dir())
        for sd in subdirs[:n_per_class]:
            pngs = sorted(sd.glob("*.png"))
            if pngs:
                samples["real"].append(str(pngs[len(pngs)//2]))  # middle frame

    # Fake — one per manipulation type
    for manip in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            continue
        subdirs = sorted(d for d in manip_dir.iterdir() if d.is_dir())
        if subdirs:
            pngs = sorted(subdirs[0].glob("*.png"))
            if pngs:
                samples["fake"].append((manip, str(pngs[len(pngs)//2])))

    return samples


def main(args):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.size": 10, "figure.dpi": 150})

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

    model.eval()

    # Find the last conv layer in EfficientNet-B4
    # In timm's efficientnet, it's model.encoder.conv_head or model.encoder.blocks[-1]
    encoder = model.frame_encoder.encoder
    target_layer = None
    if hasattr(encoder, 'conv_head'):
        target_layer = encoder.conv_head
        print(f"Target layer: conv_head")
    elif hasattr(encoder, 'blocks'):
        target_layer = encoder.blocks[-1]
        print(f"Target layer: blocks[-1]")
    else:
        print("[WARN] Could not find target layer, trying last named module")
        for name, module in encoder.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                layer_name = name
        if target_layer:
            print(f"Target layer: {layer_name}")

    if target_layer is None:
        print("[ERROR] No suitable conv layer found")
        return

    gradcam = GradCAM(model, target_layer)

    # Find samples
    print("\nFinding sample frames...")
    samples = find_sample_frames(args.ff_root)
    print(f"  Real: {len(samples['real'])}, Fake: {len(samples['fake'])}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate Grad-CAM for each sample
    n_real = len(samples["real"])
    n_fake = len(samples["fake"])
    n_cols = max(n_real, n_fake)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Real row
    for i, fpath in enumerate(samples["real"]):
        tensor, img_display = load_frame(fpath, args.img_size)
        if tensor is None:
            continue
        tensor = tensor.to(device).requires_grad_(True)
        cam = gradcam.generate(tensor)[0]
        overlay = overlay_cam(img_display, cam)

        ax = axes[0, i] if n_cols > 1 else axes[0]
        ax.imshow(overlay)
        ax.set_title(f"Real #{i}", fontweight="bold", color="blue")
        ax.axis("off")

    # Fake row
    for i, (manip, fpath) in enumerate(samples["fake"]):
        tensor, img_display = load_frame(fpath, args.img_size)
        if tensor is None:
            continue
        tensor = tensor.to(device).requires_grad_(True)
        cam = gradcam.generate(tensor)[0]
        overlay = overlay_cam(img_display, cam)

        ax = axes[1, i] if n_cols > 1 else axes[1]
        ax.imshow(overlay)
        ax.set_title(f"Fake ({manip})", fontweight="bold", color="red")
        ax.axis("off")

    # Hide empty axes
    for row in range(2):
        n_used = n_real if row == 0 else n_fake
        for i in range(n_used, n_cols):
            ax = axes[row, i] if n_cols > 1 else axes[row]
            ax.axis("off")

    fig.suptitle("P3: Grad-CAM — Backbone Attention on Real vs Fake Faces",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_gradcam.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'fig5_gradcam.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W9: Grad-CAM visualization")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--img_size", type=int, default=224)
    main(p.parse_args())
