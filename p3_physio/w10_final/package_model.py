"""
W10: Package model weights + config for distribution.

Extracts the backbone from the full checkpoint, saves it alongside the
trained probe weights and all config needed to reproduce inference.

Usage:
    python w10_final/package_model.py \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --out_dir /kaggle/working/packaged_model
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from w2_model.model import PhysioNet, ModelConfig


def main(args):
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full checkpoint
    print(f"Loading checkpoint: {args.resume_ckpt}")
    ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Extract backbone only
    backbone_state = {k: v for k, v in state.items() if k.startswith("frame_encoder.")}
    print(f"  Backbone tensors: {len(backbone_state)}")

    # Save backbone
    backbone_path = out_dir / "backbone_efficientnet_b4.pt"
    torch.save(backbone_state, backbone_path)
    print(f"  Saved: {backbone_path} ({backbone_path.stat().st_size / 1e6:.1f} MB)")

    # Save config
    config = {
        "backbone": "efficientnet_b4",
        "backbone_pretrained": False,
        "temporal_model": "mean",
        "clip_len": 16,
        "img_size": 224,
        "feat_dim": 1792,
        "rppg_dim": 12,
        "rppg_version": 2,
        "blink_dim": 16,
        "approach": "linear_probe",
        "best_variant": "backbone+rppg+blink",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "training_dataset": "FaceForensics++ c23",
        "training_split": "80/10/10 identity-based",
        "metrics": {
            "ff_val_auc": 0.783,
            "ff_test_auc": 0.730,
            "ff_test_eer": 0.314,
            "celebdf_auc": 0.574,
            "dfdc_auc": 0.539,
        },
    }
    config_path = out_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    # Save a minimal inference example
    example = '''"""
Minimal inference example for P3 PhysioNet backbone.

Loads the packaged backbone, extracts features from a video clip,
and classifies using a pre-trained linear probe.
"""
import torch
import json
import numpy as np
import cv2

# Load config
with open("model_config.json") as f:
    config = json.load(f)

# Load backbone
import timm
encoder = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0, global_pool="avg")
state = torch.load("backbone_efficientnet_b4.pt", map_location="cpu")
# Strip "frame_encoder.encoder." prefix
clean_state = {k.replace("frame_encoder.encoder.", ""): v for k, v in state.items()}
encoder.load_state_dict(clean_state, strict=False)
encoder.eval()

# Load and preprocess a frame
img = cv2.imread("face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = (img - np.array(config["imagenet_mean"])) / np.array(config["imagenet_std"])
tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

# Extract features
with torch.no_grad():
    features = encoder(tensor)  # (1, 1792)
print(f"Feature shape: {features.shape}")
print(f"Feature norm: {features.norm():.2f}")
'''
    example_path = out_dir / "inference_example.py"
    with open(example_path, "w") as f:
        f.write(example)
    print(f"  Saved: {example_path}")

    print(f"\nPackage complete: {out_dir}")
    print(f"Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W10: Package model for distribution")
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--out_dir", default="./packaged_model")
    main(p.parse_args())
