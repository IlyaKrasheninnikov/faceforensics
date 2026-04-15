"""
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
