"""
W2: PhysioNet — Physiological deepfake detection model.

Architecture:
  - EfficientNet-B4 backbone (pretrained, timm)
  - Temporal block: Mamba SSM OR Transformer OR BiLSTM (runtime flag)
  - Explicit rPPG spectrum feature (128-d FFT of extracted pulse)
  - Explicit blink feature vector (16-d stats)
  - Three output heads:
      1. Classification head (real/fake)
      2. Pulse regression head (predict pulse waveform from visual features)
      3. Blink sequence head (predict per-frame eye-closed probability)

Usage:
    from w2_model.model import PhysioNet, ModelConfig
    cfg = ModelConfig()
    model = PhysioNet(cfg)
    out = model(frames, rppg_feat, blink_feat)  # see forward() for shapes
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARN] timm not installed — backbone will be a simple CNN stub")

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Backbone
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    backbone_local_weights: str = None   # path to local .pth file (skip download)
    backbone_freeze_epochs: int = 2      # freeze backbone for first N epochs

    # Temporal
    temporal_model: str = "transformer"  # "mamba" | "transformer" | "lstm"
    temporal_layers: int = 4
    temporal_dim: int = 512
    temporal_heads: int = 8              # for transformer only
    temporal_dropout: float = 0.1

    # Input features
    clip_len: int = 64                   # number of frames per clip
    img_size: int = 224
    rppg_feature_dim: int = 128          # dim of explicit rPPG FFT feature
    blink_feature_dim: int = 16          # dim of explicit blink stats feature

    # Fusion & heads
    fusion_dim: int = 512
    dropout: float = 0.3
    num_classes: int = 1                 # binary: real/fake

    # Loss control
    use_pulse_head: bool = True
    use_blink_head: bool = True

    # Ablation flags
    use_physio_fusion: bool = True     # False = ignore rPPG/blink in fusion (cls from temporal only)
    temporal_pool: str = "mean"        # "mean" = simple mean pool, "transformer" = full transformer
    use_motion_model: bool = False     # True = add frame-diff motion branch
    motion_dim: int = 64              # output dim of motion encoder


# ─── Backbone ─────────────────────────────────────────────────────────────────

class FrameEncoder(nn.Module):
    """
    Per-frame feature extractor using EfficientNet-B4.
    Processes (B*T, C, H, W) and returns (B, T, D) frame features.
    """

    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True,
                 local_weights: str = None):
        super().__init__()
        if TIMM_AVAILABLE:
            if local_weights is not None:
                # Load from local .pth file — no network download
                self.encoder = timm.create_model(
                    backbone,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
                state = torch.load(local_weights, map_location="cpu", weights_only=True)
                # Filter out classifier keys (we set num_classes=0)
                state = {k: v for k, v in state.items() if "classifier" not in k}
                self.encoder.load_state_dict(state, strict=False)
                print(f"  Loaded backbone weights from: {local_weights}")
            else:
                self.encoder = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
            self.out_dim = self.encoder.num_features
        else:
            # Fallback: tiny conv stack for testing without timm
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.out_dim = 64

    def forward(self, x: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, T, D)

        Processes frames in chunks to avoid OOM on long clips.
        Uses gradient checkpointing to save memory when training.
        chunk_size=4 is safe for T4 16GB with EfficientNet-B4.
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = []
        for i in range(0, B * T, chunk_size):
            chunk = x[i:i + chunk_size]
            if self.training and chunk.requires_grad:
                feat_chunk = torch.utils.checkpoint.checkpoint(
                    self.encoder, chunk, use_reentrant=False
                )
            else:
                feat_chunk = self.encoder(chunk)
            feats.append(feat_chunk)
        feat = torch.cat(feats, dim=0)
        return feat.view(B, T, -1)       # (B, T, D)


# ─── Temporal Models ──────────────────────────────────────────────────────────

class TransformerTemporal(nn.Module):
    """Temporal Transformer encoder with learnable positional encoding."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, d_model))  # max T=256
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  (B, T, D)"""
        T = x.size(1)
        x = x + self.pos_embed[:, :T, :]
        return self.transformer(x)


class LSTMTemporal(nn.Module):
    """Bidirectional LSTM temporal encoder."""

    def __init__(self, d_input: int, d_model: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_in)  →  (B, T, d_model)"""
        x = self.proj(x)
        out, _ = self.lstm(x)
        return out


class MambaTemporal(nn.Module):
    """Mamba SSM temporal encoder (requires mamba-ssm package)."""

    def __init__(self, d_model: int, num_layers: int):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm not installed. Install: pip install mamba-ssm\n"
                "Or use --temporal_model transformer"
            )
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  (B, T, D)"""
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))  # residual
        return x


def build_temporal_model(cfg: ModelConfig, d_input: int) -> Tuple[nn.Module, int]:
    """Build temporal model, project input to temporal_dim, return (model, out_dim)."""
    proj = nn.Linear(d_input, cfg.temporal_dim)

    if cfg.temporal_model == "mamba":
        temporal = MambaTemporal(cfg.temporal_dim, cfg.temporal_layers)
    elif cfg.temporal_model == "lstm":
        temporal = LSTMTemporal(cfg.temporal_dim, cfg.temporal_dim, cfg.temporal_layers, cfg.temporal_dropout)
    else:  # transformer (default, always available)
        temporal = TransformerTemporal(
            cfg.temporal_dim, cfg.temporal_heads, cfg.temporal_layers, cfg.temporal_dropout
        )

    return nn.Sequential(proj, temporal), cfg.temporal_dim


# ─── Output Heads ─────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class PulseRegressionHead(nn.Module):
    """Predicts a normalized pulse waveform from temporal features."""

    def __init__(self, in_dim: int, out_len: int, dropout: float):
        super().__init__()
        self.out_len = out_len
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_len),
            nn.Tanh(),                   # waveform in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D_temporal)  →  (B, T_pulse)"""
        return self.net(x)


class BlinkSequenceHead(nn.Module):
    """Predicts per-frame eye-closed probability from temporal features."""

    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  (B, T) probabilities"""
        return self.net(x).squeeze(-1)   # (B, T)


# ─── Motion Model ───────────────────────────────

class MotionEncoder(nn.Module):
    """
    Lightweight CNN that processes frame differences I(t)-I(t-1).
    Captures temporal inconsistencies in blood flow / color changes
    that deepfakes fail to replicate.

    Input: (B, T, 3, H, W) frame differences
    Output: (B, motion_dim) motion summary feature
    """

    def __init__(self, motion_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 224→112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 112→56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                     # 56→1
            nn.Flatten(),                                 # (B*T, 32)
        )
        self.temporal_pool = nn.Sequential(
            nn.Linear(32, motion_dim),
            nn.ReLU(),
        )

    def forward(self, frame_diffs: torch.Tensor) -> torch.Tensor:
        """frame_diffs: (B, T, 3, H, W) → (B, motion_dim)"""
        B, T, C, H, W = frame_diffs.shape
        x = frame_diffs.reshape(B * T, C, H, W)
        # Process in chunks to save memory
        feats = []
        for i in range(0, B * T, 8):
            feats.append(self.conv(x[i:i+8]))
        x = torch.cat(feats, dim=0)             # (B*T, 32)
        x = x.view(B, T, -1)                    # (B, T, 32)
        x = x.mean(dim=1)                       # (B, 32) — temporal mean pool
        return self.temporal_pool(x)             # (B, motion_dim)


# ─── PhysioNet Main Model ─────────────────────────────────────────────────────

class PhysioNet(nn.Module):
    """
    PhysioNet: Physiological deepfake detection model.

    Forward input:
        frames:      (B, T, 3, H, W)   — video clip frames, normalized [0,1]
        rppg_feat:   (B, rppg_dim)     — explicit rPPG FFT spectrum feature (optional)
        blink_feat:  (B, blink_dim)    — explicit blink stats feature (optional)

    Forward output:
        dict with keys:
          'logit'         (B,)         — raw classification logit (sigmoid → prob)
          'pulse_pred'    (B, T)       — predicted pulse waveform (if use_pulse_head)
          'blink_pred'    (B, T)       — predicted per-frame blink prob (if use_blink_head)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Per-frame backbone
        self.frame_encoder = FrameEncoder(cfg.backbone, cfg.backbone_pretrained,
                                          local_weights=cfg.backbone_local_weights)
        backbone_dim = self.frame_encoder.out_dim

        # 2. Temporal model (with projection from backbone_dim → temporal_dim)
        # Special case: "direct" pool skips temporal_proj entirely — classifies from
        # mean-pooled backbone features directly. Avoids the 1792→temporal_dim bottleneck
        # that loses discriminative information when temporal_dim << backbone_dim.
        self._direct_pool = (cfg.temporal_pool == "mean" and cfg.temporal_dim == 0)
        if self._direct_pool:
            temporal_out_dim = backbone_dim
            # No temporal_proj, no temporal model needed
        else:
            self.temporal_proj = nn.Linear(backbone_dim, cfg.temporal_dim)
            if cfg.temporal_model == "mamba" and MAMBA_AVAILABLE:
                self.temporal = MambaTemporal(cfg.temporal_dim, cfg.temporal_layers)
            elif cfg.temporal_model == "lstm":
                self.temporal = LSTMTemporal(cfg.temporal_dim, cfg.temporal_dim, cfg.temporal_layers, cfg.temporal_dropout)
            else:
                self.temporal = TransformerTemporal(
                    cfg.temporal_dim, cfg.temporal_heads, cfg.temporal_layers, cfg.temporal_dropout
                )
            temporal_out_dim = cfg.temporal_dim

        # 2b. Motion model (frame-difference branch)
        if cfg.use_motion_model:
            self.motion_encoder = MotionEncoder(cfg.motion_dim)

        # 3. Fusion: temporal CLS token + (optionally) explicit features + motion
        fusion_input_dim = temporal_out_dim
        if cfg.use_physio_fusion:
            physio_dim = cfg.rppg_feature_dim + cfg.blink_feature_dim
            # BatchNorm normalizes physio features (SNR, PCC, PSD have wildly different scales)
            self.physio_bn = nn.BatchNorm1d(physio_dim)
            fusion_input_dim += physio_dim
        if cfg.use_motion_model:
            fusion_input_dim += cfg.motion_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # 4. Output heads
        self.cls_head = ClassificationHead(cfg.fusion_dim, cfg.dropout)
        self._temporal_out_dim = temporal_out_dim

        if cfg.use_pulse_head:
            self.pulse_head = PulseRegressionHead(temporal_out_dim, cfg.clip_len, cfg.dropout)

        if cfg.use_blink_head:
            self.blink_head = BlinkSequenceHead(temporal_out_dim, cfg.dropout)

        # Learnable gate: sigmoid(temporal_gate) blends mean-pool bypass with transformer output.
        # sigmoid(-6) ≈ 0.0025 → starts as near-pure mean-pool; gate opens as transformer converges.
        self.temporal_gate = nn.Parameter(torch.full((1,), -6.0))

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "frame_encoder" not in name:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone weights for phased training."""
        for p in self.frame_encoder.parameters():
            p.requires_grad = not freeze

    def forward(
        self,
        frames: torch.Tensor,
        rppg_feat: Optional[torch.Tensor] = None,
        blink_feat: Optional[torch.Tensor] = None,
        frame_diffs: Optional[torch.Tensor] = None,
    ) -> dict:
        B, T = frames.shape[:2]
        device = frames.device

        # 1. Per-frame encoding
        frame_feats = self.frame_encoder(frames)           # (B, T, backbone_dim)

        # 2. Temporal modeling
        if self._direct_pool:
            # Direct mean pool in backbone feature space — no projection bottleneck
            temporal_out = frame_feats                     # (B, T, backbone_dim)
            cls_token = frame_feats.mean(dim=1)            # (B, backbone_dim)
        else:
            temporal_in = self.temporal_proj(frame_feats)  # (B, T, temporal_dim)
            if self.cfg.temporal_pool == "mean":
                temporal_out = temporal_in
                cls_token = temporal_in.mean(dim=1)        # (B, temporal_dim)
            else:
                # Gated temporal encoder:
                # sigmoid(-6)≈0.003 at init → near-pure mean-pool; opens as transformer converges
                mean_pool = temporal_in.mean(dim=1, keepdim=True).expand_as(temporal_in)
                gate = torch.sigmoid(self.temporal_gate)
                transformer_out = self.temporal(temporal_in)
                temporal_out = (1 - gate) * mean_pool + gate * transformer_out
                cls_token = temporal_out.mean(dim=1)       # (B, temporal_dim)

        # 3. Fusion
        parts = [cls_token]
        if self.cfg.use_physio_fusion:
            if rppg_feat is None:
                rppg_feat = torch.zeros(B, self.cfg.rppg_feature_dim, device=device)
            if blink_feat is None:
                blink_feat = torch.zeros(B, self.cfg.blink_feature_dim, device=device)
            physio = torch.cat([rppg_feat, blink_feat], dim=-1)
            physio = self.physio_bn(physio)  # normalize heterogeneous feature scales
            parts.append(physio)
        if self.cfg.use_motion_model and hasattr(self, 'motion_encoder'):
            if frame_diffs is not None:
                motion_feat = self.motion_encoder(frame_diffs)
            else:
                motion_feat = torch.zeros(B, self.cfg.motion_dim, device=device)
            parts.append(motion_feat)
        fused = torch.cat(parts, dim=-1)

        fused = self.fusion(fused)                         # (B, fusion_dim)

        # 5. Heads
        outputs = {"logit": self.cls_head(fused)}

        if self.cfg.use_pulse_head and hasattr(self, "pulse_head"):
            outputs["pulse_pred"] = self.pulse_head(cls_token)           # (B, T_pulse)

        if self.cfg.use_blink_head and hasattr(self, "blink_head"):
            outputs["blink_pred"] = self.blink_head(temporal_out)        # (B, T)

        return outputs

    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    # Test 1: Original config
    cfg = ModelConfig(
        backbone="efficientnet_b4",
        backbone_pretrained=False,
        temporal_model="transformer",
        clip_len=16,
        img_size=224,
    )
    model = PhysioNet(cfg)
    params = model.get_num_params()
    print(f"PhysioNet params: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")

    B, T = 2, 16
    frames = torch.randn(B, T, 3, 224, 224)
    rppg_feat = torch.randn(B, cfg.rppg_feature_dim)
    blink_feat = torch.randn(B, cfg.blink_feature_dim)
    with torch.no_grad():
        out = model(frames, rppg_feat, blink_feat)
    print(f"logit: {out['logit'].shape}")
    print("Test 1 (original) PASSED")

    # Test 2: V2 rPPG (12-d sync) + motion model
    cfg2 = ModelConfig(
        backbone="efficientnet_b4",
        backbone_pretrained=False,
        temporal_model="mean",
        temporal_dim=0,
        clip_len=16,
        img_size=224,
        rppg_feature_dim=12,
        use_motion_model=True,
        use_pulse_head=False,
        use_blink_head=False,
    )
    model2 = PhysioNet(cfg2)
    rppg2 = torch.randn(B, 12)
    blink2 = torch.randn(B, 16)
    diffs = torch.randn(B, T, 3, 224, 224)
    with torch.no_grad():
        out2 = model2(frames, rppg2, blink2, diffs)
    print(f"logit: {out2['logit'].shape}")
    print("Test 2 (v2 rPPG + motion) PASSED")
