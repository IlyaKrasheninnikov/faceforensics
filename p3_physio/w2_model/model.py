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


# ─── Backbone ─────────────────────────────────────────────────────────────────

class FrameEncoder(nn.Module):
    """
    Per-frame feature extractor using EfficientNet-B4.
    Processes (B*T, C, H, W) and returns (B, T, D) frame features.
    """

    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        if TIMM_AVAILABLE:
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,       # remove classifier head
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

    def forward(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, T, D)

        Processes frames in chunks to avoid OOM on long clips.
        Uses gradient checkpointing to save memory when training.
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
        self.frame_encoder = FrameEncoder(cfg.backbone, cfg.backbone_pretrained)
        backbone_dim = self.frame_encoder.out_dim

        # 2. Temporal model (with projection from backbone_dim → temporal_dim)
        self.temporal_proj = nn.Linear(backbone_dim, cfg.temporal_dim)

        if cfg.temporal_model == "mamba" and MAMBA_AVAILABLE:
            self.temporal = MambaTemporal(cfg.temporal_dim, cfg.temporal_layers)
        elif cfg.temporal_model == "lstm":
            self.temporal = LSTMTemporal(cfg.temporal_dim, cfg.temporal_dim, cfg.temporal_layers, cfg.temporal_dropout)
        else:
            self.temporal = TransformerTemporal(
                cfg.temporal_dim, cfg.temporal_heads, cfg.temporal_layers, cfg.temporal_dropout
            )

        # 3. Fusion: temporal CLS token + explicit features
        fusion_input_dim = cfg.temporal_dim + cfg.rppg_feature_dim + cfg.blink_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # 4. Output heads
        self.cls_head = ClassificationHead(cfg.fusion_dim, cfg.dropout)

        if cfg.use_pulse_head:
            self.pulse_head = PulseRegressionHead(cfg.temporal_dim, cfg.clip_len, cfg.dropout)

        if cfg.use_blink_head:
            self.blink_head = BlinkSequenceHead(cfg.temporal_dim, cfg.dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.temporal_proj, self.fusion]:
            if isinstance(m, nn.Linear):
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
    ) -> dict:
        B, T = frames.shape[:2]
        device = frames.device

        # 1. Per-frame encoding
        frame_feats = self.frame_encoder(frames)           # (B, T, backbone_dim)

        # 2. Temporal modeling
        temporal_in = self.temporal_proj(frame_feats)      # (B, T, temporal_dim)
        temporal_out = self.temporal(temporal_in)          # (B, T, temporal_dim)

        # CLS token = mean pooling over time
        cls_token = temporal_out.mean(dim=1)               # (B, temporal_dim)

        # 3. Handle explicit features (zero-pad if not provided)
        if rppg_feat is None:
            rppg_feat = torch.zeros(B, self.cfg.rppg_feature_dim, device=device)
        if blink_feat is None:
            blink_feat = torch.zeros(B, self.cfg.blink_feature_dim, device=device)

        # 4. Fusion
        fused = torch.cat([cls_token, rppg_feat, blink_feat], dim=-1)   # (B, fusion_input_dim)
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

    cfg = ModelConfig(
        backbone="efficientnet_b4",
        backbone_pretrained=False,   # don't download weights during test
        temporal_model="transformer",
        clip_len=16,
        img_size=224,
    )

    model = PhysioNet(cfg)
    params = model.get_num_params()
    print(f"PhysioNet params: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")

    # Dummy forward pass
    B, T = 2, 16
    frames = torch.randn(B, T, 3, 224, 224)
    rppg_feat = torch.randn(B, cfg.rppg_feature_dim)
    blink_feat = torch.randn(B, cfg.blink_feature_dim)

    with torch.no_grad():
        out = model(frames, rppg_feat, blink_feat)

    print(f"logit shape:       {out['logit'].shape}")
    print(f"pulse_pred shape:  {out['pulse_pred'].shape}")
    print(f"blink_pred shape:  {out['blink_pred'].shape}")
    print("Smoke test PASSED")
