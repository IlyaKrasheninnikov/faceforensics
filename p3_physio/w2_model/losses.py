"""
W2: Multi-task loss functions for P3 PhysioNet.

Losses:
  1. ClassificationLoss  — binary cross-entropy (real/fake)
  2. SpectralEntropyLoss — penalize low pulse periodicity in real, enforce low entropy
  3. ContrastivePulseLoss — real videos should have low spectral entropy (clear pulse peak)
                            fake videos should have high entropy (no clear pulse)
  4. BlinkAuxLoss        — binary cross-entropy on blink sequence prediction
  5. PhysioMultiTaskLoss — combines all above with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Classification Loss ───────────────────────────────────────────────────

class ClassificationLoss(nn.Module):
    """
    Weighted BCE loss for class imbalance.
    pos_weight > 1 to penalize missed fakes more (higher recall priority).
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        logit: (B,) raw logits
        label: (B,) float {0=real, 1=fake}
        """
        weight = torch.where(label == 1,
                             torch.full_like(label, self.pos_weight),
                             torch.ones_like(label))
        return F.binary_cross_entropy_with_logits(logit, label, weight=weight)


# ─── 2. Spectral Entropy Loss ─────────────────────────────────────────────────

class SpectralEntropyLoss(nn.Module):
    """
    For REAL videos: penalize high spectral entropy in the pulse waveform prediction.
    A real pulse should have a clear dominant frequency → low entropy.

    For FAKE videos: no supervision on pulse (or optionally penalize low entropy).
    By default this loss is only applied to real samples.

    entropy = -sum(p * log(p))  where p is normalized PSD of predicted waveform
    """

    def __init__(self, freq_lo: float = 0.7, freq_hi: float = 3.5, fps: float = 15.0):
        super().__init__()
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.fps = fps

    def compute_spectral_entropy(self, pulse: torch.Tensor) -> torch.Tensor:
        """
        pulse: (B, T) predicted pulse waveform
        returns: (B,) spectral entropy per sample
        """
        B, T = pulse.shape
        # FFT does not support fp16 — upcast
        pulse = pulse.float()
        # FFT magnitude squared = PSD estimate
        fft = torch.fft.rfft(pulse, dim=-1)
        psd = torch.abs(fft) ** 2 + 1e-10                     # (B, T//2+1)

        # Focus on physiological frequency band
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(pulse.device)
        mask = (freqs >= self.freq_lo) & (freqs <= self.freq_hi)

        psd_band = psd[:, mask]                                # (B, N_band)
        psd_norm = psd_band / (psd_band.sum(dim=-1, keepdim=True) + 1e-10)

        # Shannon entropy
        entropy = -(psd_norm * torch.log(psd_norm + 1e-10)).sum(dim=-1)  # (B,)
        return entropy

    def forward(
        self,
        pulse_pred: torch.Tensor,
        label: torch.Tensor,
        apply_to_fakes: bool = False,
    ) -> torch.Tensor:
        """
        pulse_pred: (B, T)
        label: (B,) {0=real, 1=fake}
        """
        entropy = self.compute_spectral_entropy(pulse_pred)    # (B,)

        real_mask = (label == 0).float()
        loss_real = (entropy * real_mask).sum() / (real_mask.sum() + 1e-8)

        if apply_to_fakes:
            # Fakes should have HIGH entropy (no pulse) — penalize low entropy
            fake_mask = (label == 1).float()
            loss_fake = ((-entropy + entropy.detach().max()) * fake_mask).sum() / (fake_mask.sum() + 1e-8)
        else:
            loss_fake = torch.tensor(0.0, device=pulse_pred.device)

        return loss_real + loss_fake


# ─── 3. Contrastive Pulse Loss ────────────────────────────────────────────────

class ContrastivePulseLoss(nn.Module):
    """
    Margin-based contrastive: real pulse entropy should be margin lower than fake.
    L = max(0, entropy_real - entropy_fake + margin)
    Applied pairwise within batch.
    """

    def __init__(self, margin: float = 0.5, fps: float = 15.0):
        super().__init__()
        self.margin = margin
        self.spectral_entropy = SpectralEntropyLoss(fps=fps)

    def forward(self, pulse_pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        entropy = self.spectral_entropy.compute_spectral_entropy(pulse_pred)

        real_mask = (label == 0)
        fake_mask = (label == 1)

        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return torch.tensor(0.0, device=pulse_pred.device)

        mean_real_entropy = entropy[real_mask].mean()
        mean_fake_entropy = entropy[fake_mask].mean()

        # Real should be lower entropy (more periodic) than fake
        loss = F.relu(mean_real_entropy - mean_fake_entropy + self.margin)
        return loss


# ─── 4. Blink Auxiliary Loss ─────────────────────────────────────────────────

class BlinkAuxLoss(nn.Module):
    """
    Per-frame binary cross-entropy for blink sequence prediction.
    Applied only to real videos where we have ground-truth blink labels.
    For fake videos: optionally apply — but fakes may have real-ish blink patterns,
    so we typically only supervise reals.
    """

    def forward(
        self,
        blink_pred: torch.Tensor,
        blink_target: torch.Tensor,
        label: torch.Tensor,
        real_only: bool = True,
    ) -> torch.Tensor:
        """
        blink_pred:   (B, T) predicted per-frame closed-eye logit
        blink_target: (B, T) ground-truth {0=open, 1=closed}
        label:        (B,) {0=real, 1=fake}
        """
        if real_only:
            real_mask = (label == 0)
            if real_mask.sum() == 0:
                return torch.tensor(0.0, device=blink_pred.device)
            blink_pred = blink_pred[real_mask]
            blink_target = blink_target[real_mask]

        return F.binary_cross_entropy_with_logits(blink_pred, blink_target.float())


# ─── 5. Combined Multi-task Loss ──────────────────────────────────────────────

class PhysioMultiTaskLoss(nn.Module):
    """
    Combined loss for PhysioNet training.

    Weights (all configurable):
        w_class:        classification BCE
        w_pulse:        spectral entropy loss on pulse head (real only)
        w_blink:        blink sequence BCE (real only)
        w_contrastive:  contrastive pulse margin loss (real vs fake pairs)
    """

    def __init__(
        self,
        w_class: float = 1.0,
        w_pulse: float = 0.4,
        w_blink: float = 0.3,
        w_contrastive: float = 0.1,
        pos_weight: float = 1.0,
        fps: float = 15.0,
    ):
        super().__init__()
        self.w_class = w_class
        self.w_pulse = w_pulse
        self.w_blink = w_blink
        self.w_contrastive = w_contrastive

        self.cls_loss = ClassificationLoss(pos_weight)
        self.pulse_loss = SpectralEntropyLoss(fps=fps)
        self.contrastive_loss = ContrastivePulseLoss(fps=fps)
        self.blink_loss = BlinkAuxLoss()

    def forward(
        self,
        model_outputs: dict,
        label: torch.Tensor,
        blink_target: torch.Tensor = None,
    ) -> dict:
        """
        model_outputs: dict from PhysioNet.forward()
            - 'logit':       (B,)
            - 'pulse_pred':  (B, T)  [optional]
            - 'blink_pred':  (B, T)  [optional]
        label:        (B,) float {0=real, 1=fake}
        blink_target: (B, T) float ground-truth blink per frame [optional]

        Returns dict: {'total', 'cls', 'pulse', 'blink', 'contrastive'}
        """
        losses = {}

        # 1. Classification
        losses["cls"] = self.cls_loss(model_outputs["logit"], label)

        # 2. Pulse spectral entropy (real videos only)
        if "pulse_pred" in model_outputs and self.w_pulse > 0:
            losses["pulse"] = self.pulse_loss(model_outputs["pulse_pred"], label)
        else:
            losses["pulse"] = torch.tensor(0.0, device=label.device)

        # 3. Blink auxiliary
        if "blink_pred" in model_outputs and blink_target is not None and self.w_blink > 0:
            losses["blink"] = self.blink_loss(model_outputs["blink_pred"], blink_target, label)
        else:
            losses["blink"] = torch.tensor(0.0, device=label.device)

        # 4. Contrastive pulse
        if "pulse_pred" in model_outputs and self.w_contrastive > 0:
            losses["contrastive"] = self.contrastive_loss(model_outputs["pulse_pred"], label)
        else:
            losses["contrastive"] = torch.tensor(0.0, device=label.device)

        # Weighted total
        losses["total"] = (
            self.w_class * losses["cls"]
            + self.w_pulse * losses["pulse"]
            + self.w_blink * losses["blink"]
            + self.w_contrastive * losses["contrastive"]
        )

        return losses


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 4, 64
    label = torch.tensor([0.0, 0.0, 1.0, 1.0])   # 2 real, 2 fake
    logit = torch.randn(B)
    pulse = torch.randn(B, T)
    blink = torch.randn(B, T)
    blink_gt = (torch.rand(B, T) > 0.9).float()    # ~10% closed frames

    criterion = PhysioMultiTaskLoss()
    losses = criterion(
        {"logit": logit, "pulse_pred": pulse, "blink_pred": blink},
        label,
        blink_target=blink_gt,
    )

    print("Loss components:")
    for k, v in losses.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k:<15s}: {val:.4f}")
