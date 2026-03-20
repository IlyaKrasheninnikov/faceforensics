"""
W2: Phase 1 — Pretrain PhysioNet on REAL videos only.

Goal: Teach the model to recognize physiological signals (pulse, blink) before
introducing fake samples. This gives the model a strong prior for "what real looks like".

Trains with:
  - Spectral entropy loss on pulse head (minimize entropy = learn periodic pulse)
  - Blink BCE loss on blink head (learn to predict eye closure)
  - NO classification loss (no fake samples yet)

Usage:
    # On Colab/Kaggle:
    python w2_model/train_pretrain.py \
        --ff_root /data/FF++ \
        --out_dir ./checkpoints \
        --epochs 10 \
        --batch_size 4 \
        --fp16

    # With Trackio logging:
    python w2_model/train_pretrain.py \
        --ff_root /data/FF++ \
        --run_name pretrain_v1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.trackio_init import ExperimentLogger
from w2_model.model import PhysioNet, ModelConfig
from w2_model.losses import SpectralEntropyLoss, BlinkAuxLoss
from w2_model.dataset import PhysioDeepfakeDataset, build_ff_plus_plus_list


def build_real_only_dataset(ff_root: str, cache_dir: str, clip_len: int, img_size: int) -> Dataset:
    """Return dataset containing only real (original) FF++ videos."""
    from pathlib import Path
    # Try multiple known layouts
    candidates = [
        Path(ff_root) / "original",                                    # flat: original/*.mp4
        Path(ff_root) / "original" / "c23" / "videos",                # nested: original/c23/videos/*.mp4
        Path(ff_root) / "original_sequences" / "youtube" / "c23" / "videos",
    ]
    real_dir = None
    for c in candidates:
        if c.exists() and list(c.glob("*.mp4")):
            real_dir = c
            break
    if real_dir is None:
        raise FileNotFoundError(f"FF++ original videos not found at {ff_root}")

    video_files = sorted(list(real_dir.glob("*.mp4")))
    print(f"Found {len(video_files)} real videos for pretraining")

    return PhysioDeepfakeDataset(
        video_paths=[str(v) for v in video_files],
        labels=[0] * len(video_files),
        clip_len=clip_len,
        img_size=img_size,
        augment=False,   # no augmentation during pretrain
        cache_dir=cache_dir,
    )


def compute_hr_mae(pulse_pred: torch.Tensor, fps: float = 15.0,
                   freq_lo: float = 0.7, freq_hi: float = 3.5) -> float:
    """
    Estimate BPM from predicted pulse waveform via FFT peak.
    Returns mean absolute error against a reference (placeholder: 75 bpm average).
    In real usage, compare against rPPG algorithm's BPM estimate.
    """
    pulse_pred = pulse_pred.float()  # rfft does not support fp16
    T = pulse_pred.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps)
    fft_mag = torch.abs(torch.fft.rfft(pulse_pred, dim=-1))

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if mask.sum() == 0:
        return 75.0

    peak_freqs = []
    for i in range(pulse_pred.shape[0]):
        band = fft_mag[i, mask]
        peak_freq = freqs[mask][band.argmax()]
        peak_freqs.append(peak_freq.item() * 60.0)

    return float(np.std(peak_freqs))  # BPM variance as proxy metric (lower = more consistent)


def warmup_cache(dataset, num_workers: int = 2):
    """Pre-extract and cache all rPPG/blink features so MediaPipe runs only once."""
    from torch.utils.data import DataLoader
    print(f"\n── Pre-caching physiological features for {len(dataset)} videos ──")
    print("   (MediaPipe runs once here; subsequent epochs load from disk cache)")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    for i, _ in enumerate(tqdm(loader, desc="Caching features", leave=False)):
        pass
    print(f"   ✓ Cached {i+1} videos\n")


def pretrain(args):
    # ─── Setup ────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    logger = ExperimentLogger(
        project="p3_physio_deepfake",
        run_name=args.run_name or "pretrain",
        config=vars(args),
        local_log_dir=args.log_dir,
    )

    # ─── Data ─────────────────────────────────────────────────────────────────
    dataset = build_real_only_dataset(
        args.ff_root, args.cache_dir, args.clip_len, args.img_size
    )

    # Pre-cache all features (MediaPipe CPU extraction) before training
    warmup_cache(dataset, num_workers=args.num_workers)
    n_val = max(10, int(len(dataset) * 0.1))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ─── Model ────────────────────────────────────────────────────────────────
    cfg = ModelConfig(
        backbone=args.backbone,
        backbone_pretrained=True,
        temporal_model=args.temporal_model,
        clip_len=args.clip_len,
        img_size=args.img_size,
        use_pulse_head=True,
        use_blink_head=True,
    )
    model = PhysioNet(cfg).to(device)
    params = model.get_num_params()
    print(f"Model: {params['total']/1e6:.1f}M params ({params['trainable']/1e6:.1f}M trainable)")

    # ─── Losses ───────────────────────────────────────────────────────────────
    pulse_loss_fn = SpectralEntropyLoss(fps=args.fps)
    blink_loss_fn = BlinkAuxLoss()

    # ─── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.frame_encoder.parameters(), "lr": args.lr_backbone},
        {"params": model.temporal_proj.parameters(), "lr": args.lr_head},
        {"params": model.temporal.parameters(), "lr": args.lr_head},
        {"params": model.pulse_head.parameters(), "lr": args.lr_head},
        {"params": model.blink_head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ─── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = {"pulse": [], "blink": [], "total": []}

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            frames = batch["frames"].to(device)
            rppg_feat = batch["rppg_feat"].to(device)
            blink_feat = batch["blink_feat"].to(device)
            blink_labels = batch["blink_labels"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(frames, rppg_feat, blink_feat)

                # Only real videos in this phase — but use label=0 for all
                all_real = torch.zeros(label.shape[0], device=device)

                loss_pulse = pulse_loss_fn(outputs["pulse_pred"], all_real)
                loss_blink = blink_loss_fn(outputs["blink_pred"], blink_labels, all_real)
                loss = loss_pulse + 0.5 * loss_blink

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_losses["pulse"].append(loss_pulse.item())
            train_losses["blink"].append(loss_blink.item())
            train_losses["total"].append(loss.item())

        # Validation
        model.eval()
        val_losses = {"pulse": [], "blink": [], "bpm_var": []}

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                frames = batch["frames"].to(device)
                rppg_feat = batch["rppg_feat"].to(device)
                blink_feat = batch["blink_feat"].to(device)
                blink_labels = batch["blink_labels"].to(device)

                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(frames, rppg_feat, blink_feat)
                    all_real = torch.zeros(frames.shape[0], device=device)
                    loss_p = pulse_loss_fn(outputs["pulse_pred"], all_real)
                    loss_b = blink_loss_fn(outputs["blink_pred"], blink_labels, all_real)

                val_losses["pulse"].append(loss_p.item())
                val_losses["blink"].append(loss_b.item())
                bpm_v = compute_hr_mae(outputs["pulse_pred"].cpu(), args.fps)
                val_losses["bpm_var"].append(bpm_v)

        scheduler.step()

        # Log metrics
        metrics = {
            "epoch": epoch,
            "train/loss_pulse": np.mean(train_losses["pulse"]),
            "train/loss_blink": np.mean(train_losses["blink"]),
            "train/loss_total": np.mean(train_losses["total"]),
            "val/loss_pulse": np.mean(val_losses["pulse"]),
            "val/loss_blink": np.mean(val_losses["blink"]),
            "val/bpm_variance": np.mean(val_losses["bpm_var"]),
            "lr": scheduler.get_last_lr()[0],
        }
        logger.log(metrics, step=epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"train_pulse={metrics['train/loss_pulse']:.3f} "
            f"train_blink={metrics['train/loss_blink']:.3f} | "
            f"val_pulse={metrics['val/loss_pulse']:.3f} "
            f"bpm_var={metrics['val/bpm_variance']:.1f}"
        )

        # Save best
        val_loss = metrics["val/loss_pulse"] + metrics["val/loss_blink"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "pretrain_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    logger.log_summary({
        "best_val_loss": best_val_loss,
        "final_epoch": args.epochs,
    })
    logger.finish()

    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {out_dir / 'pretrain_best.pt'}")
    return out_dir / "pretrain_best.pt"


def parse_args():
    p = argparse.ArgumentParser(description="P3 PhysioNet Phase 1: Pretrain on real videos")
    p.add_argument("--ff_root", required=True, help="Path to FaceForensics++ root")
    p.add_argument("--out_dir", default="./checkpoints", help="Where to save checkpoints")
    p.add_argument("--cache_dir", default="./logs/signal_cache")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--run_name", default=None)

    # Model
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--temporal_model", default="transformer", choices=["transformer", "lstm", "mamba"])
    p.add_argument("--clip_len", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--fps", type=float, default=15.0)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pretrain(args)
