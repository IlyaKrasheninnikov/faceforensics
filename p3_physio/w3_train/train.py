"""
W3: Main PhysioNet detection training script (Phase 2 — real + fake).

Loads pretrain checkpoint (from w2), fine-tunes with full multi-task loss
on real + fake data.

Usage:
    python w3_train/train.py \
        --ff_root /data/FF++ \
        --celebdf_root /data/CelebDF-v2 \
        --pretrain_ckpt ./checkpoints/pretrain_best.pt \
        --run_name w3_initial_train \
        --epochs 30 \
        --batch_size 8 \
        --fp16

    # Without pretrain checkpoint (train from scratch):
    python w3_train/train.py \
        --ff_root /data/FF++ \
        --run_name w3_scratch
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.trackio_init import ExperimentLogger
from w2_model.model import PhysioNet, ModelConfig
from w2_model.losses import PhysioMultiTaskLoss
from w2_model.dataset import build_dataloaders
from w3_train.eval import evaluate


def warmup_cache(dataset, num_workers: int = 2):
    """Pre-extract and cache all rPPG/blink features before training starts."""
    from torch.utils.data import DataLoader
    print(f"\n── Pre-caching physiological features for {len(dataset)} videos ──")
    print("   (MediaPipe runs once here; subsequent epochs load from disk cache)")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    for i, _ in enumerate(tqdm(loader, desc="Caching features", leave=False)):
        pass
    print(f"   ✓ Cached {i+1} videos\n")


def train(args):
    # ─── Setup ────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    logger = ExperimentLogger(
        project="p3_physio_deepfake",
        run_name=args.run_name or "w3_train",
        config=vars(args),
        local_log_dir=args.log_dir,
    )

    # ─── Data ─────────────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl = build_dataloaders(
        ff_root=args.ff_root,
        celebdf_root=args.celebdf_root,
        cache_dir=args.cache_dir,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment_train=True,
    )

    # Pre-cache all features (rPPG + blink via MediaPipe) before GPU training
    if not args.skip_cache:
        warmup_cache(train_dl.dataset, num_workers=args.num_workers)
    else:
        print("Skipping feature cache warmup (--skip_cache)")

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

    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded pretrain checkpoint: {args.pretrain_ckpt}")
    else:
        print("Training from scratch (no pretrain checkpoint)")

    # Freeze backbone for first N epochs
    model.freeze_backbone(freeze=True)
    print("Backbone frozen for first 2 epochs")

    params = model.get_num_params()
    print(f"Model: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")

    # ─── Loss ─────────────────────────────────────────────────────────────────
    criterion = PhysioMultiTaskLoss(
        w_class=args.w_class,
        w_pulse=args.w_pulse,
        w_blink=args.w_blink,
        w_contrastive=args.w_contrastive,
        pos_weight=args.pos_weight,
    )

    # ─── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.frame_encoder.parameters(), "lr": args.lr_backbone},
        {"params": model.temporal_proj.parameters(), "lr": args.lr_head},
        {"params": model.temporal.parameters(), "lr": args.lr_temporal},
        {"params": model.cls_head.parameters(), "lr": args.lr_head},
        {"params": model.fusion.parameters(), "lr": args.lr_head},
        {"params": model.pulse_head.parameters(), "lr": args.lr_head},
        {"params": model.blink_head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps = len(train_dl) * args.epochs
    warmup_steps = len(train_dl) * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_head, total_steps=total_steps,
        pct_start=warmup_steps / total_steps, anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0
    global_step = 0
    start_epoch = 1

    # Resume from latest checkpoint if --resume
    if args.resume and (out_dir / "latest.pt").exists():
        resume_ckpt = torch.load(out_dir / "latest.pt", map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        best_val_auc = resume_ckpt.get("val_auc", 0.0)
        # Fast-forward scheduler
        steps_to_skip = len(train_dl) * (start_epoch - 1)
        for _ in range(steps_to_skip):
            scheduler.step()
        global_step = steps_to_skip
        print(f"Resumed from epoch {start_epoch - 1}, best_val_auc={best_val_auc:.4f}")

    # ─── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # Unfreeze backbone after warmup
        if epoch == 3:
            model.freeze_backbone(freeze=False)
            print(f"Epoch {epoch}: Backbone unfrozen")

        model.train()
        epoch_losses = {k: [] for k in ["total", "cls", "pulse", "blink", "contrastive"]}

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            frames = batch["frames"].to(device)
            rppg_feat = batch["rppg_feat"].to(device)
            blink_feat = batch["blink_feat"].to(device)
            blink_labels = batch["blink_labels"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(frames, rppg_feat, blink_feat)
                losses = criterion(outputs, label, blink_target=blink_labels)

            loss_tensor = losses["total"]

            if scaler:
                scaler.scale(loss_tensor).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            scheduler.step()

            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k].append(v.item() if torch.is_tensor(v) else float(v))

            global_step += 1
            if global_step % args.log_interval == 0:
                logger.log({
                    "step/loss_total": losses["total"].item() if torch.is_tensor(losses["total"]) else losses["total"],
                    "step/loss_cls": losses["cls"].item() if torch.is_tensor(losses["cls"]) else losses["cls"],
                    "step/loss_pulse": losses["pulse"].item() if torch.is_tensor(losses["pulse"]) else losses["pulse"],
                    "step/loss_blink": losses["blink"].item() if torch.is_tensor(losses["blink"]) else losses["blink"],
                }, step=global_step)

        # ─── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_dl, device, scaler, split="val")

        metrics = {
            "epoch": epoch,
            **{f"train/{k}": np.mean(v) for k, v in epoch_losses.items() if v},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
        }
        logger.log(metrics, step=epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"loss={metrics.get('train/total', 0):.3f} | "
            f"val_auc={val_metrics.get('auc', 0):.4f} "
            f"val_eer={val_metrics.get('eer', 0):.4f} "
            f"val_ece={val_metrics.get('ece', 0):.4f}"
        )

        # Save best by AUC
        val_auc = val_metrics.get("auc", 0.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "config": cfg,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ New best AUC={val_auc:.4f} → {ckpt_path}")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc,
        }, out_dir / "latest.pt")

    # ─── Final test evaluation ─────────────────────────────────────────────────
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_dl, device, scaler, split="test")

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)

    logger.log_summary({"best_val_auc": best_val_auc, **{f"test/{k}": v for k, v in test_metrics.items()}})
    logger.finish()


def parse_args():
    p = argparse.ArgumentParser(description="P3 PhysioNet Phase 2: Full detection training")
    p.add_argument("--ff_root", default=None)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_root", default=None)
    p.add_argument("--pretrain_ckpt", default=None)
    p.add_argument("--resume", action="store_true", help="Resume from latest.pt checkpoint")
    p.add_argument("--skip_cache", action="store_true", help="Skip feature cache warmup (use if already cached)")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--cache_dir", default="./logs/signal_cache")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--run_name", default=None)

    # Model
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--temporal_model", default="transformer", choices=["transformer", "lstm", "mamba"])
    p.add_argument("--clip_len", type=int, default=64)
    p.add_argument("--img_size", type=int, default=224)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_temporal", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--log_interval", type=int, default=20)

    # Loss weights
    p.add_argument("--w_class", type=float, default=1.0)
    p.add_argument("--w_pulse", type=float, default=0.4)
    p.add_argument("--w_blink", type=float, default=0.3)
    p.add_argument("--w_contrastive", type=float, default=0.1)
    p.add_argument("--pos_weight", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
