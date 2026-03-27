"""
W3: PhysioNet training script — robust version for Kaggle T4.

Single-GPU only (no DataParallel — it causes hangs with small batch sizes).
Designed to complete 5 epochs within Kaggle's 12h limit.

Usage:
    python w3_train/train.py \
        --ff_root /data/FF++ \
        --run_name w3_baseline \
        --epochs 5 --batch_size 4 --clip_len 16
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.trackio_init import ExperimentLogger
from w2_model.model import PhysioNet, ModelConfig
from w2_model.losses import PhysioMultiTaskLoss
from w2_model.dataset import build_dataloaders
from w3_train.eval import evaluate


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Force single GPU — DataParallel with batch_size<=4 causes hangs
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger = ExperimentLogger(
        project="p3_physio_deepfake",
        run_name=args.run_name or "w3_train",
        config=vars(args),
        local_log_dir=args.log_dir,
    )

    # ─── Data ─────────────────────────────────────────────────────────────────
    fallback = [d for d in (args.fallback_cache_dir or []) if Path(d).exists()]

    # Auto-disable physio when not needed
    skip_physio = (args.w_pulse == 0 and args.w_blink == 0)
    if skip_physio and args.use_physio_fusion:
        print("Auto-disabling physio fusion (pulse/blink weights are 0)")
        args.use_physio_fusion = False
    if skip_physio:
        print("Skipping physio extraction (not used)")

    train_dl, val_dl, test_dl = build_dataloaders(
        ff_root=args.ff_root,
        celebdf_root=args.celebdf_root,
        cache_dir=args.cache_dir,
        fallback_cache_dirs=fallback or None,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment_train=True,
        skip_physio=skip_physio,
        max_train_samples=args.max_train_samples,
    )

    steps_per_epoch = len(train_dl)
    est_time = steps_per_epoch * 3.0 * args.epochs / 3600  # ~3s/step estimate
    print(f"Estimated training time: {est_time:.1f}h ({steps_per_epoch} steps/epoch × {args.epochs} epochs)")

    # ─── Model ────────────────────────────────────────────────────────────────
    cfg = ModelConfig(
        backbone=args.backbone,
        backbone_pretrained=True,
        temporal_model=args.temporal_model,
        clip_len=args.clip_len,
        img_size=args.img_size,
        use_pulse_head=(args.w_pulse > 0),
        use_blink_head=(args.w_blink > 0),
        use_physio_fusion=args.use_physio_fusion,
        temporal_pool=args.temporal_pool,
    )
    print(f"Config: physio_fusion={cfg.use_physio_fusion}, temporal_pool={cfg.temporal_pool}, "
          f"pulse_head={cfg.use_pulse_head}, blink_head={cfg.use_blink_head}")

    model = PhysioNet(cfg).to(device)

    # Load pretrain checkpoint if available
    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded pretrain: {args.pretrain_ckpt}")
    else:
        print("Training from scratch")

    # Unfreeze everything
    model.freeze_backbone(freeze=False)
    params = model.get_num_params()
    print(f"Model: {params['total']/1e6:.1f}M params, {params['trainable']/1e6:.1f}M trainable")

    # ─── Optimizer ────────────────────────────────────────────────────────────
    print(f"LR: backbone={args.lr_backbone}, head={args.lr_head}, temporal={args.lr_temporal}")
    print(f"Loss weights: cls={args.w_class}, pulse={args.w_pulse}, blink={args.w_blink}, contrastive={args.w_contrastive}")

    param_groups = [
        {"params": model.frame_encoder.parameters(), "lr": args.lr_backbone},
        {"params": model.temporal_proj.parameters(), "lr": args.lr_head},
        {"params": model.temporal.parameters(), "lr": args.lr_temporal},
        {"params": model.cls_head.parameters(), "lr": args.lr_head},
        {"params": model.fusion.parameters(), "lr": args.lr_head},
    ]
    if hasattr(model, "pulse_head"):
        param_groups.append({"params": model.pulse_head.parameters(), "lr": args.lr_head})
    if hasattr(model, "blink_head"):
        param_groups.append({"params": model.blink_head.parameters(), "lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Cosine schedule with warmup
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(steps_per_epoch, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.1, float(step) / max(1, warmup_steps))
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = PhysioMultiTaskLoss(
        w_class=args.w_class, w_pulse=args.w_pulse,
        w_blink=args.w_blink, w_contrastive=args.w_contrastive,
        pos_weight=args.pos_weight,
    )

    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0
    global_step = 0
    start_epoch = 1

    # Resume
    if args.resume and (out_dir / "latest.pt").exists():
        ckpt = torch.load(out_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("val_auc", 0.0)
        global_step = steps_per_epoch * (start_epoch - 1)
        for _ in range(global_step):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1}, best_val_auc={best_val_auc:.4f}")

    # ─── Training ─────────────────────────────────────────────────────────────
    accum_steps = args.grad_accum
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = []
        epoch_preds = []
        optimizer.zero_grad()
        epoch_start = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            frames = batch["frames"].to(device, non_blocking=True)
            rppg_feat = batch["rppg_feat"].to(device, non_blocking=True)
            blink_feat = batch["blink_feat"].to(device, non_blocking=True)
            blink_labels = batch["blink_labels"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # Diagnostics on first batch
            if epoch == start_epoch and batch_idx == 0:
                print(f"\n  [DIAG] frames: {list(frames.shape)} "
                      f"mean={frames.mean().item():.3f} std={frames.std().item():.3f}")
                print(f"  [DIAG] labels: {label.tolist()}")
                if device.type == "cuda":
                    print(f"  [DIAG] VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                outputs = model(frames, rppg_feat, blink_feat)
                losses = criterion(outputs, label, blink_target=blink_labels)

            if epoch == start_epoch and batch_idx == 0:
                with torch.no_grad():
                    prob = torch.sigmoid(outputs["logit"].float())
                    print(f"  [DIAG] logit: {outputs['logit'].tolist()} → prob: {prob.tolist()}")
                    print(f"  [DIAG] loss: {losses['total'].item():.4f}")
                    if device.type == "cuda":
                        print(f"  [DIAG] VRAM after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            loss = losses["total"] / accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == steps_per_epoch:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss.append(losses["total"].item())
            with torch.no_grad():
                probs = torch.sigmoid(outputs["logit"].float()).cpu().numpy()
                epoch_preds.extend(probs.tolist())

            global_step += 1

            # Progress update every 50 steps
            if batch_idx > 0 and batch_idx % 50 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / batch_idx * (steps_per_epoch - batch_idx)
                avg_loss = np.mean(epoch_loss[-50:])
                pbar.set_postfix_str(f"loss={avg_loss:.3f} ETA={eta/60:.0f}m")

        # ─── Epoch stats ──────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        preds_arr = np.array(epoch_preds)
        pred_mean = preds_arr.mean()
        pred_std = preds_arr.std()
        frac_fake = (preds_arr > 0.5).mean()
        mean_loss = np.mean(epoch_loss)

        print(f"  [Stats] pred_mean={pred_mean:.3f} pred_std={pred_std:.3f} "
              f"frac_fake={frac_fake:.3f} time={epoch_time/60:.1f}m")

        if pred_std < 0.05:
            print(f"  *** COLLAPSE: std={pred_std:.4f} — model not discriminating ***")
            if epoch <= 2:
                print("  → Reinitializing cls_head")
                for m in model.cls_head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        # ─── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_dl, device, scaler, split="val")
        val_auc = val_metrics.get("auc", 0.0)

        print(f"Epoch {epoch:2d} | loss={mean_loss:.3f} | "
              f"val_auc={val_auc:.4f} val_eer={val_metrics.get('eer', 0):.4f} "
              f"val_ece={val_metrics.get('ece', 0):.4f}")

        logger.log({
            "epoch": epoch,
            "train/loss": mean_loss,
            "train/pred_std": float(pred_std),
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        # Save checkpoints
        state_dict = model.state_dict()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": state_dict,
                        "val_auc": val_auc, "config": cfg, "args": vars(args)},
                       out_dir / "best_model.pt")
            print(f"  ✓ New best AUC={val_auc:.4f}")

        torch.save({"epoch": epoch, "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc}, out_dir / "latest.pt")

        # Time check — abort if we'll exceed Kaggle limit
        total_elapsed = time.time() - train_start
        remaining_epochs = args.epochs - epoch
        time_per_epoch = total_elapsed / (epoch - start_epoch + 1)
        if remaining_epochs > 0 and total_elapsed + time_per_epoch * remaining_epochs > 10 * 3600:
            print(f"  ⚠ Time limit approaching ({total_elapsed/3600:.1f}h elapsed). "
                  f"Stopping early at epoch {epoch}.")
            break

    # ─── Final test ───────────────────────────────────────────────────────────
    best_path = out_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_dl, device, scaler, split="test")

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)
    print(f"Total training time: {(time.time() - train_start)/3600:.2f}h")

    logger.log_summary({"best_val_auc": best_val_auc,
                        **{f"test/{k}": v for k, v in test_metrics.items()}})
    logger.finish()


def parse_args():
    p = argparse.ArgumentParser(description="P3 PhysioNet W3 Training")
    p.add_argument("--ff_root", default=None)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_root", default=None)
    p.add_argument("--pretrain_ckpt", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_cache", action="store_true")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--cache_dir", default="./logs/signal_cache")
    p.add_argument("--fallback_cache_dir", nargs="*", default=None)
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--run_name", default=None)

    # Model
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--temporal_model", default="transformer", choices=["transformer", "lstm", "mamba"])
    p.add_argument("--temporal_pool", default="mean", choices=["mean", "transformer"])
    p.add_argument("--use_physio_fusion", action="store_true", default=True)
    p.add_argument("--no_physio_fusion", dest="use_physio_fusion", action="store_false")
    p.add_argument("--clip_len", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=2000,
                   help="Cap training samples per epoch (0 = use all). "
                        "2000 samples / bs=4 = 500 steps ≈ 25 min/epoch on T4.")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=3e-4)
    p.add_argument("--lr_temporal", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)

    # Loss
    p.add_argument("--w_class", type=float, default=1.0)
    p.add_argument("--w_pulse", type=float, default=0.0)
    p.add_argument("--w_blink", type=float, default=0.0)
    p.add_argument("--w_contrastive", type=float, default=0.0)
    p.add_argument("--pos_weight", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
