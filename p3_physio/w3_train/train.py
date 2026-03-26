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
    fallback = [d for d in (args.fallback_cache_dir or []) if Path(d).exists()]
    # Skip physio extraction when features aren't used — saves heavy CPU work
    print(not args.use_physio_fusion, args.w_pulse == 0, args.w_blink == 0)
    skip_physio = not args.use_physio_fusion and args.w_pulse == 0 and args.w_blink == 0
    if skip_physio:
        print("Skipping physio feature extraction (not used in this config)")

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
        use_pulse_head=(args.w_pulse > 0),
        use_blink_head=(args.w_blink > 0),
        use_physio_fusion=args.use_physio_fusion,
        temporal_pool=args.temporal_pool,
    )
    print(f"Config: physio_fusion={cfg.use_physio_fusion}, temporal_pool={cfg.temporal_pool}, "
          f"pulse_head={cfg.use_pulse_head}, blink_head={cfg.use_blink_head}")
    model = PhysioNet(cfg).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded pretrain checkpoint: {args.pretrain_ckpt}")
    else:
        print("Training from scratch (no pretrain checkpoint)")

    # FIX: Train end-to-end from the start — do NOT freeze backbone.
    # Previous approach froze backbone for 2 epochs, but with random/pretrain features
    # the heads learned the trivial "always predict fake" solution and never recovered.
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.freeze_backbone(freeze=False)
    print("Training end-to-end (backbone NOT frozen)")

    params = base_model.get_num_params()
    print(f"Model: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")

    # ─── Verify learning rates ───────────────────────────────────────────────
    print(f"Learning rates: backbone={args.lr_backbone}, head={args.lr_head}, temporal={args.lr_temporal}")
    print(f"Loss weights: cls={args.w_class}, pulse={args.w_pulse}, blink={args.w_blink}, contrastive={args.w_contrastive}")
    if args.lr_head == 0:
        print("[FATAL] lr_head is 0 — model will not learn. Check your command-line args!")
        print("  Hint: use dots not commas for decimals (0.4 not 0,4)")
        raise ValueError("lr_head cannot be 0")

    # ─── Loss ─────────────────────────────────────────────────────────────────
    # WeightedRandomSampler already balances batches to ~50/50 real/fake.
    # No additional pos_weight needed — it was previously causing collapse by
    # applying 5.94x weight to fakes (majority class), training model to predict all-fake.
    print(f"pos_weight={args.pos_weight} (1.0 = no reweighting, sampler handles balance)")

    criterion = PhysioMultiTaskLoss(
        w_class=args.w_class,
        w_pulse=args.w_pulse,
        w_blink=args.w_blink,
        w_contrastive=args.w_contrastive,
        pos_weight=args.pos_weight,
    )

    # ─── Optimizer ────────────────────────────────────────────────────────────
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model

    param_groups = [
        {"params": base_model.frame_encoder.parameters(), "lr": args.lr_backbone},
        {"params": base_model.temporal_proj.parameters(), "lr": args.lr_head},
        {"params": base_model.temporal.parameters(), "lr": args.lr_temporal},
        {"params": base_model.cls_head.parameters(), "lr": args.lr_head},
        {"params": base_model.fusion.parameters(), "lr": args.lr_head},
    ]
    if hasattr(base_model, "pulse_head"):
        param_groups.append({"params": base_model.pulse_head.parameters(), "lr": args.lr_head})
    if hasattr(base_model, "blink_head"):
        param_groups.append({"params": base_model.blink_head.parameters(), "lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_dl) * args.epochs
    warmup_steps = len(train_dl) * args.warmup_epochs
    # FIX: Per-group max_lr so backbone gets lower LR than heads.
    # Previously used single max_lr which ramped backbone to 1e-4 (10x too high).
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)


    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0
    global_step = 0
    start_epoch = 1

    # Resume from latest checkpoint if --resume
    resume_skip_batches = 0
    if args.resume and (out_dir / "latest.pt").exists():
        resume_ckpt = torch.load(out_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0)
        # If checkpoint was mid-epoch, resume from that epoch and skip batches
        if "batch_idx" in resume_ckpt:
            resume_skip_batches = resume_ckpt["batch_idx"] + 1
            print(f"Resumed from epoch {start_epoch}, batch {resume_skip_batches}")
        else:
            start_epoch += 1  # completed full epoch, start next
            print(f"Resumed from epoch {start_epoch - 1} (completed)")
        best_val_auc = resume_ckpt.get("val_auc", 0.0)
        global_step = resume_ckpt.get("global_step", len(train_dl) * (start_epoch - 1))
        # Fast-forward scheduler
        for _ in range(global_step):
            scheduler.step()
        print(f"  best_val_auc={best_val_auc:.4f}, global_step={global_step}")

    # ─── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        model.train()
        epoch_loss_sums = {k: 0.0 for k in ["total", "cls", "pulse", "blink", "contrastive"]}
        epoch_loss_count = 0
        # Track predictions for collapse detection (running stats, not full list)
        pred_sum = 0.0
        pred_sq_sum = 0.0
        pred_count = 0
        pred_above_05 = 0

        accum_steps = args.grad_accum
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)):
            # Skip batches already processed (for mid-epoch resume)
            if resume_skip_batches > 0:
                resume_skip_batches -= 1
                continue

            frames = batch["frames"].to(device, non_blocking=True)
            rppg_feat = batch["rppg_feat"].to(device, non_blocking=True)
            blink_feat = batch["blink_feat"].to(device, non_blocking=True)
            blink_labels = batch["blink_labels"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # One-time diagnostic: print input statistics on first batch of first epoch
            if epoch == start_epoch and batch_idx == 0:
                print(f"\n  [DIAG] frames: shape={list(frames.shape)} "
                      f"mean={frames.mean().item():.3f} std={frames.std().item():.3f} "
                      f"min={frames.min().item():.3f} max={frames.max().item():.3f}")
                print(f"  [DIAG] rppg_feat: shape={list(rppg_feat.shape)} "
                      f"mean={rppg_feat.mean().item():.3f} std={rppg_feat.std().item():.3f} "
                      f"norm={rppg_feat.norm(dim=-1).mean().item():.3f}")
                print(f"  [DIAG] blink_feat: shape={list(blink_feat.shape)} "
                      f"mean={blink_feat.mean().item():.3f} std={blink_feat.std().item():.3f}")
                print(f"  [DIAG] labels: {label.tolist()}")
                print(f"  [DIAG] effective_batch={args.batch_size}x{args.grad_accum}={args.batch_size * args.grad_accum}")

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(frames, rppg_feat, blink_feat)
                losses = criterion(outputs, label, blink_target=blink_labels)

            # One-time: print output statistics
            if epoch == start_epoch and batch_idx == 0:
                with torch.no_grad():
                    logit = outputs["logit"]
                    prob = torch.sigmoid(logit.float())
                    print(f"  [DIAG] logit: {logit.tolist()} → prob: {prob.tolist()}")
                    print(f"  [DIAG] losses: " + " | ".join(
                        f"{k}={v.item():.4f}" if torch.is_tensor(v) else f"{k}={v:.4f}"
                        for k, v in losses.items()))

            # Scale loss by accumulation steps for correct gradient magnitude
            loss_tensor = losses["total"] / accum_steps

            if scaler:
                scaler.scale(loss_tensor).backward()
            else:
                loss_tensor.backward()

            # Step optimizer every accum_steps batches (or at end of epoch)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_dl):
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

            for k, v in losses.items():
                if k in epoch_loss_sums:
                    epoch_loss_sums[k] += v.item() if torch.is_tensor(v) else float(v)
            epoch_loss_count += 1
            # Keep last loss values for step logging
            last_losses = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in losses.items() if k in epoch_loss_sums}

            # Collect running stats for collapse detection (no list — saves RAM)
            with torch.no_grad():
                probs = torch.sigmoid(outputs["logit"].float()).cpu()
                pred_sum += probs.sum().item()
                pred_sq_sum += (probs ** 2).sum().item()
                pred_count += probs.numel()
                pred_above_05 += (probs > 0.5).sum().item()
                del probs

            # Explicitly free GPU tensors to prevent memory buildup
            del frames, rppg_feat, blink_feat, blink_labels, label, outputs, losses, loss_tensor

            global_step += 1
            if global_step % args.log_interval == 0:
                logger.log({
                    "step/loss_total": last_losses.get("total", 0),
                    "step/loss_cls": last_losses.get("cls", 0),
                    "step/loss_pulse": last_losses.get("pulse", 0),
                    "step/loss_blink": last_losses.get("blink", 0),
                }, step=global_step)

            # Periodic CUDA cache clearing to prevent OOM from fragmentation
            if device.type == "cuda" and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                import gc; gc.collect()

            # Memory monitoring: print GPU usage every 100 batches
            if device.type == "cuda" and (batch_idx + 1) % 100 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_alloc = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  [MEM] batch {batch_idx+1}: alloc={allocated:.2f}GB reserved={reserved:.2f}GB peak={max_alloc:.2f}GB")

            # Mid-epoch checkpoint every 100 batches (saves progress before Kaggle kills)
            if (batch_idx + 1) % 100 == 0:
                state_dict_mid = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "model_state_dict": state_dict_mid,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "global_step": global_step,
                }, out_dir / "latest.pt")
                del state_dict_mid
                print(f"  [Checkpoint] Saved mid-epoch at batch {batch_idx + 1}/{len(train_dl)}")

        # ─── Collapse detection ──────────────────────────────────────────────
        pred_mean = pred_sum / max(pred_count, 1)
        pred_std = max(0, pred_sq_sum / max(pred_count, 1) - pred_mean ** 2) ** 0.5
        frac_above_05 = pred_above_05 / max(pred_count, 1)
        print(f"  [Collapse check] pred_mean={pred_mean:.3f} pred_std={pred_std:.3f} "
              f"frac_fake={frac_above_05:.3f}")
        if pred_std < 0.05:
            print(f"  *** WARNING: Prediction collapse detected! std={pred_std:.4f} "
                  f"(all predictions ~{pred_mean:.2f}). Model is not discriminating. ***")

        # ─── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_dl, device, scaler, split="val")

        epoch_loss_avgs = {k: v / max(epoch_loss_count, 1) for k, v in epoch_loss_sums.items()}
        metrics = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in epoch_loss_avgs.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
            "train/pred_mean": float(pred_mean),
            "train/pred_std": float(pred_std),
        }
        logger.log(metrics, step=epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"loss={metrics.get('train/total', 0):.3f} | "
            f"val_auc={val_metrics.get('auc', 0):.4f} "
            f"val_eer={val_metrics.get('eer', 0):.4f} "
            f"val_ece={val_metrics.get('ece', 0):.4f}"
        )
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        # Save best by AUC
        val_auc = val_metrics.get("auc", 0.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": state_dict,
                "val_auc": val_auc,
                "config": cfg,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ New best AUC={val_auc:.4f} → {ckpt_path}")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc,
        }, out_dir / "latest.pt")

    # ─── Final test evaluation ─────────────────────────────────────────────────
    # FIX: weights_only=False for PyTorch 2.6 (ModelConfig is a custom class)
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
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
    p.add_argument("--fallback_cache_dir", nargs="*", default=None,
                   help="Read-only cache dirs to check before recomputing features (e.g. Kaggle input datasets)")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--run_name", default=None)

    # Model
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--temporal_model", default="transformer", choices=["transformer", "lstm", "mamba"])
    p.add_argument("--temporal_pool", default="transformer", choices=["mean", "transformer"],
                   help="'mean' = skip transformer, just mean-pool frame features. 'transformer' = full temporal encoder.")
    p.add_argument("--use_physio_fusion", action="store_true", default=True,
                   help="Fuse rPPG+blink features into classification. Use --no_physio_fusion to disable.")
    p.add_argument("--no_physio_fusion", dest="use_physio_fusion", action="store_false",
                   help="Bypass rPPG/blink fusion — classify from temporal features only.")
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
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--log_interval", type=int, default=20)

    # Loss weights
    p.add_argument("--w_class", type=float, default=1.0)
    p.add_argument("--w_pulse", type=float, default=0.4)
    p.add_argument("--w_blink", type=float, default=0.3)
    p.add_argument("--w_contrastive", type=float, default=0.1)
    p.add_argument("--pos_weight", type=float, default=1.0,
                   help="BCE pos_weight for real class. Default=1.0 triggers auto-compute from class ratio.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
