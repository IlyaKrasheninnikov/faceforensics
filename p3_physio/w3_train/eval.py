"""
W3: Evaluation utilities for PhysioNet.

Computes: AUC, EER, AP, ECE (Expected Calibration Error)
Can be used standalone or imported by training scripts.

Usage:
    python w3_train/eval.py \
        --checkpoint ./checkpoints/best_model.pt \
        --ff_root /data/FF++ \
        --split test

    # Cross-dataset evaluation:
    python w3_train/eval.py \
        --checkpoint ./checkpoints/best_model.pt \
        --celebdf_root /data/CelebDF-v2 \
        --split cross
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER): the threshold where FAR == FRR.
    Returns EER in [0, 1].
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Find crossing point
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    Perfect calibration: ECE = 0. Target: ECE < 0.08.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(probs))


def compute_metrics(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    all_scores: (N,) predicted probabilities [0, 1]
    all_labels: (N,) ground-truth {0=real, 1=fake}
    """
    metrics = {}

    if SKLEARN_AVAILABLE:
        metrics["auc"] = float(roc_auc_score(all_labels, all_scores))
        metrics["ap"] = float(average_precision_score(all_labels, all_scores))
        metrics["eer"] = compute_eer(all_scores, all_labels)
    else:
        # Manual AUC via trapezoidal rule
        sorted_idx = np.argsort(-all_scores)
        tp, fp, tn, fn = 0, 0, int((all_labels == 0).sum()), int((all_labels == 1).sum())
        tprs, fprs = [0.0], [0.0]
        for i in sorted_idx:
            if all_labels[i] == 1:
                tp += 1; fn -= 1
            else:
                fp += 1; tn -= 1
            tprs.append(tp / (tp + fn + 1e-8))
            fprs.append(fp / (fp + tn + 1e-8))
        metrics["auc"] = float(np.trapz(tprs, fprs))
        metrics["eer"] = 0.0  # skip without sklearn

    metrics["ece"] = compute_ece(all_scores, all_labels)

    # Threshold-based metrics
    preds = (all_scores >= threshold).astype(int)
    tp = int(((preds == 1) & (all_labels == 1)).sum())
    fp = int(((preds == 1) & (all_labels == 0)).sum())
    tn = int(((preds == 0) & (all_labels == 0)).sum())
    fn = int(((preds == 0) & (all_labels == 1)).sum())

    metrics["precision"] = tp / (tp + fp + 1e-8)
    metrics["recall"] = tp / (tp + fn + 1e-8)
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"] + 1e-8)
    metrics["accuracy"] = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    metrics["fpr_at_1pct_fnr"] = _fpr_at_fnr(all_scores, all_labels, target_fnr=0.01)

    return metrics


def _fpr_at_fnr(scores: np.ndarray, labels: np.ndarray, target_fnr: float = 0.01) -> float:
    """FPR at a given FNR operating point."""
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        idx = np.argmin(np.abs(fnr - target_fnr))
        return float(fpr[idx])
    except Exception:
        return 0.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    scaler=None,
    split: str = "val",
) -> Dict[str, float]:
    """
    Run inference on a dataloader and compute all metrics.
    Returns metrics dict.
    """
    model.eval()
    all_scores, all_labels = [], []

    for batch in tqdm(dataloader, desc=f"Evaluating [{split}]", leave=False):
        frames = batch["frames"].to(device)
        rppg_feat = batch["rppg_feat"].to(device)
        blink_feat = batch["blink_feat"].to(device)
        label = batch["label"].cpu().numpy()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(frames, rppg_feat, blink_feat)
            # Cast to float32 before sigmoid to avoid fp16 overflow → NaN
            probs = torch.sigmoid(outputs["logit"].float()).cpu().numpy()

        all_scores.extend(probs.tolist())
        all_labels.extend(label.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Replace any remaining NaN/Inf with 0.5 (neutral prediction)
    all_scores = np.nan_to_num(all_scores, nan=0.5, posinf=1.0, neginf=0.0)

    metrics = compute_metrics(all_scores, all_labels)
    return metrics


def eval_standalone(args):
    """Standalone evaluation from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from w2_model.model import PhysioNet
    from w2_model.dataset import build_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PhysioNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Build test dataloader
    _, _, test_dl = build_dataloaders(
        ff_root=args.ff_root,
        celebdf_root=args.celebdf_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=2,
    )

    metrics = evaluate(model, test_dl, device, split=args.split)

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS [{args.split}]:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<25s}: {v:.4f}")
    print("=" * 50)

    # Log to Trackio
    try:
        from w1_setup.trackio_init import ExperimentLogger
        logger = ExperimentLogger(project="p3_physio_deepfake", run_name=f"eval_{args.split}")
        logger.log_summary(metrics)
        logger.finish()
    except Exception:
        pass

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ff_root", default=None)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--cache_dir", default="./logs/signal_cache")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--split", default="test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_standalone(args)
