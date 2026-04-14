"""
W4: Cross-dataset evaluation harness.

Tests a trained model on multiple datasets it was NOT trained on.
Key metric: does performance hold up across datasets?

Usage:
    python w4_full_train/eval_cross.py \
        --checkpoint ./checkpoints/best_model.pt \
        --ff_root /data/FF++ \
        --celebdf_root /data/CelebDF-v2 \
        --dfdc_root /data/DFDC
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.trackio_init import ExperimentLogger
from w2_model.model import PhysioNet
from w2_model.dataset import (
    PhysioDeepfakeDataset,
    build_ff_plus_plus_list,
    build_celebdf_list,
)
from w3_train.eval import evaluate
from torch.utils.data import DataLoader


def build_cross_eval_loaders(args) -> dict:
    """Build test dataloaders for each dataset separately."""
    loaders = {}

    # FF++ — per manipulation type
    if args.ff_root:
        from w2_model.dataset import FF_MANIPULATION_TYPES
        for manip in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            vid_dir = Path(args.ff_root) / manip / "c23" / "videos"
            real_dir = Path(args.ff_root) / "original" / "c23" / "videos"
            if vid_dir.exists() and real_dir.exists():
                reals = sorted(list(real_dir.glob("*.mp4")))[:200]
                fakes = sorted(list(vid_dir.glob("*.mp4")))[:200]
                all_paths = [str(v) for v in reals + fakes]
                all_labels = [0] * len(reals) + [1] * len(fakes)
                ds = PhysioDeepfakeDataset(all_paths, all_labels, augment=False, cache_dir=args.cache_dir)
                loaders[f"FF++_{manip}"] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # CelebDF-v2
    if args.celebdf_root:
        paths, labels = build_celebdf_list(args.celebdf_root)
        if paths:
            ds = PhysioDeepfakeDataset(paths, labels, augment=False, cache_dir=args.cache_dir)
            loaders["CelebDF-v2"] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return loaders


def run_cross_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config")
    model = PhysioNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    logger = ExperimentLogger(
        project="p3_physio_deepfake",
        run_name="w4_cross_eval",
        local_log_dir=args.log_dir,
    )

    loaders = build_cross_eval_loaders(args)
    if not loaders:
        print("[ERROR] No datasets found for cross-evaluation")
        return

    rows = []
    for dataset_name, loader in loaders.items():
        print(f"\nEvaluating on: {dataset_name}")
        metrics = evaluate(model, loader, device, split=dataset_name)

        row = {"dataset": dataset_name, **{k: v for k, v in metrics.items() if isinstance(v, float)}}
        rows.append(row)

        logger.log({f"{dataset_name}/{k}": v for k, v in metrics.items() if isinstance(v, float)})
        print(f"  AUC={metrics.get('auc', 0):.4f}  EER={metrics.get('eer', 0):.4f}  "
              f"ECE={metrics.get('ece', 0):.4f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("CROSS-DATASET RESULTS:")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 80)

    out_csv = Path(args.log_dir) / "cross_eval_results.csv"
    df.to_csv(out_csv, index=False)
    logger.log_table("cross_eval", df)
    logger.finish()
    print(f"\nSaved → {out_csv}")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ff_root", default=None)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_root", default=None)
    p.add_argument("--cache_dir", default="./logs/signal_cache")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cross_eval(args)
