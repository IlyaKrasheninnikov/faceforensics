"""
W9: Publication-quality figure — Blink feature distribution (real vs fake).

Produces Figure 4 for paper:
  - Left: EAR (Eye Aspect Ratio) distribution real vs fake
  - Right: Blink rate (blinks/min) violin plot real vs fake

Usage:
    python w9_viz/plot_blink_timeline.py \
        --blink_cache /kaggle/input/datasets/goodboyxdd/blink-v1/blink \
        --ff_root /kaggle/input/.../frames \
        --out_dir ./figures
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

FF_MANIPULATION_TYPES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
}


def load_blink_features(ff_root, blink_cache, max_per_class=500):
    """Load blink features and group by real/fake."""
    ff_root = Path(ff_root)
    blink_cache = Path(blink_cache)

    real_feats, fake_feats = [], []
    real_rates, fake_rates = [], []

    for manip, label in FF_MANIPULATION_TYPES.items():
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            continue
        subdirs = sorted([d for d in manip_dir.iterdir() if d.is_dir()])
        count = 0
        for sd in subdirs:
            bp = blink_cache / manip / sd.name / "blink_feat.npy"
            if not bp.exists():
                continue
            feat = np.load(str(bp)).astype(np.float32)
            if len(feat) != 16 or np.all(feat == 0):
                continue

            # feat layout: [mean_ear, std_ear, min_ear, max_ear, range_ear,
            #               blink_count, blink_rate, mean_dur, std_dur, max_dur,
            #               mean_interval, std_interval, cv_interval, ear_velocity_mean,
            #               ear_velocity_std, regularity_score]
            blink_rate = feat[6]  # blinks/min

            if label == 0:
                real_feats.append(feat)
                real_rates.append(blink_rate)
            else:
                fake_feats.append(feat)
                fake_rates.append(blink_rate)
            count += 1
            if label == 0 and count >= max_per_class:
                break
            if label == 1 and count >= max_per_class:
                break

    return (np.array(real_feats), np.array(fake_feats),
            np.array(real_rates), np.array(fake_rates))


def plot_blink_analysis(ff_root, blink_cache, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading blink features...")
    real_feats, fake_feats, real_rates, fake_rates = load_blink_features(ff_root, blink_cache)
    print(f"  Real: {len(real_feats)}, Fake: {len(fake_feats)}")

    colors = {"real": "#2196F3", "fake": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Mean EAR distribution
    ax1 = axes[0]
    real_ear = real_feats[:, 0]  # mean_ear
    fake_ear = fake_feats[:, 0]
    ax1.hist(real_ear, bins=40, alpha=0.6, color=colors["real"], label="Real", density=True)
    ax1.hist(fake_ear, bins=40, alpha=0.6, color=colors["fake"], label="Fake", density=True)
    ks_ear, p_ear = stats.ks_2samp(real_ear, fake_ear)
    ax1.set_title(f"Mean EAR Distribution\nKS={ks_ear:.3f}, p={p_ear:.2e}", fontweight="bold")
    ax1.set_xlabel("Mean Eye Aspect Ratio")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Panel 2: Blink rate distribution
    ax2 = axes[1]
    # Filter extreme rates for visualization
    real_r = real_rates[real_rates < 80]
    fake_r = fake_rates[fake_rates < 80]
    ax2.hist(real_r, bins=40, alpha=0.6, color=colors["real"], label=f"Real (mean={real_r.mean():.1f})", density=True)
    ax2.hist(fake_r, bins=40, alpha=0.6, color=colors["fake"], label=f"Fake (mean={fake_r.mean():.1f})", density=True)
    ks_rate, p_rate = stats.ks_2samp(real_rates, fake_rates)
    ax2.set_title(f"Blink Rate (blinks/min)\nKS={ks_rate:.3f}, p={p_rate:.2e}", fontweight="bold")
    ax2.set_xlabel("Blinks per minute")
    ax2.set_ylabel("Density")
    ax2.legend()

    # Panel 3: Feature importance (which blink features differ most)
    ax3 = axes[2]
    feat_names = ["mean_ear", "std_ear", "min_ear", "max_ear", "range_ear",
                  "blink_cnt", "blink_rate", "mean_dur", "std_dur", "max_dur",
                  "mean_int", "std_int", "cv_int", "ear_vel_m",
                  "ear_vel_s", "regularity"]
    ks_vals = []
    for i in range(16):
        ks, _ = stats.ks_2samp(real_feats[:, i], fake_feats[:, i])
        ks_vals.append(ks)
    ks_vals = np.array(ks_vals)
    order = np.argsort(ks_vals)[::-1]

    y_pos = np.arange(16)
    ax3.barh(y_pos, ks_vals[order], color=["#F44336" if ks_vals[order[i]] > 0.1 else "#9E9E9E"
                                            for i in range(16)])
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([feat_names[i] for i in order], fontsize=9)
    ax3.set_xlabel("KS Statistic (real vs fake)")
    ax3.set_title("Blink Feature Discriminability", fontweight="bold")
    ax3.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5)
    ax3.invert_yaxis()

    fig.suptitle("P3: Blink Feature Analysis — Real vs Deepfake",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_blink_analysis.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'fig4_blink_analysis.png'}")


def main(args):
    plot_blink_analysis(args.ff_root, args.blink_cache, args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W9: Blink feature distribution plots")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--blink_cache", required=True)
    p.add_argument("--out_dir", default="./figures")
    main(p.parse_args())
