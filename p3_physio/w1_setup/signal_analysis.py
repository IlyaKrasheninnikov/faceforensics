"""
W1: Statistical analysis of rPPG and blink signals (real vs fake).

Reads CSVs produced by extract_rppg.py and extract_blinks.py,
runs KS-tests, Mann-Whitney tests, plots distributions,
and logs everything to Trackio.

Usage:
    python signal_analysis.py \
        --rppg_real ./logs/signal_cache/rppg_summary_real.csv \
        --rppg_fake ./logs/signal_cache/rppg_summary_fake.csv \
        --blink_real ./logs/signal_cache/blinks_summary_real.csv \
        --blink_fake ./logs/signal_cache/blinks_summary_fake.csv \
        --out_dir ./figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Try trackio, fallback gracefully
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False
    print("[WARN] trackio not installed — metrics will only be printed/saved locally")


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {p} not found — skipping")
        return pd.DataFrame()
    return pd.read_csv(p)


def ks_test(real: pd.Series, fake: pd.Series, name: str) -> dict:
    """Run KS test and print result."""
    r = real.dropna().values
    f = fake.dropna().values
    # Drop non-finite values (e.g. -inf/-99 sentinel from failed SNR)
    r = r[np.isfinite(r)]
    f = f[np.isfinite(f)]
    if len(r) == 0 or len(f) == 0:
        return {}
    ks_stat, p_val = stats.ks_2samp(r, f)
    mw_stat, mw_p = stats.mannwhitneyu(r, f, alternative="two-sided")
    print(f"  {name:<35s}  KS={ks_stat:.3f} p={p_val:.4f}  MW_p={mw_p:.4f}  "
          f"real_mean={r.mean():.2f}  fake_mean={f.mean():.2f}")
    return {
        f"{name}_ks_stat": float(ks_stat),
        f"{name}_ks_p": float(p_val),
        f"{name}_mw_p": float(mw_p),
        f"{name}_real_mean": float(r.mean()),
        f"{name}_fake_mean": float(f.mean()),
        f"{name}_delta_mean": float(r.mean() - f.mean()),
    }


def plot_distribution(real: pd.Series, fake: pd.Series, title: str, xlabel: str,
                      out_path: Path, xlim=None):
    """KDE + histogram overlay: real vs fake."""
    fig, ax = plt.subplots(figsize=(8, 4))
    r = real.dropna()
    f = fake.dropna()
    r = r[np.isfinite(r)]
    f = f[np.isfinite(f)]
    if len(r) == 0 or len(f) == 0:
        plt.close()
        return

    ax.hist(r, bins=25, alpha=0.4, color="royalblue", label=f"Real (n={len(r)})", density=True)
    ax.hist(f, bins=25, alpha=0.4, color="tomato", label=f"Fake (n={len(f)})", density=True)

    if len(r) > 5:
        r_kde = stats.gaussian_kde(r)
        xs = np.linspace(r.min(), r.max(), 300)
        ax.plot(xs, r_kde(xs), color="royalblue", lw=2)
    if len(f) > 5:
        f_kde = stats.gaussian_kde(f)
        xs = np.linspace(f.min(), f.max(), 300)
        ax.plot(xs, f_kde(xs), color="tomato", lw=2)

    ax.axvline(r.mean(), color="royalblue", ls="--", lw=1.2, alpha=0.8)
    ax.axvline(f.mean(), color="tomato", ls="--", lw=1.2, alpha=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run_analysis(rppg_real_path, rppg_fake_path, blink_real_path, blink_fake_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rppg_real = load_csv(rppg_real_path)
    rppg_fake = load_csv(rppg_fake_path)
    blink_real = load_csv(blink_real_path)
    blink_fake = load_csv(blink_fake_path)

    all_metrics = {}

    # ─── rPPG Analysis ────────────────────────────────────────────────────────
    if not rppg_real.empty and not rppg_fake.empty:
        print("\n" + "=" * 70)
        print("rPPG SIGNAL ANALYSIS")
        print("=" * 70)

        for col, label, xlim in [
            ("mean_snr_db", "Mean rPPG SNR (dB)", (-20, 30)),
            ("mean_bpm", "Estimated BPM", (0, 200)),
            ("snr_chrom", "CHROM SNR (dB)", (-20, 30)),
            ("snr_pos", "POS SNR (dB)", (-20, 30)),
            ("bpm_agreement", "BPM Agreement |CHROM - POS|", (0, 60)),
        ]:
            if col in rppg_real.columns and col in rppg_fake.columns:
                m = ks_test(rppg_real[col], rppg_fake[col], col)
                all_metrics.update(m)
                plot_distribution(
                    rppg_real[col], rppg_fake[col],
                    title=f"rPPG: {label} — Real vs Fake",
                    xlabel=label,
                    out_path=out_dir / f"dist_rppg_{col}.png",
                    xlim=xlim,
                )

    # ─── Blink Analysis ───────────────────────────────────────────────────────
    if not blink_real.empty and not blink_fake.empty:
        print("\n" + "=" * 70)
        print("BLINK SIGNAL ANALYSIS")
        print("=" * 70)

        for col, label, xlim in [
            ("blinks_per_min", "Blinks per Minute", (0, 40)),
            ("ibi_cv", "Inter-Blink Interval CV (Irregularity)", (0, 2)),
            ("ear_entropy", "EAR Signal Entropy", (0, 4)),
            ("ear_mean", "Mean EAR", (0, 0.5)),
            ("mean_blink_dur", "Mean Blink Duration (frames)", (0, 20)),
        ]:
            if col in blink_real.columns and col in blink_fake.columns:
                m = ks_test(blink_real[col], blink_fake[col], col)
                all_metrics.update(m)
                plot_distribution(
                    blink_real[col], blink_fake[col],
                    title=f"Blink: {label} — Real vs Fake",
                    xlabel=label,
                    out_path=out_dir / f"dist_blink_{col}.png",
                    xlim=xlim,
                )

    # ─── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — significant differences (p < 0.05)")
    print("=" * 70)

    sig_metrics = {k: v for k, v in all_metrics.items() if "_ks_p" in k and v < 0.05}
    for k, v in sorted(sig_metrics.items(), key=lambda x: x[1]):
        base = k.replace("_ks_p", "")
        delta = all_metrics.get(f"{base}_delta_mean", "?")
        print(f"  {base:<40s}  KS p={v:.4f}  Δmean={delta:.3f}")

    # ─── Log to Trackio ───────────────────────────────────────────────────────
    if TRACKIO_AVAILABLE and all_metrics:
        trackio.init(project="p3_physio_deepfake", space_id="GoodBoyXD/bioforensics")
        trackio.log(all_metrics)
        trackio.log({
            "n_significant_features": len(sig_metrics),
            "rppg_real_n": len(rppg_real),
            "rppg_fake_n": len(rppg_fake),
            "blink_real_n": len(blink_real),
            "blink_fake_n": len(blink_fake),
        })
        print(f"\n[Trackio] Metrics logged to project 'p3_physio_deepfake'")

    # Save to CSV fallback
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(out_dir / "w1_analysis_metrics.csv", index=False)
    print(f"\n[Local] Metrics saved → {out_dir / 'w1_analysis_metrics.csv'}")

    return all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rppg_real", default="./logs/signal_cache/rppg_summary_real.csv")
    p.add_argument("--rppg_fake", default="./logs/signal_cache/rppg_summary_fake.csv")
    p.add_argument("--blink_real", default="./logs/signal_cache/blinks_summary_real.csv")
    p.add_argument("--blink_fake", default="./logs/signal_cache/blinks_summary_fake.csv")
    p.add_argument("--out_dir", default="./figures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = run_analysis(
        args.rppg_real, args.rppg_fake,
        args.blink_real, args.blink_fake,
        args.out_dir,
    )
