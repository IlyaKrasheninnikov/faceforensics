"""
W9: Publication-quality figure — rPPG waveform & spectrum comparison (real vs fake).

Produces Figure 1 for paper: side-by-side showing:
  - Top: skin pixel intensity time series (real vs fake)
  - Bottom: FFT power spectrum in physiological frequency band

Usage:
    python w9_viz/plot_rppg_comparison.py \
        --real_video /data/FF++/original/c23/videos/000.mp4 \
        --fake_video /data/FF++/Deepfakes/c23/videos/000.mp4 \
        --out_dir ./figures
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.extract_rppg import get_face_roi_signals, chrom_method, pos_method


def load_frames(video_path, n_frames=150, fps_target=15.0):
    """Load frames from either a video file (mp4/avi) or a directory of images."""
    p = Path(video_path)
    frames = []

    if p.is_dir():
        # Directory of pre-extracted frames (common on Kaggle FF++ processed sets)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files.extend(sorted(p.glob(e)))
        if not files:
            # Try one level deeper
            for e in exts:
                files.extend(sorted(p.glob(f"*/{e}")))
        files = sorted(files)[:n_frames]
        if not files:
            raise RuntimeError(f"No image frames found in directory: {video_path}")
        for fp in files:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            frames.append(img.astype(np.float32) / 255.0)
    else:
        if not p.exists():
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"OpenCV could not open: {video_path}\n"
                f"  Hint: On Kaggle, FF++ 'processed' datasets usually contain\n"
                f"        pre-extracted frame folders (not .mp4). Pass the folder\n"
                f"        path directly (e.g. .../videos/000/) instead of an .mp4."
            )
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(orig_fps / fps_target))
        idx = 0
        while len(frames) < n_frames:
            ret, f = cap.read()
            if not ret:
                break
            if idx % step == 0:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (224, 224))
                frames.append(f.astype(np.float32) / 255.0)
            idx += 1
        cap.release()
        if not frames:
            raise RuntimeError(
                f"Opened {video_path} but read 0 frames. "
                f"Codec may be unsupported; try extracting frames first or "
                f"pass a frame-folder path."
            )
    return np.stack(frames)


def get_pulse(frames, fps=15.0):
    roi = get_face_roi_signals(frames, fps)
    combined = (roi["forehead_rgb"] * 0.4 + roi["left_cheek_rgb"] * 0.3 + roi["right_cheek_rgb"] * 0.3)
    pulse = chrom_method(combined, fps)
    return pulse, combined[:, 1]  # pulse waveform + green channel (rPPG proxy)


def plot_comparison(real_video, fake_video, out_dir, fps=15.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real video...")
    real_frames = load_frames(real_video, 150, fps)
    print("Loading fake video...")
    fake_frames = load_frames(fake_video, 150, fps)

    real_pulse, real_green = get_pulse(real_frames, fps)
    fake_pulse, fake_green = get_pulse(fake_frames, fps)

    T = min(len(real_pulse), len(fake_pulse), 150)
    time_axis = np.arange(T) / fps

    # FFT
    freqs = rfftfreq(T, d=1.0 / fps)
    real_psd = np.abs(rfft(real_pulse[:T])) ** 2
    fake_psd = np.abs(rfft(fake_pulse[:T])) ** 2
    band = (freqs >= 0.5) & (freqs <= 4.0)

    # ─── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    colors = {"real": "#2196F3", "fake": "#F44336"}

    # Top-left: Raw green channel (rPPG proxy signal)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis, real_green[:T], color=colors["real"], lw=1.2, alpha=0.8, label="Real")
    ax1.set_title("Face Green Channel Intensity (rPPG proxy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normalized intensity")
    ax1.legend()

    # Top-right: Fake green channel
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_axis, fake_green[:T], color=colors["fake"], lw=1.2, alpha=0.8, label="Fake")
    ax2.set_title("Face Green Channel Intensity (rPPG proxy)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalized intensity")
    ax2.legend()

    # Bottom-left: rPPG waveforms comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_axis, real_pulse[:T] / (np.abs(real_pulse[:T]).max() + 1e-8),
             color=colors["real"], lw=1.5, label="Real (CHROM)")
    ax3.plot(time_axis, fake_pulse[:T] / (np.abs(fake_pulse[:T]).max() + 1e-8),
             color=colors["fake"], lw=1.5, alpha=0.8, label="Fake (CHROM)")
    ax3.set_title("Extracted rPPG Waveform Comparison", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalized amplitude")
    ax3.legend()

    # Bottom-right: Power spectra
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(freqs[band], real_psd[band] / real_psd[band].max(),
             color=colors["real"], lw=2, label=f"Real (peak={freqs[band][real_psd[band].argmax()]*60:.0f} BPM)")
    ax4.plot(freqs[band], fake_psd[band] / (fake_psd[band].max() + 1e-8),
             color=colors["fake"], lw=2, alpha=0.8, label="Fake (no clear peak)")
    ax4.set_title("PSD in Physiological Band (0.5–4 Hz)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Normalized PSD")
    ax4.set_xlim(0.5, 4.0)
    # Add BPM secondary x-axis labels
    ax4_top = ax4.twiny()
    ax4_top.set_xlim(0.5 * 60, 4.0 * 60)
    ax4_top.set_xlabel("Heart Rate (BPM)")
    ax4.legend()

    fig.suptitle(
        "P3: Physiological Signal Analysis — Real vs Deepfake Video",
        fontsize=13, fontweight="bold", y=1.01
    )

    out_png = out_dir / "fig1_rppg_comparison.png"
    out_pdf = out_dir / "fig1_rppg_comparison.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_video", required=True)
    p.add_argument("--fake_video", required=True)
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--fps", type=float, default=15.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_comparison(args.real_video, args.fake_video, args.out_dir, args.fps)
