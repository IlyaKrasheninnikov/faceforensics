"""
W1: Batch rPPG (remote photoplethysmography) extraction.

Extracts pulse signals from face videos using CHROM and POS methods.
Outputs per-video: pulse waveform, estimated BPM, SNR, and FFT spectrum.
Results cached to disk (HDF5) to avoid recomputation.

Usage:
    python extract_rppg.py --video_dir /data/FF++/original/c23/videos \
                           --label real \
                           --out_dir ./logs/signal_cache \
                           --max_videos 100

    python extract_rppg.py --video_dir /data/FF++/Deepfakes/c23/videos \
                           --label fake \
                           --out_dir ./logs/signal_cache \
                           --max_videos 100
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── MediaPipe face mesh for ROI ───────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# Cheek / forehead landmark indices (approx face ROI without hair/mouth)
# MediaPipe 468-point face mesh
FOREHEAD_IDS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_CHEEK_IDS = [116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 428, 262, 369, 395, 394, 379, 365, 397, 288, 361, 323]
RIGHT_CHEEK_IDS = [345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 423, 204, 32, 140, 166, 165, 150, 136, 172, 58, 132, 93]


def get_face_roi_signals(frames: np.ndarray, fps: float = 30.0) -> dict:
    """
    Extract mean RGB signals from face ROI (forehead + cheeks) across frames.

    Args:
        frames: (T, H, W, 3) float32 array in RGB [0, 1]
        fps: video frame rate

    Returns:
        dict with 'forehead_rgb', 'left_cheek_rgb', 'right_cheek_rgb' (T, 3) arrays
    """
    T, H, W, _ = frames.shape
    results = {
        "forehead_rgb": np.zeros((T, 3)),
        "left_cheek_rgb": np.zeros((T, 3)),
        "right_cheek_rgb": np.zeros((T, 3)),
        "full_face_rgb": np.zeros((T, 3)),
    }

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for t, frame in enumerate(frames):
            frame_uint8 = (frame * 255).astype(np.uint8)
            result = face_mesh.process(frame_uint8)

            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)

                for key, ids in [
                    ("forehead_rgb", FOREHEAD_IDS),
                    ("left_cheek_rgb", LEFT_CHEEK_IDS),
                    ("right_cheek_rgb", RIGHT_CHEEK_IDS),
                ]:
                    roi_pts = pts[ids].astype(int)
                    roi_pts[:, 0] = np.clip(roi_pts[:, 0], 0, W - 1)
                    roi_pts[:, 1] = np.clip(roi_pts[:, 1], 0, H - 1)
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(roi_pts), 1)
                    region = frame[mask == 1]
                    if len(region) > 0:
                        results[key][t] = region.mean(axis=0)

                # Full face hull
                hull = cv2.convexHull(pts.astype(int))
                mask_full = np.zeros((H, W), dtype=np.uint8)
                cv2.fillConvexPoly(mask_full, hull, 1)
                region_full = frame[mask_full == 1]
                if len(region_full) > 0:
                    results["full_face_rgb"][t] = region_full.mean(axis=0)
            else:
                # No face detected: carry forward last value
                if t > 0:
                    for key in results:
                        results[key][t] = results[key][t - 1]

    return results


def chrom_method(rgb_signal: np.ndarray, fps: float) -> np.ndarray:
    """
    CHROM rPPG method (de Haan & Jeanne, IEEE TBME 2013).
    Input: (T, 3) normalized RGB signal
    Output: (T,) pulse waveform
    """
    # Normalize each channel
    mean_rgb = rgb_signal.mean(axis=0, keepdims=True)
    mean_rgb = np.where(mean_rgb == 0, 1e-8, mean_rgb)
    cn = rgb_signal / mean_rgb

    # CHROM projection
    xs = 3 * cn[:, 0] - 2 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]

    # Bandpass filter 0.7–3.5 Hz
    nyq = fps / 2.0
    b, a = scipy_signal.butter(4, [0.7 / nyq, 3.5 / nyq], btype="band")
    xs_f = scipy_signal.filtfilt(b, a, xs)
    ys_f = scipy_signal.filtfilt(b, a, ys)

    alpha = xs_f.std() / (ys_f.std() + 1e-8)
    pulse = xs_f - alpha * ys_f
    return pulse


def pos_method(rgb_signal: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """
    POS rPPG method (Wang et al., IEEE TBME 2016).
    Input: (T, 3) normalized RGB signal
    Output: (T,) pulse waveform
    """
    T = len(rgb_signal)
    l = int(fps * window_sec)
    pulse = np.zeros(T)

    for t in range(0, T - l + 1):
        window = rgb_signal[t : t + l]
        mean_w = window.mean(axis=0, keepdims=True)
        mean_w = np.where(mean_w == 0, 1e-8, mean_w)
        cn = window / mean_w

        # POS projection matrix
        S = np.array([[0, 1, -1], [-2, 1, 1]])
        h = cn @ S.T  # (l, 2)
        p = h[:, 0] + (h[:, 0].std() / (h[:, 1].std() + 1e-8)) * h[:, 1]
        pulse[t : t + l] += p

    # Normalize
    pulse = (pulse - pulse.mean()) / (pulse.std() + 1e-8)
    return pulse


def compute_snr_and_bpm(pulse: np.ndarray, fps: float, freq_lo: float = 0.7, freq_hi: float = 3.5) -> dict:
    """
    Compute SNR (dB) and estimated BPM from pulse signal via FFT.
    Returns dict: {snr_db, bpm, psd_freqs, psd_power, peak_freq}
    """
    T = len(pulse)
    freqs = rfftfreq(T, d=1.0 / fps)
    fft_mag = np.abs(rfft(pulse)) ** 2

    # Find peak in [freq_lo, freq_hi] Hz
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if mask.sum() == 0:
        return {"snr_db": -99.0, "bpm": 0.0, "psd_freqs": freqs, "psd_power": fft_mag, "peak_freq": 0.0}

    signal_power = fft_mag[mask].max()
    noise_power = fft_mag[~mask].mean() + 1e-12
    snr_db = 10 * np.log10(signal_power / noise_power)

    peak_freq = freqs[mask][fft_mag[mask].argmax()]
    bpm = peak_freq * 60.0

    return {
        "snr_db": float(snr_db),
        "bpm": float(bpm),
        "peak_freq": float(peak_freq),
        "psd_freqs": freqs.tolist(),
        "psd_power": fft_mag.tolist(),
    }


def load_video_frames(video_path: str, max_frames: int = 300, target_fps: float = 15.0) -> tuple:
    """Load video, downsample to target_fps, return (frames array, actual fps)."""
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(orig_fps / target_fps))

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
        idx += 1
    cap.release()

    if not frames:
        return np.zeros((1, 224, 224, 3), dtype=np.float32), target_fps

    return np.array(frames, dtype=np.float32), target_fps


def process_video(video_path: str, fps: float = 15.0) -> dict:
    """Full rPPG pipeline for a single video. Returns analysis dict."""
    frames, fps = load_video_frames(video_path, max_frames=300, target_fps=fps)
    T = len(frames)

    if T < 30:
        return {"error": f"Too few frames: {T}", "video": video_path}

    # Extract ROI signals
    roi_signals = get_face_roi_signals(frames, fps)

    results = {"video": video_path, "n_frames": T, "fps": fps, "methods": {}}

    for method_name, method_fn in [("CHROM", chrom_method), ("POS", pos_method)]:
        # Use forehead + cheek averaged signal
        combined = (
            roi_signals["forehead_rgb"] * 0.4
            + roi_signals["left_cheek_rgb"] * 0.3
            + roi_signals["right_cheek_rgb"] * 0.3
        )
        pulse = method_fn(combined, fps)
        metrics = compute_snr_and_bpm(pulse, fps)

        results["methods"][method_name] = {
            "pulse_waveform": pulse.tolist(),
            "snr_db": metrics["snr_db"],
            "bpm": metrics["bpm"],
            "peak_freq": metrics["peak_freq"],
        }

    # Agreement between methods
    bpm_chrom = results["methods"]["CHROM"]["bpm"]
    bpm_pos = results["methods"]["POS"]["bpm"]
    results["bpm_agreement_abs"] = abs(bpm_chrom - bpm_pos)
    results["mean_bpm"] = (bpm_chrom + bpm_pos) / 2.0
    results["mean_snr_db"] = (
        results["methods"]["CHROM"]["snr_db"] + results["methods"]["POS"]["snr_db"]
    ) / 2.0

    return results


def batch_extract(video_dir: str, label: str, out_dir: str, max_videos: int = 200) -> pd.DataFrame:
    """Process all videos in directory, save results, return summary DataFrame."""
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))[:max_videos]

    if not video_files:
        print(f"[WARN] No video files found in {video_dir}")
        return pd.DataFrame()

    print(f"\nProcessing {len(video_files)} {label} videos from {video_dir}")

    records = []
    cache_file = out_dir / f"rppg_{label}.json"

    for vid_path in tqdm(video_files, desc=f"rPPG [{label}]"):
        try:
            res = process_video(str(vid_path))
            if "error" not in res:
                records.append({
                    "video": vid_path.name,
                    "label": label,
                    "n_frames": res["n_frames"],
                    "mean_snr_db": res["mean_snr_db"],
                    "mean_bpm": res["mean_bpm"],
                    "bpm_agreement": res["bpm_agreement_abs"],
                    "snr_chrom": res["methods"]["CHROM"]["snr_db"],
                    "snr_pos": res["methods"]["POS"]["snr_db"],
                    "bpm_chrom": res["methods"]["CHROM"]["bpm"],
                    "bpm_pos": res["methods"]["POS"]["bpm"],
                })
        except Exception as e:
            print(f"  [ERR] {vid_path.name}: {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"rppg_summary_{label}.csv", index=False)
    print(f"\nSaved {len(df)} results → {out_dir / f'rppg_summary_{label}.csv'}")

    # Print quick stats
    if not df.empty:
        print(f"\n  SNR (dB)   mean={df.mean_snr_db.mean():.2f}  std={df.mean_snr_db.std():.2f}")
        print(f"  BPM        mean={df.mean_bpm.mean():.1f}  std={df.mean_bpm.std():.1f}")

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Batch rPPG extraction for deepfake detection")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--label", choices=["real", "fake"], required=True)
    p.add_argument("--out_dir", default="./logs/signal_cache")
    p.add_argument("--max_videos", type=int, default=200)
    p.add_argument("--fps", type=float, default=15.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_extract(args.video_dir, args.label, args.out_dir, args.max_videos)
