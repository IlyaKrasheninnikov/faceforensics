"""
Pre-extract rPPG features from FF++ PNG folders.

Reads consecutive PNG frames from each video folder, runs CHROM+POS rPPG
extraction, computes 128-d FFT spectrum feature, and saves as:
    <video_folder>/rppg_feat.npy   — shape (128,) float32
    <video_folder>/rppg_meta.json  — snr_db, bpm, n_frames used

This pre-computation happens once (~2-4 hours for 6000 videos on T4).
After that, PNGClipDataset loads rppg_feat.npy instead of returning zeros.

Usage (Kaggle):
    python w3_train/extract_rppg_png.py \
        --ff_root /kaggle/input/datasets/sheldonhomes/faceforensics-c23-processed/ff/ff++/frames \
        --out_dir /kaggle/working/rppg_cache \
        --fps 25 \
        --max_frames 64 \
        --num_workers 4

    Then train with:
        --rppg_cache /kaggle/working/rppg_cache
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

FF_MANIPULATION_TYPES = ["original", "Deepfakes", "Face2Face", "FaceSwap",
                          "NeuralTextures", "FaceShifter"]

RPPG_FEAT_DIM = 128   # FFT bins kept as feature vector


# ─── rPPG core (inlined from w1_setup/extract_rppg.py, no MediaPipe needed) ──

def _crop_mean_rgb(frame: np.ndarray, box, H: int, W: int) -> np.ndarray:
    x, y, w, h = box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(3, dtype=np.float32)
    return frame[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.float32)


def _face_roi_signals_opencv(frames: np.ndarray) -> np.ndarray:
    """
    Extract combined forehead+cheek mean RGB signal using OpenCV Haar cascade.
    FF++ frames are already face-cropped at 224×224, so Haar usually finds the face.
    Falls back to centered 60% crop if detection fails.

    Returns: (T, 3) float32 RGB signal in [0, 1]
    """
    T, H, W, _ = frames.shape
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    signal = np.zeros((T, 3), dtype=np.float32)
    last_box = None

    for t, frame in enumerate(frames):
        frame_uint8 = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30))

        if len(faces) > 0:
            last_box = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        elif last_box is None:
            cx, cy = W // 2, H // 2
            fw, fh = int(W * 0.6), int(H * 0.6)
            last_box = (cx - fw // 2, cy - fh // 2, fw, fh)

        x, y, w, h = last_box
        forehead  = (x + w // 4,     y,              w // 2, h // 5)
        l_cheek   = (x,              y + h * 2 // 5, w // 3, h // 4)
        r_cheek   = (x + w * 2 // 3, y + h * 2 // 5, w // 3, h // 4)

        sig = (
            _crop_mean_rgb(frame, forehead, H, W) * 0.4 +
            _crop_mean_rgb(frame, l_cheek,  H, W) * 0.3 +
            _crop_mean_rgb(frame, r_cheek,  H, W) * 0.3
        )
        signal[t] = sig

    return signal


def chrom_method(rgb: np.ndarray, fps: float) -> np.ndarray:
    mean_rgb = rgb.mean(axis=0, keepdims=True)
    mean_rgb = np.where(mean_rgb == 0, 1e-8, mean_rgb)
    cn = rgb / mean_rgb
    xs = 3 * cn[:, 0] - 2 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    nyq = fps / 2.0
    if nyq <= 3.5:
        return (xs - ys)
    # Use order=2 so padlen=9; skip filtering entirely if signal still too short
    order = 2
    b, a = scipy_signal.butter(order, [0.7 / nyq, 3.5 / nyq], btype="band")
    padlen = 3 * order + 1  # scipy default: 3*(max(len(a),len(b))-1)
    if len(xs) <= padlen:
        alpha = xs.std() / (ys.std() + 1e-8)
        return xs - alpha * ys
    xs_f = scipy_signal.filtfilt(b, a, xs)
    ys_f = scipy_signal.filtfilt(b, a, ys)
    alpha = xs_f.std() / (ys_f.std() + 1e-8)
    return xs_f - alpha * ys_f


def pos_method(rgb: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    T = len(rgb)
    l = max(2, int(fps * window_sec))
    pulse = np.zeros(T, dtype=np.float32)
    for t in range(0, T - l + 1):
        w = rgb[t:t + l]
        m = w.mean(axis=0, keepdims=True)
        m = np.where(m == 0, 1e-8, m)
        cn = w / m
        S = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
        h = cn @ S.T
        p = h[:, 0] + (h[:, 0].std() / (h[:, 1].std() + 1e-8)) * h[:, 1]
        pulse[t:t + l] += p
    std = pulse.std()
    if std > 1e-8:
        pulse = (pulse - pulse.mean()) / std
    return pulse


def pulse_to_fft_feature(pulse: np.ndarray, fps: float,
                          n_bins: int = RPPG_FEAT_DIM) -> np.ndarray:
    """
    Compute normalized FFT magnitude spectrum → fixed-size feature vector.
    Only keeps the physiologically relevant 0–4 Hz range, resampled to n_bins.
    """
    T = len(pulse)
    freqs = rfftfreq(T, d=1.0 / fps)
    fft_mag = np.abs(rfft(pulse)).astype(np.float32)

    # Keep only 0–4 Hz (covers resting HR 0.7–3.5 Hz with margin)
    mask = freqs <= 4.0
    freqs_crop = freqs[mask]
    fft_crop   = fft_mag[mask]

    if len(fft_crop) < 2:
        return np.zeros(n_bins, dtype=np.float32)

    # Resample to fixed n_bins via linear interpolation
    x_old = np.linspace(0, 1, len(fft_crop))
    x_new = np.linspace(0, 1, n_bins)
    feat = np.interp(x_new, x_old, fft_crop).astype(np.float32)

    # L2-normalize so scale doesn't matter
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat /= norm

    return feat


def hybrid_rppg_feature(rgb_signal: np.ndarray, fps: float) -> tuple:
    """
    Compute hybrid rPPG feature: average of CHROM and POS FFT features.
    Returns (feat_128d, snr_db, bpm).
    """
    pulse_chrom = chrom_method(rgb_signal, fps)
    pulse_pos   = pos_method(rgb_signal, fps)

    feat_chrom = pulse_to_fft_feature(pulse_chrom, fps)
    feat_pos   = pulse_to_fft_feature(pulse_pos, fps)
    feat = (feat_chrom + feat_pos) / 2.0
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat /= norm

    # SNR from CHROM pulse
    T = len(pulse_chrom)
    freqs = rfftfreq(T, d=1.0 / fps)
    fft_mag = np.abs(rfft(pulse_chrom)) ** 2
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    if hr_mask.sum() > 0 and fft_mag[hr_mask].max() > 1e-12:
        sig_power   = fft_mag[hr_mask].max()
        noise_power = fft_mag[~hr_mask].mean() + 1e-12
        snr_db = float(10 * np.log10(sig_power / noise_power))
        peak_freq = float(freqs[hr_mask][fft_mag[hr_mask].argmax()])
        bpm = peak_freq * 60.0
    else:
        snr_db, bpm = -99.0, 0.0

    return feat.astype(np.float32), snr_db, bpm


# ─── Per-video worker ─────────────────────────────────────────────────────────

def process_video_folder(args_tuple) -> dict:
    """
    Worker function: load PNGs from video_dir, extract rPPG feature, save to out_dir.

    Feature is saved at: out_dir/<manip>/<video_name>/rppg_feat.npy
    This mirrors the source structure so PNGClipDataset can find it by path.
    out_dir is writable (/kaggle/working/rppg_cache); source video_dir may be read-only.
    """
    video_dir, out_dir, fps, max_frames, force_recompute = args_tuple
    video_dir = Path(video_dir)
    out_dir   = Path(out_dir)

    # Mirror structure: out_dir / manip / video_name /
    save_dir  = out_dir / video_dir.parent.name / video_dir.name
    feat_path = save_dir / "rppg_feat.npy"
    meta_path = save_dir / "rppg_meta.json"

    if not force_recompute and feat_path.exists() and meta_path.exists():
        return {"video": video_dir.name, "status": "cached"}

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load frames
    frame_files = sorted([f for f in os.listdir(video_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not frame_files:
        return {"video": video_dir.name, "status": "no_frames"}

    # Subsample to max_frames evenly
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames).astype(int)
        frame_files = [frame_files[i] for i in indices]

    frames = []
    for f in frame_files:
        img = cv2.imread(str(video_dir / f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        frames.append(img.astype(np.float32) / 255.0)

    if len(frames) < 16:
        np.save(feat_path, np.zeros(RPPG_FEAT_DIM, dtype=np.float32))
        with open(meta_path, "w") as f:
            json.dump({"snr_db": -99.0, "bpm": 0.0, "n_frames": len(frames),
                       "status": "too_few_frames"}, f)
        return {"video": video_dir.name, "status": "too_few_frames"}

    frames_arr = np.stack(frames, axis=0)  # (T, H, W, 3)

    try:
        rgb_signal = _face_roi_signals_opencv(frames_arr)
        feat, snr_db, bpm = hybrid_rppg_feature(rgb_signal, fps)
    except Exception as e:
        feat = np.zeros(RPPG_FEAT_DIM, dtype=np.float32)
        snr_db, bpm = -99.0, 0.0
        return {"video": video_dir.name, "status": f"error: {type(e).__name__}: {e}"}

    np.save(feat_path, feat)
    with open(meta_path, "w") as f:
        json.dump({"snr_db": snr_db, "bpm": bpm, "n_frames": len(frames)}, f)

    return {"video": video_dir.name, "status": "ok", "snr_db": snr_db, "bpm": bpm}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    ff_root = Path(args.ff_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video folders
    all_folders = []
    for manip in FF_MANIPULATION_TYPES:
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            print(f"  {manip}: NOT FOUND")
            continue
        subdirs = [d for d in manip_dir.iterdir() if d.is_dir()]
        all_folders.extend(subdirs)
        print(f"  {manip}: {len(subdirs)} folders")

    print(f"\nTotal: {len(all_folders)} video folders")

    # Build work list — save features into out_dir (source may be read-only)
    work = [(str(d), str(out_dir), args.fps, args.max_frames, args.force)
            for d in all_folders]

    start = time.time()
    n_ok = n_cached = n_err = 0
    first_error = None

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(process_video_folder, w): w for w in work}
            pbar = tqdm(as_completed(futures), total=len(work), desc="rPPG extract")
            for fut in pbar:
                r = fut.result()
                if r["status"] == "ok":
                    n_ok += 1
                elif r["status"] == "cached":
                    n_cached += 1
                else:
                    n_err += 1
                    if first_error is None:
                        first_error = r["status"]
                pbar.set_postfix(ok=n_ok, cached=n_cached, err=n_err)
    else:
        for w in tqdm(work, desc="rPPG extract"):
            r = process_video_folder(w)
            if r["status"] == "ok":
                n_ok += 1
            elif r["status"] == "cached":
                n_cached += 1
            else:
                n_err += 1
                if first_error is None:
                    first_error = r["status"]

    elapsed = time.time() - start
    print(f"\nDone: {n_ok} extracted, {n_cached} cached, {n_err} errors")
    if first_error:
        print(f"First error: {first_error}")
    print(f"Time: {elapsed / 60:.1f} min")
    print(f"Features saved to: {out_dir}/<manip>/<video_name>/rppg_feat.npy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ff_root",     required=True)
    p.add_argument("--out_dir",     default="./rppg_cache",
                   help="Unused (features saved in video folders), kept for compat")
    p.add_argument("--fps",         type=float, default=25.0,
                   help="Assumed FPS of the source video (FF++ is 25fps)")
    p.add_argument("--max_frames",  type=int,   default=64,
                   help="Max frames to use per video for rPPG extraction")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--force",       action="store_true",
                   help="Recompute even if rppg_feat.npy already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
