"""
V2 rPPG feature extraction: Left/Right cheek synchronization features.

Key insight:
  - Real faces: rPPG signals from left & right cheeks are highly correlated (PCC ~0.72)
  - Fake faces: rPPG signals from left & right cheeks are uncorrelated (PCC ~0.08)
  - Single pooled FFT spectrum (v1) looks IDENTICAL for real/fake — zero discrimination.

This script extracts LEFT and RIGHT cheek rPPG signals separately, then computes:
  - Per-cheek: SNR, PSD_mean, MAD, SD  (4 × 2 = 8 features)
  - Cross-cheek: PCC, max_cross_correlation, SNR_diff, SD_ratio  (4 features)
  Total: 12 scalar features per video

Output: <out_dir>/<manip>/<video_name>/rppg_v2_feat.npy  — shape (12,) float32
        <out_dir>/<manip>/<video_name>/rppg_v2_meta.json — all metrics for analysis

Usage (Kaggle):
    python w3_train/extract_rppg_v2_png.py \\
        --ff_root /kaggle/input/ff-c23-processed/FaceForensics++_c23_processed \\
        --out_dir /kaggle/working/rppg_v2_cache \\
        --fps 25 --max_frames 64 --num_workers 4
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

RPPG_V2_FEAT_DIM = 12  # total feature dimension


# ─── Face ROI: separate left/right cheek green channel signals ──────────────

def _extract_lr_cheek_green(frames: np.ndarray) -> tuple:
    """
    Extract GREEN channel mean from LEFT and RIGHT cheek ROIs separately.

    Uses Haar cascade on NATIVE resolution frames (1280×720 in FF++).
    Detects face in first frame, then uses fixed ROI for temporal consistency.
    Cheek ROIs at native resolution are ~100px wide → much cleaner signal.

    Returns: (left_green, right_green) each shape (T,) float32
    """
    T, H, W, _ = frames.shape
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    # Detect face on first few frames, pick the largest stable detection
    face_box = None
    for t in range(min(5, T)):
        frame_uint8 = (frames[t] * 255).astype(np.uint8)
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(50, 50))
        if len(faces) > 0:
            face_box = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            break

    if face_box is None:
        # Fallback: assume face is centered (common for FF++ preprocessed frames)
        cx, cy = W // 2, H // 4  # face typically in upper portion
        fw, fh = int(W * 0.4), int(H * 0.5)
        face_box = (cx - fw // 2, cy, fw, fh)

    x, y, w, h = face_box

    # Fixed cheek ROIs (consistent across all frames → clean temporal signal)
    # Left cheek: left 30% of face, vertically at 40-65% of face height
    lc_x1 = max(0, x + int(w * 0.05))
    lc_x2 = max(lc_x1 + 5, x + int(w * 0.35))
    lc_y1 = max(0, y + int(h * 0.40))
    lc_y2 = max(lc_y1 + 5, y + int(h * 0.65))

    # Right cheek: right 30% of face, vertically at 40-65% of face height
    rc_x1 = max(0, x + int(w * 0.65))
    rc_x2 = max(rc_x1 + 5, min(W, x + int(w * 0.95)))
    rc_y1 = max(0, y + int(h * 0.40))
    rc_y2 = max(rc_y1 + 5, y + int(h * 0.65))

    left_signal = np.zeros(T, dtype=np.float32)
    right_signal = np.zeros(T, dtype=np.float32)

    for t in range(T):
        frame = frames[t]
        # Green channel mean over cheek ROI
        left_signal[t] = frame[lc_y1:lc_y2, lc_x1:lc_x2, 1].mean()
        right_signal[t] = frame[rc_y1:rc_y2, rc_x1:rc_x2, 1].mean()

    return left_signal, right_signal


# ─── Signal processing ──────────────────────────────────────────────────────

def _detrend(signal: np.ndarray, fps: float) -> np.ndarray:
    """Remove low-frequency drift using moving average subtraction."""
    if len(signal) < 3:
        return signal
    
    win = max(2, min(int(fps), len(signal) // 2))  # Ensure window isn't too large
    if win >= len(signal):
        win = max(2, len(signal) // 3)
    
    kernel = np.ones(win) / win
    
    # Use 'same' mode but ensure output length matches input
    trend = np.convolve(signal, kernel, mode='same')
    
    # Ensure shapes match (convolve with 'same' can sometimes produce off-by-one)
    if len(trend) != len(signal):
        # Trim or pad to match input length
        if len(trend) > len(signal):
            trend = trend[:len(signal)]
        else:
            # Pad with edge values if needed (shouldn't happen with 'same')
            trend = np.pad(trend, (0, len(signal) - len(trend)), 'edge')
    
    return signal - trend


def _bandpass_filter(signal: np.ndarray, fps: float,
                     low_hz: float = 0.7, high_hz: float = 3.5) -> np.ndarray:
    """Butterworth bandpass filter for heart rate range (42-210 BPM).
    For very short signals (<30 frames), use order=1 to minimize padlen requirement."""
    nyq = fps / 2.0
    if nyq <= high_hz or len(signal) < 8:
        return signal
    # Use order=1 for short signals (padlen=4), order=2 for longer ones (padlen=9)
    order = 1 if len(signal) < 30 else 2
    b, a = scipy_signal.butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        return signal
    return scipy_signal.filtfilt(b, a, signal).astype(np.float32)


def _compute_snr(signal: np.ndarray, fps: float) -> float:
    """Signal-to-noise ratio: peak HR power vs average noise power."""
    if len(signal) < 4:
        return -99.0
    freqs = rfftfreq(len(signal), d=1.0 / fps)
    fft_power = np.abs(rfft(signal)) ** 2
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    if hr_mask.sum() == 0:
        return -99.0
    sig_power = fft_power[hr_mask].max()
    noise_mask = ~hr_mask & (freqs > 0)  # exclude DC
    noise_power = fft_power[noise_mask].mean() + 1e-12
    return float(10 * np.log10(sig_power / noise_power + 1e-12))


def _compute_psd_mean(signal: np.ndarray, fps: float) -> float:
    """Mean power spectral density in HR range using Welch's method."""
    if len(signal) < 8:
        return 0.0
    nperseg = min(len(signal), 256)
    f, psd = scipy_signal.welch(signal, fs=fps, nperseg=nperseg)
    hr_mask = (f >= 0.7) & (f <= 3.5)
    if hr_mask.sum() == 0:
        return 0.0
    return float(psd[hr_mask].mean())


def compute_sync_features(left_raw: np.ndarray, right_raw: np.ndarray,
                           fps: float) -> tuple:
    """
    Compute left/right cheek synchronization features.

    Returns: (feat_12d, meta_dict)
      feat_12d: np.array shape (12,) float32
      meta_dict: dict with all individual metrics
    """
    # Step 1: Detrend (remove illumination drift)
    left_dt = _detrend(left_raw, fps)
    right_dt = _detrend(right_raw, fps)

    # Step 2: Bandpass filter (0.7-3.5 Hz = 42-210 BPM)
    left_f = _bandpass_filter(left_dt, fps)
    right_f = _bandpass_filter(right_dt, fps)

    # Step 3: Per-cheek metrics (on filtered signals)
    left_snr = _compute_snr(left_f, fps)
    right_snr = _compute_snr(right_f, fps)
    left_psd = _compute_psd_mean(left_f, fps)
    right_psd = _compute_psd_mean(right_f, fps)
    left_sd = float(left_f.std())
    right_sd = float(right_f.std())

    # Step 4: Cross-cheek synchronization — compute on BOTH raw (detrended)
    # and filtered signals, take the more informative one.
    # For 24-frame signals, raw PCC is more reliable than filtered PCC.

    # PCC on detrended (not filtered) signal — works even with 24 frames
    pcc_raw = 0.0
    if left_dt.std() > 1e-8 and right_dt.std() > 1e-8:
        pcc_raw = float(np.corrcoef(left_dt, right_dt)[0, 1])
    if np.isnan(pcc_raw):
        pcc_raw = 0.0

    # PCC on filtered signal
    pcc_filt = 0.0
    if left_f.std() > 1e-8 and right_f.std() > 1e-8:
        pcc_filt = float(np.corrcoef(left_f, right_f)[0, 1])
    if np.isnan(pcc_filt):
        pcc_filt = 0.0

    # Max cross-correlation (on detrended — more robust for short signals)
    max_xcorr = 0.0
    if left_dt.std() > 1e-8 and right_dt.std() > 1e-8:
        left_norm = (left_dt - left_dt.mean()) / (left_dt.std() * len(left_dt))
        right_norm = (right_dt - right_dt.mean()) / right_dt.std()
        xcorr = np.correlate(left_norm, right_norm, mode='full')
        max_xcorr = float(xcorr.max())
    if np.isnan(max_xcorr):
        max_xcorr = 0.0

    # Asymmetry features
    snr_diff = abs(left_snr - right_snr)
    sd_ratio = min(left_sd, right_sd) / (max(left_sd, right_sd) + 1e-8)

    # Build feature vector: 12 dimensions
    feat = np.array([
        left_snr, right_snr, snr_diff,       # 0-2: per-cheek SNR + asymmetry
        left_sd, right_sd, sd_ratio,          # 3-5: per-cheek SD + ratio
        pcc_raw,                              # 6: PCC on detrended signal (KEY!)
        pcc_filt,                             # 7: PCC on filtered signal
        max_xcorr,                            # 8: max cross-correlation
        left_psd, right_psd,                  # 9-10: per-cheek PSD
        abs(pcc_raw - pcc_filt),              # 11: PCC consistency (real should be stable)
    ], dtype=np.float32)

    # Replace any NaN/inf
    feat = np.nan_to_num(feat, nan=0.0, posinf=50.0, neginf=-50.0)

    meta = {
        "left_snr": left_snr, "right_snr": right_snr, "snr_diff": snr_diff,
        "left_sd": left_sd, "right_sd": right_sd, "sd_ratio": sd_ratio,
        "pcc_raw": pcc_raw, "pcc_filt": pcc_filt,
        "max_xcorr": max_xcorr,
        "left_psd": left_psd, "right_psd": right_psd,
        "pcc": pcc_raw,  # backward compat: "pcc" key used by diagnostic scripts
    }

    return feat, meta


# ─── Per-video worker ──────────────────────────────────────────────────────

def process_video_folder(args_tuple) -> dict:
    """
    Worker: load PNGs, extract L/R cheek rPPG, compute sync features, save.
    """
    video_dir, out_dir, fps, max_frames, force_recompute = args_tuple
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)

    save_dir = out_dir / video_dir.parent.name / video_dir.name
    feat_path = save_dir / "rppg_v2_feat.npy"
    meta_path = save_dir / "rppg_v2_meta.json"

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
        # Keep NATIVE resolution for rPPG — cheek ROIs need to be large
        # enough (100+ px) for clean green channel signal
        frames.append(img.astype(np.float32) / 255.0)

    if len(frames) < 8:
        np.save(feat_path, np.zeros(RPPG_V2_FEAT_DIM, dtype=np.float32))
        with open(meta_path, "w") as f:
            json.dump({"status": "too_few_frames", "n_frames": len(frames)}, f)
        return {"video": video_dir.name, "status": "too_few_frames"}

    frames_arr = np.stack(frames, axis=0)  # (T, H, W, 3) RGB

    try:
        left_green, right_green = _extract_lr_cheek_green(frames_arr)
        feat, meta = compute_sync_features(left_green, right_green, fps)
        meta["n_frames"] = len(frames)
        meta["status"] = "ok"
    except Exception as e:
        feat = np.zeros(RPPG_V2_FEAT_DIM, dtype=np.float32)
        meta = {"status": f"error: {type(e).__name__}: {e}", "n_frames": len(frames)}

    np.save(feat_path, feat)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return {
        "video": video_dir.name,
        "status": meta.get("status", "ok"),
        "pcc": meta.get("pcc", 0.0),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    ff_root = Path(args.ff_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    work = [(str(d), str(out_dir), args.fps, args.max_frames, args.force)
            for d in all_folders]

    start = time.time()
    n_ok = n_cached = n_err = 0
    pcc_real, pcc_fake = [], []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(process_video_folder, w): w for w in work}
            pbar = tqdm(as_completed(futures), total=len(work), desc="rPPG v2 extract")
            for fut in pbar:
                r = fut.result()
                if r["status"] == "ok":
                    n_ok += 1
                    # Track PCC for real vs fake summary
                    folder_path = futures[fut][0]
                    manip = Path(folder_path).parent.name
                    if manip == "original":
                        pcc_real.append(r.get("pcc", 0))
                    else:
                        pcc_fake.append(r.get("pcc", 0))
                elif r["status"] == "cached":
                    n_cached += 1
                else:
                    n_err += 1
                pbar.set_postfix(ok=n_ok, cached=n_cached, err=n_err)
    else:
        for w in tqdm(work, desc="rPPG v2 extract"):
            r = process_video_folder(w)
            if r["status"] == "ok":
                n_ok += 1
                manip = Path(w[0]).parent.name
                if manip == "original":
                    pcc_real.append(r.get("pcc", 0))
                else:
                    pcc_fake.append(r.get("pcc", 0))
            elif r["status"] == "cached":
                n_cached += 1
            else:
                n_err += 1

    elapsed = time.time() - start
    print(f"\nDone: {n_ok} extracted, {n_cached} cached, {n_err} errors")
    print(f"Time: {elapsed / 60:.1f} min")

    # Summary: confirm PCC is discriminative
    if pcc_real and pcc_fake:
        print(f"\n=== PCC DISCRIMINATION CHECK ===")
        print(f"Real PCC: {np.mean(pcc_real):.3f} ± {np.std(pcc_real):.3f}  (n={len(pcc_real)})")
        print(f"Fake PCC: {np.mean(pcc_fake):.3f} ± {np.std(pcc_fake):.3f}  (n={len(pcc_fake)})")
        print(f"Expected: real >> fake (e.g., 0.7 vs 0.1)")
        if np.mean(pcc_real) > np.mean(pcc_fake) + 0.1:
            print("✓ PCC shows clear real/fake separation!")
        else:
            print("⚠ PCC separation is weak — check ROI extraction quality")

    print(f"\nFeatures saved to: {out_dir}/<manip>/<video_name>/rppg_v2_feat.npy")


def parse_args():
    p = argparse.ArgumentParser(description="V2 rPPG extraction: L/R cheek sync features")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--out_dir", default="./rppg_v2_cache")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
