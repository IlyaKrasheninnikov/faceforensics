"""
V2 rPPG feature extraction from FULL MP4 VIDEOS.

Key difference from extract_rppg_v2_png.py:
  - Reads MP4 videos directly (not PNG folders)
  - Works with xdxd003/ff-c23 dataset (full-length videos, ~300 frames)
  - With 300+ frames we get 10+ heartbeat cycles → discriminative PCC
  - Expected: real PCC ~0.72, fake PCC ~0.08 (per Sync_rPPG paper)

Dataset structure (xdxd003/ff-c23 on Kaggle):
    <ff_root>/original/c23/videos/000.mp4, 001.mp4, ...
    <ff_root>/Deepfakes/c23/videos/000_003.mp4, ...
    <ff_root>/Face2Face/c23/videos/000_003.mp4, ...
    <ff_root>/FaceSwap/c23/videos/000_003.mp4, ...
    <ff_root>/NeuralTextures/c23/videos/000_003.mp4, ...
    <ff_root>/FaceShifter/c23/videos/000_003.mp4, ...

  OR flat layout:
    <ff_root>/original/000.mp4, 001.mp4, ...
    <ff_root>/Deepfakes/000_003.mp4, ...

Output (compatible with train_physio_png.py --rppg_cache):
    <out_dir>/<manip>/<video_stem>/rppg_v2_feat.npy  — shape (12,) float32
    <out_dir>/<manip>/<video_stem>/rppg_v2_meta.json  — all metrics

Usage (Kaggle):
    python w3_train/extract_rppg_v2_mp4.py \\
        --ff_root /kaggle/input/ff-c23 \\
        --out_dir /kaggle/working/rppg_v2_cache \\
        --target_fps 25 --max_frames 300 --num_workers 4
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


# ─── Video loading ─────────────────────────────────────────────────────────

def load_video_frames(video_path: str, max_frames: int = 300,
                      target_fps: float = 25.0) -> tuple:
    """
    Load MP4 video, downsample to target_fps, return frames at native resolution.

    Returns: (frames_array, actual_fps)
      frames_array: np.ndarray shape (T, H, W, 3) float32 [0,1] RGB
      actual_fps: float — the effective fps after downsampling
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Downsample: keep every step-th frame to approximate target_fps
    step = max(1, round(orig_fps / target_fps))
    actual_fps = orig_fps / step

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # Keep NATIVE resolution — cheek ROIs need to be large
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
        idx += 1
    cap.release()

    if len(frames) < 8:
        return None, actual_fps

    return np.stack(frames, axis=0), actual_fps


# ─── Face ROI: separate left/right cheek green channel signals ──────────────

def _extract_lr_cheek_green(frames: np.ndarray) -> tuple:
    """
    Extract GREEN channel mean from LEFT and RIGHT cheek ROIs separately.

    Uses Haar cascade on native resolution frames.
    Detects face in first 5 frames, uses fixed ROI for temporal consistency.

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
        # Fallback: assume face is centered
        cx, cy = W // 2, H // 4
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
        left_signal[t] = frame[lc_y1:lc_y2, lc_x1:lc_x2, 1].mean()
        right_signal[t] = frame[rc_y1:rc_y2, rc_x1:rc_x2, 1].mean()

    return left_signal, right_signal


# ─── Signal processing ──────────────────────────────────────────────────────

def _detrend(signal: np.ndarray, fps: float) -> np.ndarray:
    """Remove low-frequency drift using moving average subtraction."""
    if len(signal) < 3:
        return signal

    win = max(2, min(int(fps), len(signal) // 2))
    if win >= len(signal):
        win = max(2, len(signal) // 3)

    kernel = np.ones(win) / win
    trend = np.convolve(signal, kernel, mode='same')

    if len(trend) != len(signal):
        if len(trend) > len(signal):
            trend = trend[:len(signal)]
        else:
            trend = np.pad(trend, (0, len(signal) - len(trend)), 'edge')

    return signal - trend


def _bandpass_filter(signal: np.ndarray, fps: float,
                     low_hz: float = 0.7, high_hz: float = 3.5) -> np.ndarray:
    """Butterworth bandpass filter for heart rate range (42-210 BPM)."""
    nyq = fps / 2.0
    if nyq <= high_hz or len(signal) < 8:
        return signal
    # order=2 is fine for long signals (300+ frames)
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
    noise_mask = ~hr_mask & (freqs > 0)
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
    """
    left_dt = _detrend(left_raw, fps)
    right_dt = _detrend(right_raw, fps)

    left_f = _bandpass_filter(left_dt, fps)
    right_f = _bandpass_filter(right_dt, fps)

    # Per-cheek metrics
    left_snr = _compute_snr(left_f, fps)
    right_snr = _compute_snr(right_f, fps)
    left_psd = _compute_psd_mean(left_f, fps)
    right_psd = _compute_psd_mean(right_f, fps)
    left_sd = float(left_f.std())
    right_sd = float(right_f.std())

    # Cross-cheek synchronization
    pcc_raw = 0.0
    if left_dt.std() > 1e-8 and right_dt.std() > 1e-8:
        pcc_raw = float(np.corrcoef(left_dt, right_dt)[0, 1])
    if np.isnan(pcc_raw):
        pcc_raw = 0.0

    pcc_filt = 0.0
    if left_f.std() > 1e-8 and right_f.std() > 1e-8:
        pcc_filt = float(np.corrcoef(left_f, right_f)[0, 1])
    if np.isnan(pcc_filt):
        pcc_filt = 0.0

    max_xcorr = 0.0
    if left_dt.std() > 1e-8 and right_dt.std() > 1e-8:
        left_norm = (left_dt - left_dt.mean()) / (left_dt.std() * len(left_dt))
        right_norm = (right_dt - right_dt.mean()) / right_dt.std()
        xcorr = np.correlate(left_norm, right_norm, mode='full')
        max_xcorr = float(xcorr.max())
    if np.isnan(max_xcorr):
        max_xcorr = 0.0

    snr_diff = abs(left_snr - right_snr)
    sd_ratio = min(left_sd, right_sd) / (max(left_sd, right_sd) + 1e-8)

    feat = np.array([
        left_snr, right_snr, snr_diff,
        left_sd, right_sd, sd_ratio,
        pcc_raw,
        pcc_filt,
        max_xcorr,
        left_psd, right_psd,
        abs(pcc_raw - pcc_filt),
    ], dtype=np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=50.0, neginf=-50.0)

    meta = {
        "left_snr": left_snr, "right_snr": right_snr, "snr_diff": snr_diff,
        "left_sd": left_sd, "right_sd": right_sd, "sd_ratio": sd_ratio,
        "pcc_raw": pcc_raw, "pcc_filt": pcc_filt,
        "max_xcorr": max_xcorr,
        "left_psd": left_psd, "right_psd": right_psd,
        "pcc": pcc_raw,
    }

    return feat, meta


# ─── Per-video worker ──────────────────────────────────────────────────────

def process_video_mp4(args_tuple) -> dict:
    """
    Worker: load MP4 video, extract L/R cheek rPPG, compute sync features, save.
    """
    video_path, out_dir, target_fps, max_frames, force_recompute = args_tuple
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    # Determine manipulation type from parent directory
    # Handles both: .../Deepfakes/c23/videos/000_003.mp4  → manip = "Deepfakes"
    #           and: .../Deepfakes/000_003.mp4            → manip = "Deepfakes"
    manip = _get_manip_type(video_path)
    video_stem = video_path.stem  # e.g., "000_003"

    save_dir = out_dir / manip / video_stem
    feat_path = save_dir / "rppg_v2_feat.npy"
    meta_path = save_dir / "rppg_v2_meta.json"

    if not force_recompute and feat_path.exists() and meta_path.exists():
        return {"video": video_stem, "manip": manip, "status": "cached"}

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load video frames at native resolution
    frames_arr, actual_fps = load_video_frames(
        str(video_path), max_frames=max_frames, target_fps=target_fps
    )

    if frames_arr is None or len(frames_arr) < 8:
        np.save(feat_path, np.zeros(RPPG_V2_FEAT_DIM, dtype=np.float32))
        n = 0 if frames_arr is None else len(frames_arr)
        with open(meta_path, "w") as f:
            json.dump({"status": "too_few_frames", "n_frames": n}, f)
        return {"video": video_stem, "manip": manip, "status": "too_few_frames"}

    try:
        left_green, right_green = _extract_lr_cheek_green(frames_arr)
        feat, meta = compute_sync_features(left_green, right_green, actual_fps)
        meta["n_frames"] = len(frames_arr)
        meta["actual_fps"] = actual_fps
        meta["status"] = "ok"
    except Exception as e:
        feat = np.zeros(RPPG_V2_FEAT_DIM, dtype=np.float32)
        meta = {"status": f"error: {type(e).__name__}: {e}",
                "n_frames": len(frames_arr)}

    np.save(feat_path, feat)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return {
        "video": video_stem,
        "manip": manip,
        "status": meta.get("status", "ok"),
        "pcc": meta.get("pcc", 0.0),
        "n_frames": meta.get("n_frames", 0),
    }


def _get_manip_type(video_path: Path) -> str:
    """
    Extract manipulation type from path. Handles:
      .../Deepfakes/c23/videos/000_003.mp4  → "Deepfakes"
      .../Deepfakes/000_003.mp4             → "Deepfakes"
      .../original/c23/videos/000.mp4       → "original"
    """
    parts = video_path.parts
    for i, part in enumerate(parts):
        if part in FF_MANIPULATION_TYPES:
            return part
    # Fallback: use grandparent or parent name
    return video_path.parent.name


# ─── Discover videos ────────────────────────────────────────────────────────

def find_all_videos(ff_root: Path) -> list:
    """
    Find all MP4 videos in the FF++ dataset. Supports multiple layouts:
      1. <root>/<manip>/c23/videos/*.mp4   (standard FF++ layout)
      2. <root>/<manip>/*.mp4              (flat layout)
      3. <root>/<manip>/videos/*.mp4       (alternative)
    """
    all_videos = []
    for manip in FF_MANIPULATION_TYPES:
        manip_dir = ff_root / manip
        if not manip_dir.exists():
            print(f"  {manip}: NOT FOUND at {manip_dir}")
            continue

        # Try all possible layouts
        candidates = [
            manip_dir,                              # flat: <manip>/*.mp4
            manip_dir / "c23" / "videos",           # standard: <manip>/c23/videos/*.mp4
            manip_dir / "c23",                      # alt: <manip>/c23/*.mp4
            manip_dir / "videos",                   # alt: <manip>/videos/*.mp4
        ]

        found = []
        for cand in candidates:
            if cand.exists():
                vids = sorted(list(cand.glob("*.mp4")))
                if vids:
                    found = vids
                    print(f"  {manip}: {len(vids)} videos ({cand})")
                    break

        if not found:
            # Recurse one more level
            vids = sorted(list(manip_dir.rglob("*.mp4")))
            if vids:
                found = vids
                print(f"  {manip}: {len(vids)} videos (recursive under {manip_dir})")
            else:
                print(f"  {manip}: 0 videos found")

        all_videos.extend(found)

    return all_videos


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    ff_root = Path(args.ff_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for MP4 videos in: {ff_root}")
    all_videos = find_all_videos(ff_root)
    print(f"\nTotal: {len(all_videos)} MP4 videos")

    if not all_videos:
        print("ERROR: No MP4 videos found. Check --ff_root path.")
        print("Expected structure: <ff_root>/<manip>/c23/videos/*.mp4")
        print("              or:   <ff_root>/<manip>/*.mp4")
        return

    # Show sample paths for verification
    print(f"\nSample paths:")
    for v in all_videos[:3]:
        print(f"  {v}")
    if len(all_videos) > 3:
        print(f"  ... ({len(all_videos) - 3} more)")

    work = [(str(v), str(out_dir), args.target_fps, args.max_frames, args.force)
            for v in all_videos]

    start = time.time()
    n_ok = n_cached = n_err = 0
    pcc_real, pcc_fake = [], []
    frame_counts = []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(process_video_mp4, w): w for w in work}
            pbar = tqdm(as_completed(futures), total=len(work), desc="rPPG v2 MP4")
            for fut in pbar:
                r = fut.result()
                if r["status"] == "ok":
                    n_ok += 1
                    manip = r.get("manip", "")
                    if manip == "original":
                        pcc_real.append(r.get("pcc", 0))
                    else:
                        pcc_fake.append(r.get("pcc", 0))
                    frame_counts.append(r.get("n_frames", 0))
                elif r["status"] == "cached":
                    n_cached += 1
                else:
                    n_err += 1
                pbar.set_postfix(ok=n_ok, cached=n_cached, err=n_err)
    else:
        for w in tqdm(work, desc="rPPG v2 MP4"):
            r = process_video_mp4(w)
            if r["status"] == "ok":
                n_ok += 1
                manip = r.get("manip", "")
                if manip == "original":
                    pcc_real.append(r.get("pcc", 0))
                else:
                    pcc_fake.append(r.get("pcc", 0))
                frame_counts.append(r.get("n_frames", 0))
            elif r["status"] == "cached":
                n_cached += 1
            else:
                n_err += 1

    elapsed = time.time() - start
    print(f"\nDone: {n_ok} extracted, {n_cached} cached, {n_err} errors")
    print(f"Time: {elapsed / 60:.1f} min")

    if frame_counts:
        print(f"\nFrame counts: mean={np.mean(frame_counts):.0f}, "
              f"min={np.min(frame_counts)}, max={np.max(frame_counts)}")

    # Summary: confirm PCC is discriminative
    if pcc_real and pcc_fake:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(pcc_real, pcc_fake)
        print(f"\n{'='*50}")
        print(f"  PCC DISCRIMINATION CHECK (full MP4 videos)")
        print(f"{'='*50}")
        print(f"  Real PCC: {np.mean(pcc_real):.3f} +/- {np.std(pcc_real):.3f}  (n={len(pcc_real)})")
        print(f"  Fake PCC: {np.mean(pcc_fake):.3f} +/- {np.std(pcc_fake):.3f}  (n={len(pcc_fake)})")
        print(f"  KS stat:  {ks_stat:.3f}  p={ks_p:.2e}")
        print(f"  Expected: real ~0.72, fake ~0.08 (per Sync_rPPG paper)")
        if ks_p < 0.05:
            print(f"  >>> SIGNIFICANT (p < 0.05) — rPPG v2 features ARE discriminative!")
        else:
            print(f"  >>> NOT significant — check extraction quality")
        print(f"{'='*50}")

    print(f"\nFeatures saved to: {out_dir}/<manip>/<video_stem>/rppg_v2_feat.npy")
    print(f"Compatible with: train_physio_png.py --rppg_cache {out_dir} --rppg_version 2 --rppg_dim 12")


def parse_args():
    p = argparse.ArgumentParser(
        description="V2 rPPG extraction from MP4 videos: L/R cheek sync features")
    p.add_argument("--ff_root", required=True,
                   help="Root of FF++ MP4 dataset (e.g., /kaggle/input/ff-c23)")
    p.add_argument("--out_dir", default="./rppg_v2_cache",
                   help="Output cache directory")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Downsample video to this FPS (default 25)")
    p.add_argument("--max_frames", type=int, default=300,
                   help="Max frames per video (default 300 = ~12s at 25fps)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Parallel workers for extraction")
    p.add_argument("--force", action="store_true",
                   help="Force recompute even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
