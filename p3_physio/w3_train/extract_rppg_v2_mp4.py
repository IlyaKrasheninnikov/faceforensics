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
                      target_fps: float = 25.0,
                      max_height: int = 480) -> tuple:
    """
    Load MP4 video, downsample to target_fps, resize to max_height (480p).

    Why 480p is enough:
      - Native FF++ is 1080p → face ~400px wide → cheek ROI ~120px
      - At 480p → face ~200px wide → cheek ROI ~60px
      - Green channel mean over 60×50px region is very stable for rPPG
      - Sync_rPPG paper used similar resolution with dlib landmarks → PCC=0.72
      - Memory: 300 × 480 × 854 × 3 × 4 bytes ≈ 1.5 GB (vs ~7 GB at 1080p)

    Returns: (frames_array, actual_fps)
      frames_array: np.ndarray shape (T, H, W, 3) float32 [0,1] RGB
      actual_fps: float — the effective fps after downsampling
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Downsample: keep every step-th frame to approximate target_fps
    step = max(1, round(orig_fps / target_fps))
    actual_fps = orig_fps / step

    # Resize to max_height preserving aspect ratio
    need_resize = orig_h > max_height and orig_h > 0
    if need_resize:
        scale = max_height / orig_h
        new_h = max_height
        new_w = int(orig_w * scale)
    else:
        new_h, new_w = orig_h, orig_w

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if need_resize:
                frame = cv2.resize(frame, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
        idx += 1
    cap.release()

    if len(frames) < 8:
        return None, actual_fps

    return np.stack(frames, axis=0), actual_fps


# ─── MediaPipe FaceMesh landmark-based cheek ROIs ──────────────────────────
#
# Why landmarks matter: Haar cascade gives a rough bounding box, and
# percentage-based cheek ROIs from that box overlap or include non-cheek
# skin → both sides sample similar tissue → PCC is high for BOTH real & fake.
#
# MediaPipe FaceMesh gives 468 precise anatomical landmarks. We use them to
# define cheek polygons that are truly spatially separated (different vascular
# territories), which is what makes PCC discriminative.
#
# Left cheek polygon (viewer's left = subject's right):
#   landmarks 50, 101, 118, 117, 116, 123, 147, 213, 192, 187, 205, 36
# Right cheek polygon (viewer's right = subject's left):
#   landmarks 280, 330, 347, 346, 345, 352, 376, 433, 416, 411, 425, 266

LEFT_CHEEK_IDS  = [50, 101, 118, 117, 116, 123, 147, 213, 192, 187, 205, 36]
RIGHT_CHEEK_IDS = [280, 330, 347, 346, 345, 352, 376, 433, 416, 411, 425, 266]

# Smaller fallback: just the core cheek triangle (fewer landmarks, still separate)
LEFT_CHEEK_SMALL  = [50, 117, 187, 205, 36]
RIGHT_CHEEK_SMALL = [280, 346, 411, 425, 266]


_MP_CACHE = {"type": None, "obj": None, "initialized": False}


def _init_mediapipe():
    """Initialize MediaPipe FaceMesh. Cached per-process (only runs once per worker)."""
    if _MP_CACHE["initialized"]:
        return (_MP_CACHE["type"], _MP_CACHE["obj"])
    _MP_CACHE["initialized"] = True
    # Try legacy solutions API
    try:
        import mediapipe as mp
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
        _MP_CACHE["type"], _MP_CACHE["obj"] = "legacy", fm
        return ("legacy", fm)
    except Exception:
        pass

    # Try Tasks API (Kaggle/Colab)
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        import urllib.request, tempfile

        model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                model_path,
            )
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        landmarker = FaceLandmarker.create_from_options(opts)
        _MP_CACHE["type"], _MP_CACHE["obj"] = "tasks", landmarker
        return ("tasks", landmarker)
    except Exception:
        pass

    _MP_CACHE["type"], _MP_CACHE["obj"] = "none", None
    return ("none", None)


def _get_landmarks(backend_type, backend_obj, frame_uint8_rgb, H, W):
    """Get 468 landmarks as list of (x_px, y_px) from a single frame."""
    if backend_type == "legacy":
        results = backend_obj.process(frame_uint8_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        return [(l.x * W, l.y * H) for l in lm]

    elif backend_type == "tasks":
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_uint8_rgb)
        results = backend_obj.detect(mp_image)
        if not results.face_landmarks:
            return None
        lm = results.face_landmarks[0]
        return [(l.x * W, l.y * H) for l in lm]

    return None


def _polygon_mask(pts, H, W):
    """Create binary mask from polygon points."""
    pts_int = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_int, 1)
    return mask.astype(bool)


def _extract_lr_cheek_green(frames: np.ndarray) -> tuple:
    """
    Extract GREEN channel mean from LEFT and RIGHT cheek ROIs separately.

    Uses MediaPipe FaceMesh landmarks to define precise cheek polygons.
    Detects landmarks on first frame with a face, uses fixed mask for all frames
    (temporal consistency — no per-frame jitter).

    Fallback: Haar cascade + percentage ROIs if MediaPipe unavailable.

    Returns: (left_green, right_green) each shape (T,) float32
    """
    T, H, W, _ = frames.shape

    # Try MediaPipe landmark-based ROIs
    backend_type, backend_obj = _init_mediapipe()

    left_mask = right_mask = None
    if backend_type != "none":
        # Find landmarks on first few frames (pick first success)
        for t in range(min(5, T)):
            frame_uint8 = (frames[t] * 255).astype(np.uint8)
            landmarks = _get_landmarks(backend_type, backend_obj, frame_uint8, H, W)
            if landmarks and len(landmarks) >= 468:
                lc_pts = [(int(landmarks[i][0]), int(landmarks[i][1]))
                          for i in LEFT_CHEEK_IDS]
                rc_pts = [(int(landmarks[i][0]), int(landmarks[i][1]))
                          for i in RIGHT_CHEEK_IDS]
                left_mask = _polygon_mask(lc_pts, H, W)
                right_mask = _polygon_mask(rc_pts, H, W)

                # Sanity: masks should have reasonable area (>100 pixels)
                if left_mask.sum() < 100 or right_mask.sum() < 100:
                    left_mask = right_mask = None
                    continue
                break

        # Don't close — backend is cached per-process for reuse

    # Fallback: Haar cascade + percentage ROIs
    if left_mask is None or right_mask is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)

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
            cx, cy = W // 2, H // 4
            fw, fh = int(W * 0.4), int(H * 0.5)
            face_box = (cx - fw // 2, cy, fw, fh)

        x, y, w, h = face_box
        # Narrower ROIs than before — avoid center overlap
        lc_x1, lc_x2 = max(0, x + int(w * 0.03)), x + int(w * 0.28)
        lc_y1, lc_y2 = y + int(h * 0.45), y + int(h * 0.65)
        rc_x1, rc_x2 = x + int(w * 0.72), min(W, x + int(w * 0.97))
        rc_y1, rc_y2 = y + int(h * 0.45), y + int(h * 0.65)

        left_mask = np.zeros((H, W), dtype=bool)
        left_mask[max(0,lc_y1):min(H,lc_y2), max(0,lc_x1):min(W,lc_x2)] = True
        right_mask = np.zeros((H, W), dtype=bool)
        right_mask[max(0,rc_y1):min(H,rc_y2), max(0,rc_x1):min(W,rc_x2)] = True

    left_signal = np.zeros(T, dtype=np.float32)
    right_signal = np.zeros(T, dtype=np.float32)

    for t in range(T):
        green = frames[t, :, :, 1]  # green channel
        left_signal[t] = green[left_mask].mean()
        right_signal[t] = green[right_mask].mean()

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
    video_path, out_dir, target_fps, max_frames, max_height, force_recompute = args_tuple
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

    # Load video frames at 480p (enough for ~60px cheek ROIs)
    frames_arr, actual_fps = load_video_frames(
        str(video_path), max_frames=max_frames, target_fps=target_fps,
        max_height=max_height
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

def find_all_videos(ff_root: Path, manip_filter: list = None) -> list:
    """
    Find all MP4 videos in the FF++ dataset. Supports multiple layouts:
      1. <root>/<manip>/c23/videos/*.mp4   (standard FF++ layout)
      2. <root>/<manip>/*.mp4              (flat layout)
      3. <root>/<manip>/videos/*.mp4       (alternative)

    manip_filter: if set, only process these manipulation types
                  e.g., ["original", "Deepfakes"]
    """
    types_to_scan = manip_filter if manip_filter else FF_MANIPULATION_TYPES
    all_videos = []
    for manip in types_to_scan:
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

    # Parse --manips filter (e.g., "original,Deepfakes")
    manip_filter = None
    if args.manips:
        manip_filter = [m.strip() for m in args.manips.split(",")]
        print(f"Filtering to: {manip_filter}")

    print(f"Scanning for MP4 videos in: {ff_root}")
    all_videos = find_all_videos(ff_root, manip_filter=manip_filter)
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

    work = [(str(v), str(out_dir), args.target_fps, args.max_frames,
             args.max_height, args.force)
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
                   help="Root of FF++ MP4 dataset (e.g., /kaggle/input/ff-c23/FaceForensics++_C23)")
    p.add_argument("--out_dir", default="./rppg_v2_cache",
                   help="Output cache directory")
    p.add_argument("--manips", type=str, default=None,
                   help="Comma-separated manipulation types to process "
                        "(e.g., 'original,Deepfakes'). Default: all 6 types. "
                        "Use this to split work across 3 parallel Kaggle runs.")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Downsample video to this FPS (default 25)")
    p.add_argument("--max_frames", type=int, default=300,
                   help="Max frames per video (default 300 = ~12s at 25fps)")
    p.add_argument("--max_height", type=int, default=480,
                   help="Resize frames to this height (default 480). "
                        "480p gives ~60px cheek ROIs — enough for clean rPPG.")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Parallel workers (default 1 — MediaPipe is not fork-safe. "
                        "Use 1 for landmark mode. 2+ is OK if you add --no_landmarks)")
    p.add_argument("--force", action="store_true",
                   help="Force recompute even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
