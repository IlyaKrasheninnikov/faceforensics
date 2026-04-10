"""
Blink feature extraction from FULL MP4 VIDEOS.

Extracts 16-d blink feature vector per video using MediaPipe FaceMesh EAR.
Features: blinks/min, mean duration, IBI variability, EAR stats, entropy, etc.

Output (compatible with train_physio_png.py --blink_cache):
    <out_dir>/<manip>/<video_stem>/blink_feat.npy   — shape (16,) float32
    <out_dir>/<manip>/<video_stem>/blink_meta.json   — all blink stats

Usage (Kaggle):
    python w3_train/extract_blinks_mp4.py \
        --ff_root /kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23 \
        --out_dir /kaggle/working/blink_cache \
        --manips "original,Deepfakes,Face2Face"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

FF_MANIPULATION_TYPES = ["original", "Deepfakes", "Face2Face", "FaceSwap",
                          "NeuralTextures", "FaceShifter"]

BLINK_FEAT_DIM = 16

# MediaPipe 478-landmark indices for left and right eyes
LEFT_EYE_IDS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDS = [33,  160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_ids, H, W):
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H] for i in eye_ids])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


# ─── MediaPipe backend ────────────────────────────────────────────────────

_MP_CACHE = {"type": None, "obj": None, "opts": None, "initialized": False}


def _init_mediapipe():
    """Initialize MediaPipe. Cached per-process."""
    if _MP_CACHE["initialized"]:
        return _MP_CACHE["type"], _MP_CACHE["obj"], _MP_CACHE["opts"]
    _MP_CACHE["initialized"] = True

    # Try legacy solutions API
    try:
        import mediapipe as mp
        _ = mp.solutions.face_mesh.FaceMesh
        _MP_CACHE["type"] = "legacy"
        _MP_CACHE["obj"] = mp.solutions.face_mesh
        print("[Blink] MediaPipe legacy solutions API")
        return "legacy", mp.solutions.face_mesh, None
    except Exception:
        pass

    # Try Tasks API
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        import urllib.request, tempfile

        model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not os.path.exists(model_path):
            print("[Blink] Downloading face_landmarker.task...")
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
        _MP_CACHE["type"] = "tasks"
        _MP_CACHE["opts"] = opts
        print("[Blink] MediaPipe Tasks API")
        return "tasks", None, opts
    except Exception as e:
        print(f"[Blink] MediaPipe unavailable: {e}")

    _MP_CACHE["type"] = "none"
    return "none", None, None


def extract_ear_series(video_path, target_fps=15.0, max_frames=600):
    """Extract per-frame EAR from video. Returns (ear_mean, ears_left, ears_right, fps) or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(orig_fps / target_fps))

    ears_left, ears_right = [], []
    idx = 0

    backend_type, backend_obj, backend_opts = _init_mediapipe()

    if backend_type == "legacy":
        with backend_obj.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while len(ears_left) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step != 0:
                    idx += 1
                    continue
                H, W = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(frame_rgb)
                if result.multi_face_landmarks:
                    lms = result.multi_face_landmarks[0].landmark
                    ears_left.append(eye_aspect_ratio(lms, LEFT_EYE_IDS, H, W))
                    ears_right.append(eye_aspect_ratio(lms, RIGHT_EYE_IDS, H, W))
                else:
                    ears_left.append(ears_left[-1] if ears_left else 0.3)
                    ears_right.append(ears_right[-1] if ears_right else 0.3)
                idx += 1

    elif backend_type == "tasks":
        from mediapipe.tasks.python.vision import FaceLandmarker
        import mediapipe as mp
        landmarker = FaceLandmarker.create_from_options(backend_opts)
        try:
            while len(ears_left) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step != 0:
                    idx += 1
                    continue
                H, W = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect(mp_image)
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    ears_left.append(eye_aspect_ratio(lms, LEFT_EYE_IDS, H, W))
                    ears_right.append(eye_aspect_ratio(lms, RIGHT_EYE_IDS, H, W))
                else:
                    ears_left.append(ears_left[-1] if ears_left else 0.3)
                    ears_right.append(ears_right[-1] if ears_right else 0.3)
                idx += 1
        finally:
            landmarker.close()
    else:
        cap.release()
        return None

    cap.release()

    if len(ears_left) < 10:
        return None

    ear_mean = (np.array(ears_left) + np.array(ears_right)) / 2.0
    return ear_mean, ears_left, ears_right, target_fps


def detect_blinks(ear_series, fps, consec_frames=2):
    """Detect blink events from EAR time series. Returns stats dict."""
    # Otsu threshold
    ear_norm = ((ear_series - ear_series.min()) / (np.ptp(ear_series) + 1e-8) * 255).astype(np.uint8)
    _, thresh_img = cv2.threshold(ear_norm.reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
    threshold = ear_series.min() + np.ptp(ear_series) * (thresh_img[0, 0] / 255.0)
    threshold = min(threshold, 0.25)

    closed = (ear_series < threshold).astype(int)
    blinks = []
    in_blink = False
    blink_start = 0
    for i, c in enumerate(closed):
        if c == 1 and not in_blink:
            in_blink = True
            blink_start = i
        elif c == 0 and in_blink:
            duration = i - blink_start
            if duration >= consec_frames:
                blinks.append({"start": blink_start, "end": i, "duration_frames": duration})
            in_blink = False

    T = len(ear_series)
    duration_sec = T / fps
    n_blinks = len(blinks)
    bpm_blink = n_blinks / duration_sec * 60.0 if duration_sec > 0 else 0.0
    durations = [b["duration_frames"] for b in blinks]
    starts = [b["start"] for b in blinks]
    ibi = np.diff(starts).tolist() if len(starts) > 1 else []

    hist, _ = np.histogram(ear_series, bins=20, range=(0, 0.5), density=True)
    hist = hist + 1e-10
    ear_entropy = float(scipy_entropy(hist))

    return {
        "n_blinks": n_blinks,
        "blinks_per_min": float(bpm_blink),
        "mean_blink_duration_frames": float(np.mean(durations)) if durations else 0.0,
        "std_blink_duration_frames": float(np.std(durations)) if durations else 0.0,
        "ibi_mean_frames": float(np.mean(ibi)) if ibi else 0.0,
        "ibi_std_frames": float(np.std(ibi)) if ibi else 0.0,
        "ibi_cv": float(np.std(ibi) / (np.mean(ibi) + 1e-8)) if ibi else 0.0,
        "ear_mean": float(ear_series.mean()),
        "ear_std": float(ear_series.std()),
        "ear_entropy": ear_entropy,
        "ear_threshold_used": float(threshold),
    }


def compute_blink_feature(blink_stats):
    """Convert blink stats to 16-d normalized feature vector."""
    feat = np.array([
        blink_stats.get("blinks_per_min", 0.0) / 30.0,      # 0: normalized blink rate
        blink_stats.get("mean_blink_duration_frames", 0.0) / 10.0,  # 1: mean duration
        blink_stats.get("std_blink_duration_frames", 0.0) / 5.0,    # 2: duration variability
        blink_stats.get("ibi_cv", 0.0),                      # 3: inter-blink interval CV
        blink_stats.get("ibi_mean_frames", 0.0) / 100.0,     # 4: mean IBI
        blink_stats.get("ibi_std_frames", 0.0) / 50.0,       # 5: IBI std
        blink_stats.get("ear_mean", 0.3),                     # 6: mean EAR (eye openness)
        blink_stats.get("ear_std", 0.0),                      # 7: EAR variability
        blink_stats.get("ear_entropy", 0.0) / 5.0,           # 8: EAR signal regularity
        float(blink_stats.get("n_blinks", 0)) / 20.0,        # 9: raw blink count
        blink_stats.get("ear_threshold_used", 0.25),          # 10: adaptive threshold
        # Derived features
        1.0 if blink_stats.get("n_blinks", 0) == 0 else 0.0, # 11: zero-blink flag
        min(blink_stats.get("blinks_per_min", 0.0) / 60.0, 1.0),  # 12: high blink rate flag
        blink_stats.get("ear_std", 0.0) / (blink_stats.get("ear_mean", 0.3) + 1e-8),  # 13: relative EAR variation
        0.0,  # 14: reserved
        0.0,  # 15: reserved
    ], dtype=np.float32)[:BLINK_FEAT_DIM]
    return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)


# ─── Per-video worker ──────────────────────────────────────────────────────

def process_video(video_path, out_dir, force=False):
    """Extract blink features from one MP4 video."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    # Determine manip type from path
    manip = "unknown"
    for part in video_path.parts:
        if part in FF_MANIPULATION_TYPES:
            manip = part
            break
    if manip == "unknown":
        manip = video_path.parent.name

    video_stem = video_path.stem
    save_dir = out_dir / manip / video_stem
    feat_path = save_dir / "blink_feat.npy"
    meta_path = save_dir / "blink_meta.json"

    if not force and feat_path.exists() and meta_path.exists():
        return {"video": video_stem, "manip": manip, "status": "cached"}

    save_dir.mkdir(parents=True, exist_ok=True)

    result = extract_ear_series(str(video_path), target_fps=15.0, max_frames=600)
    if result is None:
        np.save(feat_path, np.zeros(BLINK_FEAT_DIM, dtype=np.float32))
        with open(meta_path, "w") as f:
            json.dump({"status": "no_face_or_too_short"}, f)
        return {"video": video_stem, "manip": manip, "status": "error"}

    ear_mean, ears_left, ears_right, fps = result
    blink_stats = detect_blinks(ear_mean, fps)
    feat = compute_blink_feature(blink_stats)

    np.save(feat_path, feat)
    meta = {k: v for k, v in blink_stats.items() if k != "ear_series"}
    meta["status"] = "ok"
    meta["n_frames"] = len(ear_mean)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return {
        "video": video_stem,
        "manip": manip,
        "status": "ok",
        "blinks_per_min": blink_stats["blinks_per_min"],
    }


# ─── Discover videos ────────────────────────────────────────────────────────

def find_all_videos(ff_root, manip_filter=None):
    types_to_scan = manip_filter if manip_filter else FF_MANIPULATION_TYPES
    all_videos = []
    for manip in types_to_scan:
        manip_dir = Path(ff_root) / manip
        if not manip_dir.exists():
            print(f"  {manip}: NOT FOUND")
            continue
        for cand in [manip_dir, manip_dir / "c23" / "videos", manip_dir / "c23", manip_dir / "videos"]:
            if cand.exists():
                vids = sorted(list(cand.glob("*.mp4")))
                if vids:
                    all_videos.extend(vids)
                    print(f"  {manip}: {len(vids)} videos ({cand})")
                    break
        else:
            vids = sorted(list(manip_dir.rglob("*.mp4")))
            if vids:
                all_videos.extend(vids)
                print(f"  {manip}: {len(vids)} videos (recursive)")
    return all_videos


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    ff_root = Path(args.ff_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manip_filter = [m.strip() for m in args.manips.split(",")] if args.manips else None
    if manip_filter:
        print(f"Filtering to: {manip_filter}")

    print(f"Scanning for MP4 videos in: {ff_root}")
    all_videos = find_all_videos(ff_root, manip_filter)
    print(f"\nTotal: {len(all_videos)} MP4 videos")

    if not all_videos:
        print("ERROR: No MP4 videos found.")
        return

    start = time.time()
    n_ok = n_cached = n_err = 0
    bpm_real, bpm_fake = [], []

    for vpath in tqdm(all_videos, desc="Blink extract"):
        r = process_video(str(vpath), str(out_dir), force=args.force)
        if r["status"] == "ok":
            n_ok += 1
            if r["manip"] == "original":
                bpm_real.append(r.get("blinks_per_min", 0))
            else:
                bpm_fake.append(r.get("blinks_per_min", 0))
        elif r["status"] == "cached":
            n_cached += 1
        else:
            n_err += 1

    elapsed = time.time() - start
    print(f"\nDone: {n_ok} extracted, {n_cached} cached, {n_err} errors")
    print(f"Time: {elapsed / 60:.1f} min")

    if bpm_real and bpm_fake:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(bpm_real, bpm_fake)
        print(f"\n{'='*50}")
        print(f"  BLINK RATE DISCRIMINATION CHECK")
        print(f"{'='*50}")
        print(f"  Real: {np.mean(bpm_real):.1f} +/- {np.std(bpm_real):.1f} blinks/min  (n={len(bpm_real)})")
        print(f"  Fake: {np.mean(bpm_fake):.1f} +/- {np.std(bpm_fake):.1f} blinks/min  (n={len(bpm_fake)})")
        print(f"  KS stat: {ks_stat:.3f}  p={ks_p:.2e}")
        if ks_p < 0.05:
            print(f"  >>> SIGNIFICANT — blink features ARE discriminative!")
        else:
            print(f"  >>> Not significant (but still useful as auxiliary model input)")
        print(f"{'='*50}")

    print(f"\nFeatures saved to: {out_dir}/<manip>/<video_stem>/blink_feat.npy")
    print(f"Compatible with: train_physio_png.py --blink_cache {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Blink feature extraction from MP4 videos")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--out_dir", default="./blink_cache")
    p.add_argument("--manips", type=str, default=None,
                   help="Comma-separated manip types (e.g., 'original,Deepfakes')")
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
