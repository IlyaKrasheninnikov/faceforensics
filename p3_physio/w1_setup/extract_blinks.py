"""
W1: Blink detection via MediaPipe FaceMesh + Eye Aspect Ratio (EAR).

For each video: computes EAR time series, detects blink events, measures:
  - blinks per minute
  - mean blink duration (frames)
  - inter-blink interval statistics
  - EAR signal entropy (regularity)

Usage:
    python extract_blinks.py --video_dir /data/FF++/original/c23/videos \
                             --label real \
                             --out_dir ./logs/signal_cache

    python extract_blinks.py --video_dir /data/FF++/Deepfakes/c23/videos \
                             --label fake \
                             --out_dir ./logs/signal_cache
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

# ─── MediaPipe compatibility shim ─────────────────────────────────────────────
# Priority: mp_legacy (solutions API, mp<=0.10.3) → mp_tasks (Tasks API, mp>=0.10)
#           → opencv (brightness fallback, last resort)
_FACE_BACKEND = None
_mp_face_mesh_cls = None      # used by mp_legacy
_mp_landmarker_opts = None    # used by mp_tasks


def _init_face_backend():
    global _FACE_BACKEND, _mp_face_mesh_cls, _mp_landmarker_opts

    # 1) Try legacy solutions API (mediapipe <= 0.10.3)
    try:
        import mediapipe as mp
        _ = mp.solutions.face_mesh.FaceMesh  # raises AttributeError on new API
        _mp_face_mesh_cls = mp.solutions.face_mesh
        _FACE_BACKEND = "mp_legacy"
        print("[FaceBackend] MediaPipe legacy solutions API")
        return
    except Exception:
        pass

    # 2) Try new Tasks API (mediapipe >= 0.10 on Kaggle/Colab)
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        # Download the face landmarker model if not already cached
        import urllib.request, os, tempfile
        model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not os.path.exists(model_path):
            print("[FaceBackend] Downloading MediaPipe face_landmarker.task model (~3MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                model_path,
            )

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        _mp_landmarker_opts = FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _FACE_BACKEND = "mp_tasks"
        print("[FaceBackend] MediaPipe Tasks API (FaceLandmarker)")
        return
    except Exception as e:
        print(f"[FaceBackend] MediaPipe Tasks unavailable ({e}) — falling back to brightness heuristic")

    # 3) Last resort: brightness proxy
    _FACE_BACKEND = "opencv"
    print("[FaceBackend] Using fixed eye-region brightness heuristic")


_init_face_backend()

# MediaPipe 478-landmark indices for left and right eyes (same for both APIs)
LEFT_EYE_IDS  = [362, 385, 387, 263, 373, 380]  # p1..p6
RIGHT_EYE_IDS = [33,  160, 158, 133, 153, 144]  # p1..p6


def eye_aspect_ratio(landmarks, eye_ids: list, H: int, W: int) -> float:
    """
    Compute Eye Aspect Ratio (EAR) — Soukupova & Cech 2016.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    EAR ≈ 0.3 for open eye, ~0 for closed.

    landmarks: list of landmark objects with .x/.y (normalized) attributes.
    """
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H] for i in eye_ids])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


def _fixed_eye_brightness(frame_gray: np.ndarray) -> float:
    """
    Fast blink proxy for face-cropped videos (FF++ style).
    FF++ videos are already tightly cropped to the face, so fixed relative
    boxes work reliably: upper-middle strip covers both eyes.

    Blink signal: mean brightness of the eye-band.
    When eyes close → darker eyelids appear → brightness drops.
    Returns a pseudo-EAR in ~[0.1, 0.4]: lower = more closed.
    """
    H, W = frame_gray.shape
    # Eye band: rows 25-45% of height, columns 15-85% of width
    y1, y2 = int(H * 0.25), int(H * 0.45)
    x1, x2 = int(W * 0.15), int(W * 0.85)
    band = frame_gray[y1:y2, x1:x2].astype(np.float32)
    mean_brightness = float(band.mean()) / 255.0  # [0, 1]
    # Scale: typical open-eye region ~0.55-0.75 brightness → pseudo-EAR ~0.25-0.35
    # Closed eye / blink → brightness drops by ~0.1-0.2
    pseudo_ear = float(np.clip(mean_brightness * 0.55, 0.05, 0.4))
    return pseudo_ear


def extract_ear_series(video_path: str, target_fps: float = 15.0, max_frames: int = 600):
    """
    Extract per-frame EAR (left, right, mean) from video.
    Returns (ear_mean, ears_left, ears_right, fps) tuple, or dict with 'error'.

    Backends:
      mp_legacy  — MediaPipe FaceMesh solutions API (mp <= 0.10.3)
      mp_tasks   — MediaPipe Tasks FaceLandmarker API (mp >= 0.10, Kaggle/Colab)
      opencv     — Fixed eye-region brightness proxy (last resort)
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(orig_fps / target_fps))

    ears_left: list = []
    ears_right: list = []
    idx = 0

    # ── MediaPipe legacy solutions path ───────────────────────────────────────
    if _FACE_BACKEND == "mp_legacy":
        with _mp_face_mesh_cls.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
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
                    ear_l = eye_aspect_ratio(lms, LEFT_EYE_IDS, H, W)
                    ear_r = eye_aspect_ratio(lms, RIGHT_EYE_IDS, H, W)
                else:
                    ear_l = ears_left[-1] if ears_left else 0.3
                    ear_r = ears_right[-1] if ears_right else 0.3
                ears_left.append(ear_l)
                ears_right.append(ear_r)
                idx += 1

    # ── MediaPipe Tasks API path (Kaggle: mediapipe >= 0.10) ──────────────────
    elif _FACE_BACKEND == "mp_tasks":
        from mediapipe.tasks.python.vision import FaceLandmarker
        import mediapipe as mp

        landmarker = FaceLandmarker.create_from_options(_mp_landmarker_opts)
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
                    lms = result.face_landmarks[0]  # list of NormalizedLandmark
                    ear_l = eye_aspect_ratio(lms, LEFT_EYE_IDS, H, W)
                    ear_r = eye_aspect_ratio(lms, RIGHT_EYE_IDS, H, W)
                else:
                    ear_l = ears_left[-1] if ears_left else 0.3
                    ear_r = ears_right[-1] if ears_right else 0.3
                ears_left.append(ear_l)
                ears_right.append(ear_r)
                idx += 1
        finally:
            landmarker.close()

    # ── Brightness fallback ───────────────────────────────────────────────────
    else:
        while len(ears_left) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step != 0:
                idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pseudo_ear = _fixed_eye_brightness(gray)
            ears_left.append(pseudo_ear)
            ears_right.append(pseudo_ear)
            idx += 1

    cap.release()

    if len(ears_left) < 10:
        return {"error": "Too short"}

    ear_mean = (np.array(ears_left) + np.array(ears_right)) / 2.0
    return ear_mean, ears_left, ears_right, target_fps


def detect_blinks(ear_series: np.ndarray, fps: float, threshold: float = None, consec_frames: int = 2) -> dict:
    """
    Detect blink events from EAR time series.

    threshold: EAR below this = eye closed. If None, uses Otsu's method on EAR histogram.
    consec_frames: minimum consecutive frames below threshold to count as a blink.

    Returns dict with blink stats.
    """
    if threshold is None:
        # Otsu thresholding on EAR values (treat as grayscale histogram)
        ear_norm = ((ear_series - ear_series.min()) / (np.ptp(ear_series) + 1e-8) * 255).astype(np.uint8)
        _, thresh_img = cv2.threshold(ear_norm.reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
        threshold = ear_series.min() + np.ptp(ear_series) * (thresh_img[0, 0] / 255.0)
        threshold = min(threshold, 0.25)  # cap at standard threshold

    closed = (ear_series < threshold).astype(int)

    # Find blink events (runs of closed)
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
    ibi = np.diff(starts).tolist() if len(starts) > 1 else []  # inter-blink intervals in frames

    # EAR signal entropy (regularity measure — periodic/regular blinks = lower entropy)
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
        "ibi_cv": float(np.std(ibi) / (np.mean(ibi) + 1e-8)) if ibi else 0.0,  # coeff of variation
        "ear_mean": float(ear_series.mean()),
        "ear_std": float(ear_series.std()),
        "ear_entropy": ear_entropy,
        "ear_threshold_used": float(threshold),
        "ear_series": ear_series.tolist(),
        "blink_events": blinks,
    }


def process_video(video_path: str) -> dict:
    result = extract_ear_series(video_path, target_fps=15.0, max_frames=600)
    if isinstance(result, dict) and "error" in result:
        return result

    ear_mean, ears_left, ears_right, fps = result
    blink_stats = detect_blinks(ear_mean, fps)

    return {
        "video": str(video_path),
        "n_frames": len(ear_mean),
        "fps": fps,
        **blink_stats,
    }


def batch_extract(video_dir: str, label: str, out_dir: str, max_videos: int = 200) -> pd.DataFrame:
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) +
        list(video_dir.glob("*/*.mp4")) + list(video_dir.glob("*/*.avi"))
    )[:max_videos]

    if not video_files:
        print(f"[WARN] No videos in {video_dir}")
        return pd.DataFrame()

    print(f"\nProcessing {len(video_files)} {label} videos for blink detection...")

    records = []
    for vid_path in tqdm(video_files, desc=f"Blinks [{label}]"):
        try:
            res = process_video(str(vid_path))
            if "error" not in res:
                records.append({
                    "video": vid_path.name,
                    "label": label,
                    "n_frames": res["n_frames"],
                    "n_blinks": res["n_blinks"],
                    "blinks_per_min": res["blinks_per_min"],
                    "mean_blink_dur": res["mean_blink_duration_frames"],
                    "ibi_mean": res["ibi_mean_frames"],
                    "ibi_cv": res["ibi_cv"],
                    "ear_mean": res["ear_mean"],
                    "ear_entropy": res["ear_entropy"],
                })
        except Exception as e:
            print(f"  [ERR] {vid_path.name}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"blinks_summary_{label}.csv", index=False)
    print(f"\nSaved {len(df)} results → {out_dir / f'blinks_summary_{label}.csv'}")

    if not df.empty:
        print(f"\n  Blinks/min  mean={df.blinks_per_min.mean():.1f}  std={df.blinks_per_min.std():.1f}")
        print(f"  IBI CV      mean={df.ibi_cv.mean():.3f}  (higher = more irregular = more natural)")
        print(f"  EAR entropy mean={df.ear_entropy.mean():.3f}")

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Batch blink extraction for deepfake detection")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--label", choices=["real", "fake"], required=True)
    p.add_argument("--out_dir", default="./logs/signal_cache")
    p.add_argument("--max_videos", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_extract(args.video_dir, args.label, args.out_dir, args.max_videos)
