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
_FACE_BACKEND = None
_mp_face_mesh_cls = None


def _init_face_backend():
    global _FACE_BACKEND, _mp_face_mesh_cls
    try:
        import mediapipe as mp
        _ = mp.solutions.face_mesh.FaceMesh
        _mp_face_mesh_cls = mp.solutions.face_mesh
        _FACE_BACKEND = "mp_legacy"
        print("[FaceBackend] MediaPipe legacy solutions API")
    except Exception:
        _FACE_BACKEND = "opencv"
        print("[FaceBackend] MediaPipe solutions unavailable — blink via eye-region heuristic")


_init_face_backend()

# MediaPipe 468-landmark indices for left and right eyes
LEFT_EYE_IDS  = [362, 385, 387, 263, 373, 380]  # p1..p6
RIGHT_EYE_IDS = [33,  160, 158, 133, 153, 144]  # p1..p6


def eye_aspect_ratio(landmarks, eye_ids: list, H: int, W: int) -> float:
    """
    Compute Eye Aspect Ratio (EAR) — Soukupova & Cech 2016.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    EAR ≈ 0.3 for open eye, ~0 for closed.
    """
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H] for i in eye_ids])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


def _ear_from_eye_bbox(frame_gray: np.ndarray, eye_box) -> float:
    """
    Fallback EAR estimate using eye-region pixel intensity variance.
    When the eye is closed, the region is more uniform (less variance).
    Maps variance to a pseudo-EAR in [0, 0.4] range.
    """
    x, y, w, h = eye_box
    H_f, W_f = frame_gray.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W_f, x + w), min(H_f, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.3
    region = frame_gray[y1:y2, x1:x2].astype(np.float32)
    # Normalize variance to pseudo-EAR: closed eye → low var → low EAR
    var = float(region.var())
    return float(np.clip(var / 500.0, 0.0, 0.4))


def extract_ear_series(video_path: str, target_fps: float = 15.0, max_frames: int = 600):
    """
    Extract per-frame EAR (left, right, mean) from video.
    Returns (ear_mean, ears_left, ears_right, fps) tuple, or dict with 'error'.
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(orig_fps / target_fps))

    ears_left: list = []
    ears_right: list = []
    idx = 0

    # ── MediaPipe legacy path ─────────────────────────────────────────────────
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

    # ── OpenCV fallback: eye cascade ─────────────────────────────────────────
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        last_eye_boxes = None

        while len(ears_left) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step != 0:
                idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            ear_l = ears_left[-1] if ears_left else 0.3
            ear_r = ears_right[-1] if ears_right else 0.3

            if len(faces) > 0:
                fx, fy, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                face_gray = gray[fy:fy+fh, fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[0])  # sort by x → left/right
                    # Shift eye boxes to full-frame coordinates
                    left_box  = (fx + eyes[0][0], fy + eyes[0][1], eyes[0][2], eyes[0][3])
                    right_box = (fx + eyes[1][0], fy + eyes[1][1], eyes[1][2], eyes[1][3])
                    last_eye_boxes = (left_box, right_box)
                elif len(eyes) == 1:
                    box = (fx + eyes[0][0], fy + eyes[0][1], eyes[0][2], eyes[0][3])
                    last_eye_boxes = (box, box)

            if last_eye_boxes is not None:
                ear_l = _ear_from_eye_bbox(gray, last_eye_boxes[0])
                ear_r = _ear_from_eye_bbox(gray, last_eye_boxes[1])

            ears_left.append(ear_l)
            ears_right.append(ear_r)
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
        ear_norm = ((ear_series - ear_series.min()) / (ear_series.ptp() + 1e-8) * 255).astype(np.uint8)
        _, thresh_img = cv2.threshold(ear_norm.reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
        threshold = ear_series.min() + (ear_series.ptp()) * (thresh_img[0, 0] / 255.0)
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
