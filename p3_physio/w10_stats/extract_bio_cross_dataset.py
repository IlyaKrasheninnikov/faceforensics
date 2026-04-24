"""
E3: Cross-dataset bio-signal extraction for CelebDF-v2 and DFDC.

Extracts the 12-d rPPG-v2 feature (left/right-cheek synchrony) and the 16-d
blink feature using the same algorithms as w3_train/extract_rppg_v2_png.py
and w3_train/extract_blinks_mp4.py — but adapted to the CelebDF and DFDC
directory layouts, both of which store per-video face crops as PNG/JPG.

Output matches the schema expected by multiseed_and_stats.py:

    <rppg_out_dir>/<class_name>/<video_id>/rppg_v2_feat.npy   shape (12,)
    <rppg_out_dir>/<class_name>/<video_id>/rppg_v2_meta.json
    <blink_out_dir>/<class_name>/<video_id>/blink_feat.npy    shape (16,)
    <blink_out_dir>/<class_name>/<video_id>/blink_meta.json

The "class_name" is either a CelebDF split/class ("Train_real", "Train_fake",
"Test_real", "Test_fake") or a DFDC split/class ("train_real", "train_fake",
"validation_real", "validation_fake"). The downstream multiseed script finds
every <class>/<video> subfolder so the class naming doesn't affect loading.

Usage on Kaggle (four commands):

  # CelebDF rPPG
  python p3_physio/w10_stats/extract_bio_cross_dataset.py rppg celebdf \\
      --data_root $CDF \\
      --out_dir /kaggle/working/celebdf_rppg_cache \\
      --max_frames 64 --fps 25 --num_workers 4

  # CelebDF blink
  python p3_physio/w10_stats/extract_bio_cross_dataset.py blink celebdf \\
      --data_root $CDF \\
      --out_dir /kaggle/working/celebdf_blink_cache \\
      --max_frames 150 --fps 15 --num_workers 4

  # DFDC rPPG
  python p3_physio/w10_stats/extract_bio_cross_dataset.py rppg dfdc \\
      --data_root $DFDC \\
      --out_dir /kaggle/working/dfdc_rppg_cache \\
      --max_frames 64 --fps 25 --num_workers 4

  # DFDC blink
  python p3_physio/w10_stats/extract_bio_cross_dataset.py blink dfdc \\
      --data_root $DFDC \\
      --out_dir /kaggle/working/dfdc_blink_cache \\
      --max_frames 150 --fps 15 --num_workers 4

After all four finish, rerun the probe stage with the new caches wired in
(see multiseed_and_stats.py probe sub-command).
"""

import argparse
import json
import os
import sys
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Make the w3 module path importable — we reuse the existing rPPG logic
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR.parent / "w3_train"))

# ───────────────────────────────────────────────────────────────────────────
# Scanners: produce [(video_id, class_name, [frame_paths]), ...]
# class_name becomes the first folder level in the cache (replaces manip for FF)
# ───────────────────────────────────────────────────────────────────────────

def scan_celebdf(root):
    """CelebDF layout: <root>/{Test,Train}/{real,fake}/<video_id>/*.png"""
    root = Path(root)
    items = []
    for split in ["Test", "Train"]:
        for lname in ["real", "fake"]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            class_name = f"{split}_{lname}"   # e.g. Train_real
            for sd in sorted(d for d in ldir.iterdir() if d.is_dir()):
                frames = sorted(
                    os.path.join(sd, f) for f in os.listdir(sd)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )
                if frames:
                    items.append((sd.name, class_name, frames))
    return items


def scan_dfdc(root):
    """
    DFDC-faces layout: <root>/{validation,train}/{real,fake}/<video_id>_<something>.jpg
    Face crops are flat files — group by video_id prefix.
    """
    root = Path(root)
    items = []
    for split in ["validation", "train"]:
        for lname in ["real", "fake"]:
            ldir = root / split / lname
            if not ldir.exists():
                continue
            class_name = f"{split}_{lname}"
            vid_to_files = {}
            for f in ldir.iterdir():
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    stem = f.stem
                    vid_id = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem.split("_")[0]
                    vid_to_files.setdefault(vid_id, []).append(str(f))
            for vid_id, files in sorted(vid_to_files.items()):
                files = sorted(files)
                if files:
                    items.append((vid_id, class_name, files))
    return items


def scan_ff(root):
    """FF++ layout: <root>/<manip>/<video_id>/*.png  — handled like CelebDF per-manip."""
    root = Path(root)
    FF_MANIPS = ["original", "Deepfakes", "Face2Face", "FaceSwap",
                 "NeuralTextures", "FaceShifter"]
    items = []
    for manip in FF_MANIPS:
        mdir = root / manip
        if not mdir.exists():
            continue
        for sd in sorted(d for d in mdir.iterdir() if d.is_dir()):
            frames = sorted(
                os.path.join(sd, f) for f in os.listdir(sd)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if frames:
                items.append((sd.name, manip, frames))
    return items


SCANNERS = {"celebdf": scan_celebdf, "dfdc": scan_dfdc, "ff": scan_ff}


# ───────────────────────────────────────────────────────────────────────────
# rPPG v2 worker — imports the existing logic from w3_train/extract_rppg_v2_png
# ───────────────────────────────────────────────────────────────────────────

def _load_frames_for_rppg(frame_paths, max_frames):
    """
    Load up to max_frames uniformly sampled RGB frames.

    CelebDF clips have variable per-frame resolution, so we can't blindly
    np.stack.  We use the first-loaded frame as the reference size; later
    frames are resized to match.  If the reference is smaller than 256 px
    on either axis, the whole clip is upscaled to 256x256 so the cheek
    ROIs are large enough for rPPG (per extract_rppg_v2_png notes, cheek
    needs >~100 px wide).
    """
    if len(frame_paths) > max_frames:
        idxs = np.linspace(0, len(frame_paths) - 1, max_frames).astype(int)
        frame_paths = [frame_paths[i] for i in idxs]

    frames = []
    target_hw = None
    for f in frame_paths:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if target_hw is None:
            H, W = img.shape[:2]
            # Upscale small crops so cheek ROI has enough pixels
            if min(H, W) < 200:
                target_hw = (max(256, H), max(256, W))
            else:
                target_hw = (H, W)
        Ht, Wt = target_hw
        if img.shape[:2] != target_hw:
            img = cv2.resize(img, (Wt, Ht))
        frames.append(img.astype(np.float32) / 255.0)

    if len(frames) < 8:
        return None
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def _rppg_worker(task):
    """Run rPPG v2 extraction on a single video folder. Returns status dict."""
    vid_id, class_name, frame_paths, out_dir, fps, max_frames, force = task

    from extract_rppg_v2_png import (
        _extract_lr_cheek_green, compute_sync_features, RPPG_V2_FEAT_DIM,
    )

    save_dir = Path(out_dir) / class_name / vid_id
    feat_path = save_dir / "rppg_v2_feat.npy"
    meta_path = save_dir / "rppg_v2_meta.json"

    if not force and feat_path.exists() and meta_path.exists():
        return {"vid": vid_id, "class": class_name, "status": "cached"}

    save_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames_for_rppg(frame_paths, max_frames)

    if frames is None:
        np.save(feat_path, np.zeros(RPPG_V2_FEAT_DIM, dtype=np.float32))
        with open(meta_path, "w") as f:
            json.dump({"status": "too_few_frames", "n_frames": 0}, f)
        return {"vid": vid_id, "class": class_name, "status": "too_few_frames"}

    try:
        left_green, right_green = _extract_lr_cheek_green(frames)
        feat, meta = compute_sync_features(left_green, right_green, fps)
        meta["n_frames"] = int(len(frames))
        meta["status"] = "ok"
    except Exception as e:
        feat = np.zeros(RPPG_V2_FEAT_DIM, dtype=np.float32)
        meta = {"status": f"error: {type(e).__name__}: {e}",
                "n_frames": int(len(frames))}

    np.save(feat_path, feat)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return {"vid": vid_id, "class": class_name,
            "status": meta.get("status", "ok"),
            "pcc": meta.get("pcc", 0.0)}


# ───────────────────────────────────────────────────────────────────────────
# Blink worker — adapts the EAR/blink logic from extract_blinks_mp4 to PNG frames
# ───────────────────────────────────────────────────────────────────────────

BLINK_FEAT_DIM = 16
LEFT_EYE_IDS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDS = [33, 160, 158, 133, 153, 144]


def _eye_aspect_ratio(landmarks, eye_ids, H, W):
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H] for i in eye_ids])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


def _eye_aspect_ratio_tasks(landmarks_np, eye_ids, H, W):
    """Same EAR but for Tasks API landmarks (list of NormalizedLandmark)."""
    pts = np.array([[landmarks_np[i].x * W, landmarks_np[i].y * H] for i in eye_ids])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


# Cached per-process (workers inherit only at process start)
_MP_CACHE = {"backend": None, "face_mesh": None, "landmarker": None}


def _init_mediapipe():
    """Initialize MediaPipe FaceMesh. Prefers legacy API, falls back to Tasks."""
    if _MP_CACHE["backend"] is not None:
        return _MP_CACHE["backend"]

    # Try legacy API first (mp <= 0.10.3)
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        _MP_CACHE["backend"] = "legacy"
        _MP_CACHE["face_mesh"] = face_mesh
        return "legacy"
    except Exception:
        pass

    # Tasks API fallback
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )
        import urllib.request
        model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task",
                model_path,
            )
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = FaceLandmarker.create_from_options(opts)
        _MP_CACHE["backend"] = "tasks"
        _MP_CACHE["landmarker"] = landmarker
        return "tasks"
    except Exception as e:
        raise RuntimeError(f"MediaPipe not available: {e}")


def _extract_ear_series_from_frames(frame_paths, max_frames):
    """Load frames and compute per-frame EAR. Returns (ear_mean, fps_effective)."""
    backend = _init_mediapipe()

    # Uniformly subsample
    if len(frame_paths) > max_frames:
        idxs = np.linspace(0, len(frame_paths) - 1, max_frames).astype(int)
        frame_paths = [frame_paths[i] for i in idxs]

    ears_left, ears_right = [], []

    if backend == "legacy":
        face_mesh = _MP_CACHE["face_mesh"]
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            H, W = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                ears_left.append(_eye_aspect_ratio(lms, LEFT_EYE_IDS, H, W))
                ears_right.append(_eye_aspect_ratio(lms, RIGHT_EYE_IDS, H, W))
            else:
                # Smoother interpolation than outright skip
                ears_left.append(ears_left[-1] if ears_left else 0.3)
                ears_right.append(ears_right[-1] if ears_right else 0.3)
    else:
        import mediapipe as mp
        landmarker = _MP_CACHE["landmarker"]
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            H, W = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                ears_left.append(_eye_aspect_ratio_tasks(lms, LEFT_EYE_IDS, H, W))
                ears_right.append(_eye_aspect_ratio_tasks(lms, RIGHT_EYE_IDS, H, W))
            else:
                ears_left.append(ears_left[-1] if ears_left else 0.3)
                ears_right.append(ears_right[-1] if ears_right else 0.3)

    if len(ears_left) < 10:
        return None, None

    ear_mean = (np.array(ears_left) + np.array(ears_right)) / 2.0
    # Effective fps: many CDF/DFDC crops are sparse frames with unknown original
    # timing. We assume frames approximate 1fps-equivalent temporal spacing and
    # re-scale to 15 fps below. We'll record this explicitly.
    fps_effective = 15.0  # same convention as extract_blinks_mp4
    return ear_mean, fps_effective


def _detect_blinks(ear_series, fps, consec_frames=2):
    """Adaptive-threshold blink detector (matches extract_blinks_mp4)."""
    from scipy.stats import entropy as scipy_entropy

    ear_norm = ((ear_series - ear_series.min()) /
                (np.ptp(ear_series) + 1e-8) * 255).astype(np.uint8)
    _, thresh_img = cv2.threshold(ear_norm.reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
    threshold = ear_series.min() + np.ptp(ear_series) * (thresh_img[0, 0] / 255.0)
    threshold = min(threshold, 0.25)

    closed = (ear_series < threshold).astype(int)
    blinks = []
    in_blink = False
    blink_start = 0
    for i, c in enumerate(closed):
        if c == 1 and not in_blink:
            in_blink = True; blink_start = i
        elif c == 0 and in_blink:
            dur = i - blink_start
            if dur >= consec_frames:
                blinks.append({"start": blink_start, "end": i, "duration_frames": dur})
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


def _blink_feature_vec(stats):
    """Pack blink stats into the 16-d feature matching extract_blinks_mp4."""
    feat = np.array([
        stats.get("blinks_per_min", 0.0) / 30.0,
        stats.get("mean_blink_duration_frames", 0.0) / 10.0,
        stats.get("std_blink_duration_frames", 0.0) / 5.0,
        stats.get("ibi_cv", 0.0),
        stats.get("ibi_mean_frames", 0.0) / 100.0,
        stats.get("ibi_std_frames", 0.0) / 50.0,
        stats.get("ear_mean", 0.3),
        stats.get("ear_std", 0.0),
        stats.get("ear_entropy", 0.0) / 5.0,
        float(stats.get("n_blinks", 0)) / 20.0,
        stats.get("ear_threshold_used", 0.25),
        1.0 if stats.get("n_blinks", 0) == 0 else 0.0,
        min(stats.get("blinks_per_min", 0.0) / 60.0, 1.0),
        stats.get("ear_std", 0.0) / (stats.get("ear_mean", 0.3) + 1e-8),
        0.0,
        0.0,
    ], dtype=np.float32)[:BLINK_FEAT_DIM]
    return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)


def _cleanup_mediapipe():
    """Best-effort teardown to silence the harmless __del__ traceback."""
    try:
        lm = _MP_CACHE.get("landmarker")
        if lm is not None:
            try:
                lm.close()
            except Exception:
                pass
            _MP_CACHE["landmarker"] = None
        fm = _MP_CACHE.get("face_mesh")
        if fm is not None:
            try:
                fm.close()
            except Exception:
                pass
            _MP_CACHE["face_mesh"] = None
        _MP_CACHE["backend"] = None
    except Exception:
        pass


def _blink_worker(task):
    import atexit
    atexit.register(_cleanup_mediapipe)  # silence __del__ traceback at worker exit

    vid_id, class_name, frame_paths, out_dir, fps, max_frames, force = task

    save_dir = Path(out_dir) / class_name / vid_id
    feat_path = save_dir / "blink_feat.npy"
    meta_path = save_dir / "blink_meta.json"

    if not force and feat_path.exists() and meta_path.exists():
        return {"vid": vid_id, "class": class_name, "status": "cached"}

    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        ear, fps_eff = _extract_ear_series_from_frames(frame_paths, max_frames)
    except Exception as e:
        np.save(feat_path, np.zeros(BLINK_FEAT_DIM, dtype=np.float32))
        with open(meta_path, "w") as f:
            json.dump({"status": f"error: {type(e).__name__}: {e}"}, f)
        return {"vid": vid_id, "class": class_name, "status": "error"}

    if ear is None:
        np.save(feat_path, np.zeros(BLINK_FEAT_DIM, dtype=np.float32))
        with open(meta_path, "w") as f:
            json.dump({"status": "no_landmarks"}, f)
        return {"vid": vid_id, "class": class_name, "status": "no_landmarks"}

    stats = _detect_blinks(ear, fps_eff)
    feat = _blink_feature_vec(stats)

    np.save(feat_path, feat)
    with open(meta_path, "w") as f:
        meta = dict(stats); meta["n_frames"] = int(len(ear)); meta["status"] = "ok"
        json.dump(meta, f)

    return {"vid": vid_id, "class": class_name, "status": "ok"}


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main(args):
    scanner = SCANNERS[args.dataset]
    print(f"[{args.mode}][{args.dataset}] scanning {args.data_root}")
    items = scanner(args.data_root)
    print(f"[{args.mode}][{args.dataset}] found {len(items)} video folders/groups")

    if not items:
        print(f"[{args.mode}][{args.dataset}] nothing to do, exiting")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class counts (before subsetting)
    per_class = {}
    for _, cls, _ in items:
        per_class[cls] = per_class.get(cls, 0) + 1
    for cls, n in sorted(per_class.items()):
        print(f"  {cls}: {n}")

    # Sanity-check mode: take first k per class (balanced)
    if args.max_videos is not None:
        per_class_keep = max(1, args.max_videos // max(1, len(per_class)))
        by_class = {}
        for item in items:
            by_class.setdefault(item[1], []).append(item)
        subset = []
        for cls in sorted(by_class):
            subset.extend(by_class[cls][:per_class_keep])
        print(f"[sanity] reduced {len(items)} → {len(subset)} (first {per_class_keep} per class)")
        items = subset

    tasks = [(vid, cls, frames, str(out_dir), args.fps, args.max_frames, args.force)
             for vid, cls, frames in items]

    worker = _rppg_worker if args.mode == "rppg" else _blink_worker
    start = time.time()
    n_ok = n_cached = n_err = 0

    # Blink w/ MediaPipe multi-worker on Kaggle 16 GB can OOM-kill one worker
    # and the BrokenProcessPool kills the whole run. For blink we default to
    # num_workers=1 and also catch BrokenProcessPool/worker exceptions and
    # fall back to in-process single-worker mode so partial progress is kept.
    used_pool = False
    if args.num_workers > 1:
        used_pool = True
        try:
            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                futures = {pool.submit(worker, t): t for t in tasks}
                pbar = tqdm(as_completed(futures), total=len(tasks),
                            desc=f"{args.mode} {args.dataset}")
                for fut in pbar:
                    try:
                        r = fut.result()
                    except Exception as e:
                        # Don't bring down the whole pool for one bad clip
                        print(f"  [WARN] worker exception: {type(e).__name__}: {e}")
                        n_err += 1
                        pbar.set_postfix(ok=n_ok, cached=n_cached, err=n_err)
                        continue
                    if r["status"] == "ok":
                        n_ok += 1
                    elif r["status"] == "cached":
                        n_cached += 1
                    else:
                        n_err += 1
                    pbar.set_postfix(ok=n_ok, cached=n_cached, err=n_err)
        except Exception as e:
            # ProcessPool was torn down (BrokenProcessPool or similar).  Fall
            # back to single-worker so any already-cached results are kept
            # and the remaining clips are processed in-process.
            print(f"  [WARN] ProcessPool failed ({type(e).__name__}: {e}); "
                  f"falling back to single-worker in-process mode")
            # Re-scan output dir so existing feats count as 'cached'
            args.num_workers = 1
            used_pool = False

    if not used_pool:
        for t in tqdm(tasks, desc=f"{args.mode} {args.dataset} (single-worker)"):
            try:
                r = worker(t)
            except Exception as e:
                print(f"  [WARN] {t[0]}: {type(e).__name__}: {e}")
                n_err += 1
                continue
            if r["status"] == "ok":
                n_ok += 1
            elif r["status"] == "cached":
                n_cached += 1
            else:
                n_err += 1

    elapsed = time.time() - start
    print(f"\n[{args.mode}][{args.dataset}] done: ok={n_ok} cached={n_cached} "
          f"err={n_err} in {elapsed / 60:.1f} min")
    print(f"[{args.mode}][{args.dataset}] wrote caches under {out_dir}")

    # Clean up MediaPipe resources (silences __del__ traceback at interpreter exit)
    if args.mode == "blink":
        _cleanup_mediapipe()


def build_parser():
    p = argparse.ArgumentParser(description="Cross-dataset bio-signal extraction")
    p.add_argument("mode", choices=["rppg", "blink"])
    p.add_argument("dataset", choices=["celebdf", "dfdc", "ff"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fps", type=float, default=25.0,
                   help="Target fps. rPPG: 25. Blink: 15.")
    p.add_argument("--max_frames", type=int, default=64,
                   help="Max frames loaded per video. rPPG: 64. Blink: 150.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max_videos", type=int, default=None,
                   help="Sanity-check mode: only process first N videos (balanced across classes).")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
