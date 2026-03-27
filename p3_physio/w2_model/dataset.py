"""
W2: PhysioDeepfakeDataset — video clip dataset with physiological feature extraction.

Supports:
  - FaceForensics++ (FF++) with multiple manipulation types
  - CelebDF-v2
  - Identity-aware splitting (prevents leakage)
  - Lazy per-video rPPG and blink extraction with disk caching
  - Robust error handling (no worker crashes)
"""

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.extract_rppg import get_face_roi_signals, chrom_method, pos_method, compute_snr_and_bpm
from w1_setup.extract_blinks import extract_ear_series, detect_blinks


# ─── Constants ────────────────────────────────────────────────────────────────

FF_MANIPULATION_TYPES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
    "DeepFakeDetection": 1,
}

CELEBDF_REAL_DIR = "real"
CELEBDF_FAKE_DIR = "synthesis"

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Feature extraction helpers ───────────────────────────────────────────────

def frames_to_rppg_feature(frames: np.ndarray, fps: float = 15.0, feat_dim: int = 128) -> np.ndarray:
    """Extract rPPG FFT spectrum feature from frames.
    Returns: (feat_dim,) normalized FFT magnitude in 0.5–4 Hz band."""
    if len(frames) < 15:
        return np.zeros(feat_dim, dtype=np.float32)
    try:
        roi = get_face_roi_signals(frames, fps)
        combined = (
            roi["forehead_rgb"] * 0.4
            + roi["left_cheek_rgb"] * 0.3
            + roi["right_cheek_rgb"] * 0.3
        )
        pulse = chrom_method(combined, fps)
        from scipy.fft import rfft, rfftfreq
        T = len(pulse)
        freqs = rfftfreq(T, d=1.0 / fps)
        fft_mag = np.abs(rfft(pulse))
        mask = (freqs >= 0.5) & (freqs <= 4.0)
        if mask.sum() == 0:
            return np.zeros(feat_dim, dtype=np.float32)
        band_mag = fft_mag[mask]
        indices = np.linspace(0, len(band_mag) - 1, feat_dim)
        feat = np.interp(indices, np.arange(len(band_mag)), band_mag)
        feat = feat / (feat.max() + 1e-8)
        return feat.astype(np.float32)
    except Exception:
        return np.zeros(feat_dim, dtype=np.float32)


def frames_to_blink_feature(video_path: str, fps: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """Extract blink stats feature (16-d) and per-frame blink labels."""
    try:
        result = extract_ear_series(video_path, target_fps=fps, max_frames=600)
        if isinstance(result, dict):
            return np.zeros(16, dtype=np.float32), np.zeros(1, dtype=np.float32)
        ear_mean, ears_left, ears_right, fps_used = result
        blink_stats = detect_blinks(np.array(ear_mean), fps_used)
        feat = np.array([
            blink_stats.get("blinks_per_min", 0.0) / 30.0,
            blink_stats.get("mean_blink_duration_frames", 0.0) / 10.0,
            blink_stats.get("ibi_cv", 0.0),
            blink_stats.get("ear_mean", 0.3),
            blink_stats.get("ear_std", 0.0),
            blink_stats.get("ear_entropy", 0.0) / 5.0,
            float(blink_stats.get("n_blinks", 0)) / 20.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)[:16]
        ear_arr = np.array(ear_mean)
        threshold = blink_stats.get("ear_threshold_used", 0.25)
        blink_labels = (ear_arr < threshold).astype(np.float32)
        return feat, blink_labels
    except Exception:
        return np.zeros(16, dtype=np.float32), np.zeros(1, dtype=np.float32)


# ─── Video Loader ─────────────────────────────────────────────────────────────

def load_video_clip(
    video_path: str,
    clip_len: int = 32,
    fps_target: float = 15.0,
    img_size: int = 224,
    start_frame: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load a clip of `clip_len` frames from video.
    Returns: (clip_len, H, W, 3) float32 [0,1], or None on error."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(orig_fps / fps_target))

        needed_raw = clip_len * step
        if total < needed_raw:
            start = 0
        else:
            start = start_frame if start_frame is not None else random.randint(0, total - needed_raw)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        idx = 0
        while len(frames) < clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (img_size, img_size))
                frames.append(frame.astype(np.float32) / 255.0)
            idx += 1
        cap.release()

        if len(frames) == 0:
            return None
        while len(frames) < clip_len:
            frames.append(frames[-1])
        return np.stack(frames[:clip_len], axis=0)
    except Exception:
        return None


# ─── Unique cache key ─────────────────────────────────────────────────────────

def _cache_key(video_path: str) -> str:
    """Generate a unique cache key from the full path.
    Avoids collision between original/000.mp4 and Deepfakes/000.mp4."""
    # Use last 2 path components: "Deepfakes/000" or "original/000"
    p = Path(video_path)
    return f"{p.parent.name}_{p.stem}"


# ─── Main Dataset ─────────────────────────────────────────────────────────────

class PhysioDeepfakeDataset(Dataset):
    """Dataset for deepfake detection with physiological features."""

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        clip_len: int = 32,
        img_size: int = 224,
        fps: float = 15.0,
        cache_dir: str = "./logs/signal_cache",
        fallback_cache_dirs: Optional[List[str]] = None,
        augment: bool = True,
        rppg_feat_dim: int = 128,
        blink_feat_dim: int = 16,
        skip_physio: bool = False,
    ):
        assert len(video_paths) == len(labels)
        self.video_paths = video_paths
        self.labels = labels
        self.clip_len = clip_len
        self.img_size = img_size
        self.fps = fps
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fallback_cache_dirs = [Path(d) for d in (fallback_cache_dirs or []) if Path(d).exists()]
        self.augment = augment
        self.rppg_feat_dim = rppg_feat_dim
        self.blink_feat_dim = blink_feat_dim
        self.skip_physio = skip_physio

    def __len__(self):
        return len(self.video_paths)

    def _cache_path(self, video_path: str, suffix: str) -> Path:
        key = _cache_key(video_path)
        return self.cache_dir / f"{key}_{suffix}.npy"

    def _find_cache(self, video_path: str, suffix: str) -> Optional[Path]:
        primary = self._cache_path(video_path, suffix)
        if primary.exists():
            return primary
        key = _cache_key(video_path)
        for fb in self._fallback_cache_dirs:
            candidate = fb / f"{key}_{suffix}.npy"
            if candidate.exists():
                return candidate
            # Also check old-style cache (stem only) for backward compat
            old_candidate = fb / f"{Path(video_path).stem}_{suffix}.npy"
            if old_candidate.exists():
                return old_candidate
        return None

    def _get_rppg_feat(self, video_path: str, frames: np.ndarray) -> np.ndarray:
        cached = self._find_cache(video_path, "rppg")
        if cached is not None:
            try:
                return np.load(str(cached))
            except Exception:
                pass
        feat = frames_to_rppg_feature(frames, self.fps, self.rppg_feat_dim)
        try:
            np.save(str(self._cache_path(video_path, "rppg")), feat)
        except OSError:
            pass
        return feat

    def _get_blink_data(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        cached_feat = self._find_cache(video_path, "blink_feat")
        cached_labels = self._find_cache(video_path, "blink_labels")
        if cached_feat is not None and cached_labels is not None:
            try:
                return np.load(str(cached_feat)), np.load(str(cached_labels))
            except Exception:
                pass
        feat, labels = frames_to_blink_feature(video_path, self.fps)
        try:
            np.save(str(self._cache_path(video_path, "blink_feat")), feat)
            np.save(str(self._cache_path(video_path, "blink_labels")), labels)
        except OSError:
            pass
        return feat, labels

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load frames with full error recovery
        frames = load_video_clip(video_path, self.clip_len, self.fps, self.img_size)
        if frames is None:
            frames = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)

        # Physio features
        if self.skip_physio:
            rppg_feat = np.zeros(self.rppg_feat_dim, dtype=np.float32)
            blink_feat = np.zeros(self.blink_feat_dim, dtype=np.float32)
            blink_labels = np.zeros(self.clip_len, dtype=np.float32)
        else:
            rppg_feat = self._get_rppg_feat(video_path, frames)
            blink_feat, blink_labels = self._get_blink_data(video_path)
            if len(blink_labels) >= self.clip_len:
                blink_labels = blink_labels[:self.clip_len]
            else:
                blink_labels = np.pad(blink_labels, (0, self.clip_len - len(blink_labels)))

        # ImageNet normalization
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        # (T, H, W, 3) → (T, 3, H, W)
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

        return {
            "frames": frames_t,
            "rppg_feat": torch.from_numpy(rppg_feat),
            "blink_feat": torch.from_numpy(blink_feat),
            "blink_labels": torch.from_numpy(blink_labels),
            "label": torch.tensor(float(label), dtype=torch.float32),
            "video_path": video_path,
        }


# ─── Dataset builders ─────────────────────────────────────────────────────────

def build_ff_plus_plus_list(ff_root: str, compression: str = "c23") -> Tuple[List, List]:
    """Scan FF++ directory structure and return (video_paths, labels)."""
    ff_root = Path(ff_root)
    video_paths, labels = [], []
    for manip_type, label in FF_MANIPULATION_TYPES.items():
        candidates = [
            ff_root / manip_type,
            ff_root / manip_type / compression / "videos",
        ]
        for vid_dir in candidates:
            if vid_dir.exists():
                vids = sorted(list(vid_dir.glob("*.mp4")))
                if vids:
                    video_paths.extend([str(v) for v in vids])
                    labels.extend([label] * len(vids))
                    print(f"  {manip_type}: {len(vids)} videos ({vid_dir})")
                    break
    return video_paths, labels


def build_celebdf_list(celebdf_root: str) -> Tuple[List, List]:
    root = Path(celebdf_root)
    video_paths, labels = [], []
    for subdir, label in [(CELEBDF_REAL_DIR, 0), (CELEBDF_FAKE_DIR, 1)]:
        d = root / subdir
        if d.exists():
            vids = sorted(list(d.glob("*.mp4")))
            video_paths.extend([str(v) for v in vids])
            labels.extend([label] * len(vids))
    return video_paths, labels


def _extract_source_id(video_path: str) -> str:
    """Extract source identity from FF++ video filename."""
    stem = Path(video_path).stem
    return stem.split("_")[0]


def build_dataloaders(
    ff_root: Optional[str] = None,
    celebdf_root: Optional[str] = None,
    dfdc_root: Optional[str] = None,
    cache_dir: str = "./logs/signal_cache",
    fallback_cache_dirs: Optional[List[str]] = None,
    clip_len: int = 32,
    img_size: int = 224,
    batch_size: int = 4,
    num_workers: int = 2,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
    augment_train: bool = True,
    skip_physio: bool = False,
    max_train_samples: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders with identity-aware splitting."""
    random.seed(seed)
    np.random.seed(seed)

    all_paths, all_labels = [], []

    if ff_root:
        p, l = build_ff_plus_plus_list(ff_root)
        all_paths.extend(p); all_labels.extend(l)
        print(f"FF++: {len(p)} videos (real={l.count(0)}, fake={l.count(1)})")

    if celebdf_root:
        p, l = build_celebdf_list(celebdf_root)
        all_paths.extend(p); all_labels.extend(l)
        print(f"CelebDF: {len(p)} videos (real={l.count(0)}, fake={l.count(1)})")

    if not all_paths:
        raise ValueError("No datasets found!")

    # Identity-aware split
    id_to_indices = {}
    for i, path in enumerate(all_paths):
        src_id = _extract_source_id(path)
        id_to_indices.setdefault(src_id, []).append(i)

    unique_ids = sorted(id_to_indices.keys())
    random.shuffle(unique_ids)

    n_ids = len(unique_ids)
    n_train_ids = int(n_ids * train_split)
    n_val_ids = int(n_ids * val_split)

    train_ids = set(unique_ids[:n_train_ids])
    val_ids = set(unique_ids[n_train_ids:n_train_ids + n_val_ids])

    train_idx, val_idx, test_idx = [], [], []
    for src_id, indices in id_to_indices.items():
        if src_id in train_ids:
            train_idx.extend(indices)
        elif src_id in val_ids:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    test_paths = [all_paths[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    print(f"\nIdentity-aware split: {n_train_ids} train / {n_val_ids} val / {n_ids - n_train_ids - n_val_ids} test source IDs")

    fb = fallback_cache_dirs
    train_ds = PhysioDeepfakeDataset(train_paths, train_labels, clip_len, img_size,
                                      augment=augment_train, cache_dir=cache_dir,
                                      fallback_cache_dirs=fb, skip_physio=skip_physio)
    val_ds = PhysioDeepfakeDataset(val_paths, val_labels, clip_len, img_size,
                                    augment=False, cache_dir=cache_dir,
                                    fallback_cache_dirs=fb, skip_physio=skip_physio)
    test_ds = PhysioDeepfakeDataset(test_paths, test_labels, clip_len, img_size,
                                     augment=False, cache_dir=cache_dir,
                                     fallback_cache_dirs=fb, skip_physio=skip_physio)

    # Balanced sampler
    train_labels_arr = np.array(train_labels)
    n_real = (train_labels_arr == 0).sum()
    n_fake = (train_labels_arr == 1).sum()
    weights = np.where(train_labels_arr == 0, 1.0 / (n_real + 1), 1.0 / (n_fake + 1))

    # Cap samples per epoch if requested (for Kaggle time limits)
    num_samples = len(weights)
    if max_train_samples > 0:
        num_samples = min(num_samples, max_train_samples)
    sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

    # DataLoader kwargs — timeout only for multi-worker
    dl_kwargs = dict(pin_memory=True)
    if num_workers > 0:
        dl_kwargs["timeout"] = 300
        dl_kwargs["persistent_workers"] = True

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, **dl_kwargs)

    n_real_int, n_fake_int = int(n_real), int(n_fake)
    train_dl.class_ratio = n_fake_int / max(n_real_int, 1)

    print(f"Dataset: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"Balance: {n_real_int} real / {n_fake_int} fake (ratio={train_dl.class_ratio:.2f})")
    print(f"Sampler: {num_samples} samples/epoch, {num_samples // batch_size} steps/epoch")
    return train_dl, val_dl, test_dl
