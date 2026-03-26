"""
W2: PhysioDeepfakeDataset — video clip dataset with physiological feature extraction.

Supports:
  - FaceForensics++ (FF++) with multiple manipulation types
  - CelebDF-v2
  - DFDC (preview)
  - Lazy per-video rPPG and blink extraction with disk caching
  - Physio-specific augmentations: pulse stripping, blink freezing

Usage:
    from w2_model.dataset import PhysioDeepfakeDataset, build_dataloaders

    train_dl, val_dl = build_dataloaders(
        ff_root="/data/FF++",
        celebdf_root="/data/CelebDF-v2",
        cache_dir="./logs/signal_cache",
        clip_len=64,
        batch_size=8,
    )
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Import signal extractors from W1 (they handle MediaPipe compatibility internally)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from w1_setup.extract_rppg import get_face_roi_signals, chrom_method, pos_method, compute_snr_and_bpm
from w1_setup.extract_blinks import extract_ear_series, detect_blinks


# ─── Constants ────────────────────────────────────────────────────────────────

# FF++ manipulation type → label
FF_MANIPULATION_TYPES = {
    "original": 0,          # real
    "Deepfakes": 1,          # fake
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
    "FaceShifter": 1,
    "DeepFakeDetection": 1,  # additional FF++ category
}

CELEBDF_REAL_DIR = "real"
CELEBDF_FAKE_DIR = "synthesis"


# ─── Feature extraction helpers ───────────────────────────────────────────────

def frames_to_rppg_feature(frames: np.ndarray, fps: float = 15.0, feat_dim: int = 128) -> np.ndarray:
    """
    Extract rPPG FFT spectrum feature from frames.
    Returns: (feat_dim,) normalized FFT magnitude in 0.5–4 Hz band.
    """
    if len(frames) < 15:
        return np.zeros(feat_dim, dtype=np.float32)

    roi = get_face_roi_signals(frames, fps)
    combined = (
        roi["forehead_rgb"] * 0.4
        + roi["left_cheek_rgb"] * 0.3
        + roi["right_cheek_rgb"] * 0.3
    )
    pulse = chrom_method(combined, fps)

    # FFT in physiological band
    from scipy.fft import rfft, rfftfreq
    T = len(pulse)
    freqs = rfftfreq(T, d=1.0 / fps)
    fft_mag = np.abs(rfft(pulse))

    # Interpolate to fixed feat_dim within [0.5, 4] Hz
    mask = (freqs >= 0.5) & (freqs <= 4.0)
    if mask.sum() == 0:
        return np.zeros(feat_dim, dtype=np.float32)

    band_mag = fft_mag[mask]
    # Resample to feat_dim
    indices = np.linspace(0, len(band_mag) - 1, feat_dim)
    feat = np.interp(indices, np.arange(len(band_mag)), band_mag)
    # Normalize
    feat = feat / (feat.max() + 1e-8)
    return feat.astype(np.float32)


def frames_to_blink_feature(video_path: str, fps: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract blink stats feature (16-d) and per-frame blink labels from video path.
    Returns: (blink_feat: (16,), blink_labels: (T,))
    """
    result = extract_ear_series(video_path, target_fps=fps, max_frames=600)
    if isinstance(result, dict):  # error case
        return np.zeros(16, dtype=np.float32), np.zeros(1, dtype=np.float32)

    ear_mean, ears_left, ears_right, fps_used = result
    blink_stats = detect_blinks(np.array(ear_mean), fps_used)

    # Encode into 16-d feature vector
    feat = np.array([
        blink_stats.get("blinks_per_min", 0.0) / 30.0,         # normalize by max expected
        blink_stats.get("mean_blink_duration_frames", 0.0) / 10.0,
        blink_stats.get("ibi_cv", 0.0),
        blink_stats.get("ear_mean", 0.3),
        blink_stats.get("ear_std", 0.0),
        blink_stats.get("ear_entropy", 0.0) / 5.0,
        float(blink_stats.get("n_blinks", 0)) / 20.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          # reserved for future features
    ], dtype=np.float32)[:16]

    # Per-frame blink labels (eye closed = 1)
    ear_arr = np.array(ear_mean)
    threshold = blink_stats.get("ear_threshold_used", 0.25)
    blink_labels = (ear_arr < threshold).astype(np.float32)

    return feat, blink_labels


# ─── Augmentations ────────────────────────────────────────────────────────────

def pulse_strip_augmentation(frames: np.ndarray, fps: float = 15.0) -> np.ndarray:
    """
    Remove rPPG signal from real video frames by temporal median filtering on LAB-a channel.
    This creates a "fake-looking" real video for training augmentation.
    """
    frames_aug = frames.copy()
    window = max(3, int(fps * 0.5))  # ~0.5s window

    for i in range(frames.shape[0]):
        frame_lab = cv2.cvtColor((frames[i] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        start = max(0, i - window // 2)
        end = min(len(frames), i + window // 2 + 1)
        # Median over time window in a-channel to flatten pulse variations
        a_window = np.array([(cv2.cvtColor((frames[j] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[:, :, 1]) for j in range(start, end)])
        frame_lab[:, :, 1] = np.median(a_window, axis=0)
        frame_back = cv2.cvtColor(np.clip(frame_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        frames_aug[i] = frame_back.astype(np.float32) / 255.0

    return frames_aug


def blink_freeze_augmentation(frames: np.ndarray, blink_labels: np.ndarray) -> np.ndarray:
    """
    Remove blinks from real video by copying the last open-eye frame over blink frames.
    Creates a fake-looking video with no blinks.
    """
    frames_aug = frames.copy()
    last_open_idx = 0

    for i in range(len(frames)):
        if blink_labels[i] > 0.5:
            frames_aug[i] = frames_aug[last_open_idx]
        else:
            last_open_idx = i

    return frames_aug


# ─── Video Loader ─────────────────────────────────────────────────────────────

def load_video_clip(
    video_path: str,
    clip_len: int = 64,
    fps_target: float = 15.0,
    img_size: int = 224,
    start_frame: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load a clip of `clip_len` frames from video.
    Returns: (clip_len, H, W, 3) float32 [0,1], or None on error.
    """
    cap = cv2.VideoCapture(video_path)
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

    if len(frames) < clip_len:
        if len(frames) == 0:
            return None
        # Pad by repeating last frame
        while len(frames) < clip_len:
            frames.append(frames[-1])

    return np.stack(frames[:clip_len], axis=0)  # (T, H, W, 3)


# ─── Main Dataset ─────────────────────────────────────────────────────────────

class PhysioDeepfakeDataset(Dataset):
    """
    Dataset yielding:
        frames:      (T, 3, H, W) float32 tensor
        rppg_feat:   (128,) float32 tensor — explicit rPPG feature
        blink_feat:  (16,) float32 tensor — explicit blink stats
        blink_labels:(T,) float32 tensor — per-frame eye-closed GT
        label:       float32 {0=real, 1=fake}

    Signal features are cached to disk to avoid recomputation.
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        clip_len: int = 64,
        img_size: int = 224,
        fps: float = 15.0,
        cache_dir: str = "./logs/signal_cache",
        fallback_cache_dirs: Optional[List[str]] = None,
        augment: bool = True,
        pulse_strip_prob: float = 0.3,
        blink_freeze_prob: float = 0.2,
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
        # Fallback: read-only cache dirs (e.g. Kaggle /kaggle/input/...)
        self._fallback_cache_dirs = [Path(d) for d in (fallback_cache_dirs or []) if Path(d).exists()]
        self.augment = augment
        self.pulse_strip_prob = pulse_strip_prob
        self.blink_freeze_prob = blink_freeze_prob
        self.rppg_feat_dim = rppg_feat_dim
        self.blink_feat_dim = blink_feat_dim
        self.skip_physio = skip_physio  # Skip rPPG/blink extraction (return zeros)

    def __len__(self):
        return len(self.video_paths)

    def _cache_path(self, video_path: str, suffix: str) -> Path:
        name = Path(video_path).stem
        return self.cache_dir / f"{name}_{suffix}.npy"

    def _find_cache(self, video_path: str, suffix: str) -> Optional[Path]:
        """Check primary cache_dir, then any fallback read-only cache dirs."""
        primary = self._cache_path(video_path, suffix)
        if primary.exists():
            return primary
        # Check fallback dirs (e.g. Kaggle read-only input datasets)
        for fb in self._fallback_cache_dirs:
            candidate = fb / f"{Path(video_path).stem}_{suffix}.npy"
            if candidate.exists():
                return candidate
        return None

    def _get_rppg_feat(self, video_path: str, frames: np.ndarray) -> np.ndarray:
        cached = self._find_cache(video_path, "rppg")
        if cached is not None:
            return np.load(str(cached))
        feat = frames_to_rppg_feature(frames, self.fps, self.rppg_feat_dim)
        try:
            np.save(str(self._cache_path(video_path, "rppg")), feat)
        except OSError:
            pass  # read-only filesystem, skip caching
        return feat

    def _get_blink_data(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        cached_feat = self._find_cache(video_path, "blink_feat")
        cached_labels = self._find_cache(video_path, "blink_labels")
        if cached_feat is not None and cached_labels is not None:
            return np.load(str(cached_feat)), np.load(str(cached_labels))
        feat, labels = frames_to_blink_feature(video_path, self.fps)
        try:
            np.save(str(self._cache_path(video_path, "blink_feat")), feat)
            np.save(str(self._cache_path(video_path, "blink_labels")), labels)
        except OSError:
            pass  # read-only filesystem, skip caching
        return feat, labels

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load frames
        frames = load_video_clip(video_path, self.clip_len, self.fps, self.img_size)
        if frames is None:
            # Return zeros on error
            frames = np.zeros((self.clip_len, self.img_size, self.img_size, 3), dtype=np.float32)

        # Explicit features (skip heavy MediaPipe extraction when not needed)
        if self.skip_physio:
            rppg_feat = np.zeros(self.rppg_feat_dim, dtype=np.float32)
            blink_feat = np.zeros(self.blink_feat_dim, dtype=np.float32)
            blink_labels = np.zeros(self.clip_len, dtype=np.float32)
        else:
            rppg_feat = self._get_rppg_feat(video_path, frames)
            blink_feat, blink_labels = self._get_blink_data(video_path)
            # Align blink labels to clip_len
            if len(blink_labels) >= self.clip_len:
                blink_labels = blink_labels[:self.clip_len]
            else:
                blink_labels = np.pad(blink_labels, (0, self.clip_len - len(blink_labels)))

        # Augmentations (only for real samples → relabel as fake)
        aug_label = float(label)
        if self.augment and label == 0:  # real video
            if random.random() < self.pulse_strip_prob:
                frames = pulse_strip_augmentation(frames, self.fps)
                aug_label = 1.0  # treat as fake for training

            elif random.random() < self.blink_freeze_prob:
                frames = blink_freeze_augmentation(frames, blink_labels)
                aug_label = 1.0  # treat as fake

        # ImageNet normalization — CRITICAL for pretrained EfficientNet-B4.
        # Without this, backbone features are garbage (raw [0,1] pixels ≠ expected distribution).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frames = (frames - mean) / std

        # Convert frames (T, H, W, 3) → (T, 3, H, W)
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

        return {
            "frames": frames_t,
            "rppg_feat": torch.from_numpy(rppg_feat),
            "blink_feat": torch.from_numpy(blink_feat),
            "blink_labels": torch.from_numpy(blink_labels),
            "label": torch.tensor(aug_label, dtype=torch.float32),
            "video_path": video_path,
        }


# ─── Dataset builders ─────────────────────────────────────────────────────────

def build_ff_plus_plus_list(ff_root: str, compression: str = "c23") -> Tuple[List, List]:
    """Scan FF++ directory structure and return (video_paths, labels).

    Supports multiple layouts:
      - Flat:   ff_root/original/*.mp4, ff_root/Deepfakes/*.mp4  (Kaggle xdxd003)
      - Nested: ff_root/original/c23/videos/*.mp4                (standard FF++)
      - Alt:    ff_root/original_sequences/youtube/c23/videos/*.mp4
    """
    ff_root = Path(ff_root)
    video_paths, labels = [], []

    for manip_type, label in FF_MANIPULATION_TYPES.items():
        # Try flat layout first (*.mp4 directly in folder), then nested
        candidates = [
            ff_root / manip_type,                          # flat
            ff_root / manip_type / compression / "videos", # nested
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
    """Scan CelebDF-v2 directory and return (video_paths, labels)."""
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
    """Extract source identity from FF++ video filename.

    FF++ naming conventions:
      original/000.mp4           → source id "000"
      Deepfakes/000_003.mp4      → source id "000"
      Face2Face/000.mp4           → source id "000"
      FaceSwap/000_003.mp4        → source id "000"

    The first numeric part before '_' (or the whole stem) is the source identity.
    All manipulations of the same source person share this ID.
    """
    stem = Path(video_path).stem  # e.g. "000_003" or "000"
    return stem.split("_")[0]


def build_dataloaders(
    ff_root: Optional[str] = None,
    celebdf_root: Optional[str] = None,
    dfdc_root: Optional[str] = None,
    cache_dir: str = "./logs/signal_cache",
    fallback_cache_dirs: Optional[List[str]] = None,
    clip_len: int = 64,
    img_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
    augment_train: bool = True,
    skip_physio: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders with IDENTITY-AWARE splitting.

    Critical: FF++ videos from the same source person (e.g. original/000.mp4
    and Deepfakes/000_003.mp4) must ALL go into the same split. Otherwise the
    model learns face identity instead of manipulation artifacts, causing
    val AUC < 0.5 (anti-learning).
    """
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
        raise ValueError("No datasets found! Provide at least one of: ff_root, celebdf_root, dfdc_root")

    # ── Identity-aware split ──────────────────────────────────────────────
    # Group videos by source identity so the same person's real + fake
    # versions all land in the same split (prevents identity leakage).
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
    test_ids = set(unique_ids[n_train_ids + n_val_ids:])

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

    print(f"\nIdentity-aware split: {n_train_ids} train / {n_val_ids} val / {len(test_ids)} test source IDs")
    print(f"  (no identity overlap between splits)")

    fb = fallback_cache_dirs
    train_ds = PhysioDeepfakeDataset(train_paths, train_labels, clip_len, img_size, augment=augment_train, cache_dir=cache_dir, fallback_cache_dirs=fb, skip_physio=skip_physio)
    val_ds = PhysioDeepfakeDataset(val_paths, val_labels, clip_len, img_size, augment=False, cache_dir=cache_dir, fallback_cache_dirs=fb, skip_physio=skip_physio)
    test_ds = PhysioDeepfakeDataset(test_paths, test_labels, clip_len, img_size, augment=False, cache_dir=cache_dir, fallback_cache_dirs=fb, skip_physio=skip_physio)

    # Balanced sampler for training
    train_labels_arr = np.array(train_labels)
    n_real = (train_labels_arr == 0).sum()
    n_fake = (train_labels_arr == 1).sum()
    weights = np.where(train_labels_arr == 0, 1.0 / (n_real + 1), 1.0 / (n_fake + 1))
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # drop_last=True avoids partial batches (important with small batch sizes)
    # prefetch_factor=2 limits memory buffering; persistent_workers=False forces cleanup
    dl_kwargs = dict(num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)

    # Expose class ratio so train.py can auto-compute pos_weight
    n_real_int, n_fake_int = int(n_real), int(n_fake)
    train_dl.class_ratio = n_fake_int / max(n_real_int, 1)

    print(f"Dataset splits: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"Class balance: {n_real_int} real / {n_fake_int} fake (ratio={train_dl.class_ratio:.2f})")
    return train_dl, val_dl, test_dl
