"""
W3: Physiological augmentation pipeline.

Implements augmentations that simulate deepfake artifacts to expand training data:
  1. PulseStripAugmentation  — temporal median filter removes rPPG signal
  2. BlinkFreezeAugmentation — eye blink frames replaced with open-eye frames
  3. VideoCompressAugmentation — re-encode at random CRF to simulate compression
  4. TemporalJitterAugmentation — randomly drop/repeat frames

All augmentations are implemented as callable classes compatible with
the dataset pipeline in w2_model/dataset.py.
"""

import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


class PulseStripAugmentation:
    """
    Strips rPPG signal from video frames by temporal median filtering
    on the LAB color space 'a' channel (red-green axis, sensitive to blood flow).

    Input:  (T, H, W, 3) float32 frames [0, 1]
    Output: (T, H, W, 3) float32 frames with pulse removed, float label=1 (fake)
    """

    def __init__(self, window_sec: float = 0.5, fps: float = 15.0, prob: float = 0.3):
        self.window = max(3, int(fps * window_sec))
        self.prob = prob

    def __call__(
        self, frames: np.ndarray, label: float
    ) -> Tuple[np.ndarray, float]:
        if label != 0 or random.random() > self.prob:
            return frames, label

        T = len(frames)
        frames_aug = frames.copy()

        # Pre-convert all frames to LAB
        lab_frames = []
        for f in frames:
            f_uint8 = (f * 255).clip(0, 255).astype(np.uint8)
            lab_frames.append(cv2.cvtColor(f_uint8, cv2.COLOR_RGB2LAB).astype(np.float32))

        # Temporal median of 'a' channel within sliding window
        for t in range(T):
            start = max(0, t - self.window // 2)
            end = min(T, t + self.window // 2 + 1)
            a_window = np.stack([lab_frames[j][:, :, 1] for j in range(start, end)])
            median_a = np.median(a_window, axis=0)

            lab_aug = lab_frames[t].copy()
            lab_aug[:, :, 1] = median_a

            frame_back = cv2.cvtColor(np.clip(lab_aug, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            frames_aug[t] = frame_back.astype(np.float32) / 255.0

        return frames_aug, 1.0  # relabel as fake


class BlinkFreezeAugmentation:
    """
    Removes blinks from real video by replacing closed-eye frames
    with the nearest open-eye frame.

    Requires blink labels (per-frame: 1=closed, 0=open).
    Input:  frames (T, H, W, 3), blink_labels (T,)
    Output: frames with blinks removed, label=1 (fake)
    """

    def __init__(self, prob: float = 0.2, min_blinks: int = 2):
        self.prob = prob
        self.min_blinks = min_blinks

    def __call__(
        self,
        frames: np.ndarray,
        label: float,
        blink_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        if label != 0 or random.random() > self.prob:
            return frames, label

        if blink_labels is None or blink_labels.sum() < self.min_blinks:
            return frames, label

        frames_aug = frames.copy()
        last_open_idx = 0

        for t in range(len(frames)):
            if blink_labels[t] > 0.5:
                frames_aug[t] = frames_aug[last_open_idx]
            else:
                last_open_idx = t

        return frames_aug, 1.0  # relabel as fake


class VideoCompressAugmentation:
    """
    Re-encode frames at random JPEG quality or video CRF to simulate compression artifacts.
    Applied to both real and fake samples.

    Input:  (T, H, W, 3) float32 [0, 1]
    Output: compressed frames (label unchanged)
    """

    def __init__(self, quality_range: Tuple[int, int] = (40, 90), prob: float = 0.3):
        self.quality_range = quality_range
        self.prob = prob

    def __call__(self, frames: np.ndarray, label: float) -> Tuple[np.ndarray, float]:
        if random.random() > self.prob:
            return frames, label

        quality = random.randint(*self.quality_range)
        frames_aug = []

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        for f in frames:
            f_uint8 = (f * 255).clip(0, 255).astype(np.uint8)
            f_bgr = cv2.cvtColor(f_uint8, cv2.COLOR_RGB2BGR)
            _, encoded = cv2.imencode(".jpg", f_bgr, encode_params)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            f_back = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames_aug.append(f_back)

        return np.stack(frames_aug), label


class TemporalJitterAugmentation:
    """
    Randomly drop or repeat frames to simulate temporal inconsistency.
    Applied to both real and fake; label unchanged.

    Input:  (T, H, W, 3) float32
    Output: (T, H, W, 3) float32 (same T, but with jittered content)
    """

    def __init__(self, drop_prob: float = 0.05, repeat_prob: float = 0.05, prob: float = 0.2):
        self.drop_prob = drop_prob
        self.repeat_prob = repeat_prob
        self.prob = prob

    def __call__(self, frames: np.ndarray, label: float) -> Tuple[np.ndarray, float]:
        if random.random() > self.prob:
            return frames, label

        T = len(frames)
        new_frames = []

        for t in range(T):
            r = random.random()
            if r < self.drop_prob and t > 0:
                # Drop: repeat previous
                new_frames.append(new_frames[-1])
            elif r < self.drop_prob + self.repeat_prob and t < T - 1:
                # Repeat current
                new_frames.append(frames[t])
                new_frames.append(frames[t])
            else:
                new_frames.append(frames[t])

        # Trim/pad to original T
        if len(new_frames) > T:
            new_frames = new_frames[:T]
        while len(new_frames) < T:
            new_frames.append(new_frames[-1])

        return np.stack(new_frames), label


class ColorJitterAugmentation:
    """
    Random brightness, contrast, saturation jitter.
    Applied to individual frames.
    """

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, prob: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.prob = prob

    def __call__(self, frames: np.ndarray, label: float) -> Tuple[np.ndarray, float]:
        if random.random() > self.prob:
            return frames, label

        # Apply same transform to all frames in clip (temporal consistency)
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        s = 1.0 + random.uniform(-self.saturation, self.saturation)

        frames_aug = []
        for f in frames:
            f_hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            f_hsv[:, :, 1] = np.clip(f_hsv[:, :, 1] * s, 0, 255)  # saturation
            f_hsv[:, :, 2] = np.clip(f_hsv[:, :, 2] * b, 0, 255)  # brightness
            f_rgb = cv2.cvtColor(np.clip(f_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            f_rgb = np.clip(f_rgb.astype(np.float32) * c / 255.0, 0, 1)
            frames_aug.append(f_rgb)

        return np.stack(frames_aug), label


class AugmentationPipeline:
    """Compose multiple augmentations in sequence."""

    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(
        self,
        frames: np.ndarray,
        label: float,
        blink_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        for aug in self.augmentations:
            if isinstance(aug, BlinkFreezeAugmentation):
                frames, label = aug(frames, label, blink_labels)
            else:
                frames, label = aug(frames, label)
        return frames, label


def build_default_pipeline(fps: float = 15.0) -> AugmentationPipeline:
    """Build the standard P3 augmentation pipeline."""
    return AugmentationPipeline([
        ColorJitterAugmentation(prob=0.5),
        VideoCompressAugmentation(prob=0.3),
        TemporalJitterAugmentation(prob=0.2),
        PulseStripAugmentation(fps=fps, prob=0.3),
        BlinkFreezeAugmentation(prob=0.2),
    ])


if __name__ == "__main__":
    # Quick test
    T, H, W = 32, 224, 224
    frames = np.random.rand(T, H, W, 3).astype(np.float32)
    blinks = (np.random.rand(T) > 0.9).astype(np.float32)

    pipeline = build_default_pipeline()

    frames_aug, new_label = pipeline(frames, 0.0, blinks)
    print(f"Input: frames {frames.shape}, label=0.0")
    print(f"Output: frames {frames_aug.shape}, label={new_label}")
    print("Augmentation pipeline OK")
