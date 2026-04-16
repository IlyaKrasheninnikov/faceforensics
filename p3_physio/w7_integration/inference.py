"""
W7: Clean inference API for P3 PhysioNet.

Production-ready inference: input a video path, output a structured forensic report.

Usage:
    python w7_integration/inference.py \
        --video /path/to/video.mp4 \
        --checkpoint ./checkpoints/best_model.pt

    # Batch inference:
    python w7_integration/inference.py \
        --video_dir /path/to/videos \
        --checkpoint ./checkpoints/best_model.pt \
        --out_json ./results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def predict(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    clip_len: int = 64,
    img_size: int = 224,
    fps: float = 15.0,
    calibrator=None,
) -> Dict:
    """
    Run P3 inference on a single video.

    Returns:
    {
        "score": float,          # probability of being fake [0, 1]
        "is_fake": bool,         # decision at 0.5 threshold
        "confidence": str,       # "low" / "medium" / "high"
        "pulse_anomaly": bool,   # rPPG signal anomaly detected
        "blink_anomaly": bool,   # blink pattern anomaly detected
        "mean_bpm": float,       # estimated BPM from video (-1 if not detectable)
        "blinks_per_min": float, # blink rate
        "processing_time_ms": float,
        "video": str,
        "warnings": list[str],   # any caveats
    }
    """
    from w2_model.dataset import load_video_clip
    from w1_setup.extract_rppg import get_face_roi_signals, chrom_method, compute_snr_and_bpm
    from w1_setup.extract_blinks import extract_ear_series, detect_blinks

    t0 = time.time()
    warnings = []

    # ─── Load video clip ──────────────────────────────────────────────────────
    frames = load_video_clip(video_path, clip_len, fps, img_size, start_frame=0)
    if frames is None:
        return {"error": "Could not load video", "video": video_path}

    T = len(frames)
    if T < 30:
        warnings.append(f"Short video ({T} frames at {fps}fps) — results may be unreliable")

    # ─── Explicit rPPG feature ────────────────────────────────────────────────
    roi = get_face_roi_signals(frames, fps)
    combined = (roi["forehead_rgb"] * 0.4 + roi["left_cheek_rgb"] * 0.3 + roi["right_cheek_rgb"] * 0.3)
    pulse = chrom_method(combined, fps)
    pulse_metrics = compute_snr_and_bpm(pulse, fps)

    # Construct 128-d rPPG feature
    from scipy.fft import rfft, rfftfreq
    freqs = rfftfreq(T, d=1.0 / fps)
    fft_mag = np.abs(rfft(pulse))
    mask = (freqs >= 0.5) & (freqs <= 4.0)
    if mask.sum() > 0:
        band_mag = fft_mag[mask]
        indices = np.linspace(0, len(band_mag) - 1, 128)
        rppg_feat = np.interp(indices, np.arange(len(band_mag)), band_mag)
        rppg_feat = rppg_feat / (rppg_feat.max() + 1e-8)
    else:
        rppg_feat = np.zeros(128, dtype=np.float32)
        warnings.append("Could not extract rPPG feature — very short or dark video")

    # ─── Explicit blink feature ───────────────────────────────────────────────
    blink_result = extract_ear_series(video_path, fps, max_frames=600)
    if isinstance(blink_result, dict):
        blink_feat = np.zeros(16, dtype=np.float32)
        blinks_per_min = -1.0
        warnings.append("Could not extract blink features")
    else:
        ear_mean, _, _, _ = blink_result
        blink_stats = detect_blinks(np.array(ear_mean), fps)
        blinks_per_min = blink_stats.get("blinks_per_min", -1.0)
        blink_feat = np.array([
            min(blinks_per_min / 30.0, 2.0),
            blink_stats.get("mean_blink_duration_frames", 0.0) / 10.0,
            blink_stats.get("ibi_cv", 0.0),
            blink_stats.get("ear_mean", 0.3),
            blink_stats.get("ear_std", 0.0),
            blink_stats.get("ear_entropy", 0.0) / 5.0,
            float(blink_stats.get("n_blinks", 0)) / 20.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)[:16]

    # ─── Model inference ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        frames_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, 3, H, W)
        rppg_t = torch.from_numpy(rppg_feat).float().unsqueeze(0).to(device)
        blink_t = torch.from_numpy(blink_feat).float().unsqueeze(0).to(device)

        outputs = model(frames_t, rppg_t, blink_t)
        raw_logit = outputs["logit"].cpu().item()
        score = torch.sigmoid(torch.tensor(raw_logit)).item()

    # Calibration
    if calibrator is not None:
        score = float(calibrator.predict_proba([[raw_logit]])[:, 1])

    # ─── Anomaly flags ────────────────────────────────────────────────────────
    snr_db = pulse_metrics.get("snr_db", -99)
    pulse_anomaly = snr_db < 3.0  # no clear pulse peak → suspicious
    blink_anomaly = blinks_per_min >= 0 and (blinks_per_min < 3 or blinks_per_min > 35)

    if snr_db < -5:
        warnings.append("Very low rPPG SNR — possible heavy compression or makeup")

    # ─── Confidence ───────────────────────────────────────────────────────────
    margin = abs(score - 0.5)
    if margin < 0.15:
        confidence = "low"
    elif margin < 0.35:
        confidence = "medium"
    else:
        confidence = "high"

    processing_ms = (time.time() - t0) * 1000

    return {
        "video": str(video_path),
        "score": round(score, 4),
        "is_fake": score >= 0.5,
        "confidence": confidence,
        "pulse_anomaly": pulse_anomaly,
        "blink_anomaly": blink_anomaly,
        "mean_bpm": round(pulse_metrics.get("bpm", -1), 1),
        "blinks_per_min": round(blinks_per_min, 1),
        "rppg_snr_db": round(snr_db, 2),
        "processing_time_ms": round(processing_ms, 1),
        "warnings": warnings,
    }


def batch_predict(video_dir: str, model, device, **kwargs) -> list:
    """Run inference on all videos in a directory."""
    video_dir = Path(video_dir)
    videos = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))
    print(f"Processing {len(videos)} videos...")

    results = []
    for vid in videos:
        try:
            result = predict(str(vid), model, device, **kwargs)
            results.append(result)
            status = "FAKE" if result.get("is_fake") else "REAL"
            print(f"  [{status} {result.get('score', 0):.3f}] {vid.name}")
        except Exception as e:
            results.append({"video": str(vid), "error": str(e)})

    return results


def print_forensic_report(result: dict):
    """Pretty-print a forensic report."""
    print("\n" + "=" * 60)
    print("  P3 BIO-SIGNAL FORENSICS REPORT")
    print("=" * 60)
    print(f"  Video:       {Path(result['video']).name}")
    verdict = "FAKE" if result.get("is_fake") else "REAL"
    print(f"  Verdict:     {verdict}  (score={result.get('score', '?'):.3f}, confidence={result.get('confidence', '?')})")
    print(f"\n  Physiological Signals:")
    print(f"    Pulse (rPPG):  {result.get('mean_bpm', '?')} BPM  SNR={result.get('rppg_snr_db', '?')} dB  {'⚠ ANOMALY' if result.get('pulse_anomaly') else '✓ OK'}")
    print(f"    Blink rate:    {result.get('blinks_per_min', '?')} blinks/min  {'⚠ ANOMALY' if result.get('blink_anomaly') else '✓ OK'}")
    print(f"\n  Processing:   {result.get('processing_time_ms', '?')} ms")
    if result.get("warnings"):
        print(f"\n  Warnings:")
        for w in result["warnings"]:
            print(f"    ⚠ {w}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="P3 inference — single video or batch")
    p.add_argument("--video", default=None, help="Single video path")
    p.add_argument("--video_dir", default=None, help="Directory of videos")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_json", default=None)
    p.add_argument("--clip_len", type=int, default=64)
    p.add_argument("--fps", type=float, default=15.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from w2_model.model import PhysioNet, ModelConfig
    # weights_only=False required because checkpoints pickle numpy scalars
    # and PyTorch 2.6 made weights_only=True the default.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Checkpoint may be:
    #  (a) full-fusion dict: {"config": ..., "model_state_dict": ...}
    #  (b) linear-probe dict: {"model_state_dict": ...} with no config (v13 style)
    #  (c) bare state_dict (tensors only) — packaged backbone
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
    else:
        # Reconstruct default ModelConfig matching v13 (linear-probe, backbone+rppg+blink)
        print("[inference] No 'config' key in checkpoint — using default v13 ModelConfig")
        cfg = ModelConfig(
            backbone="efficientnet_b4",
            backbone_pretrained=False,
            temporal_pool="mean",
            temporal_dim=0,
            rppg_feature_dim=128,
            blink_feature_dim=16,
            use_pulse_head=False,
            use_blink_head=False,
        )

    model = PhysioNet(cfg).to(device)
    # strict=False — v13 is a linear-probe checkpoint so classifier head keys
    # may differ from a freshly-constructed PhysioNet. Backbone weights load,
    # fusion/classifier is randomly initialized (demo still produces rPPG/blink
    # signal metrics; only the learned "score" is not meaningful without the probe).
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[inference] {len(missing)} missing keys (head will be random): {missing[:3]}...")
    if unexpected:
        print(f"[inference] {len(unexpected)} unexpected keys ignored: {unexpected[:3]}...")

    if args.video:
        result = predict(args.video, model, device, clip_len=args.clip_len, fps=args.fps)
        print_forensic_report(result)
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(result, f, indent=2)

    elif args.video_dir:
        results = batch_predict(args.video_dir, model, device, clip_len=args.clip_len, fps=args.fps)
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved → {args.out_json}")
