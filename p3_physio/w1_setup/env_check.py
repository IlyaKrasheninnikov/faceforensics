"""
W1: Environment verification script.
Run this first to confirm all dependencies are installed correctly.
Usage: python env_check.py
"""

import sys
import subprocess

REQUIRED = {
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "timm": "0.9.0",
    "cv2": "4.7.0",
    "scipy": "1.10.0",
    "sklearn": "1.2.0",
    "numpy": "1.24.0",
    "pandas": "2.0.0",
    "albumentations": "1.3.0",
    "matplotlib": "3.7.0",
    "tqdm": "4.65.0",
    "yaml": "0.0.0",
}

OPTIONAL = {
    "mediapipe": "0.10.0",   # optional: improves face ROI precision; falls back to OpenCV Haar
    "trackio": "0.1.0",
    "rppg_toolbox": "0.1.0",
    "mamba_ssm": "1.0.0",
}


def check_package(name: str, min_version: str, optional: bool = False) -> bool:
    import importlib
    import pkg_resources

    # mediapipe imports as mediapipe, cv2 as opencv-python, etc.
    import_name = name.replace("-", "_")

    try:
        mod = importlib.import_module(import_name)
        try:
            installed = pkg_resources.get_distribution(
                name.replace("cv2", "opencv-python").replace("sklearn", "scikit-learn").replace("yaml", "PyYAML")
            ).version
        except Exception:
            installed = getattr(mod, "__version__", "unknown")
        tag = "[OK]" if not optional else "[OK-optional]"
        print(f"  {tag:15s} {name:<25s} {installed}")
        return True
    except ImportError:
        tag = "[MISSING]" if not optional else "[MISSING-optional]"
        print(f"  {tag:15s} {name:<25s} — install: pip install {name.replace('cv2','opencv-python').replace('sklearn','scikit-learn')}")
        return not optional  # only fail for required


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  [GPU]  CUDA {torch.version.cuda} | Device: {torch.cuda.get_device_name(0)}")
            print(f"         VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("\n  [CPU]  No CUDA GPU detected — training will be VERY slow. Use Colab/Kaggle GPU.")
    except ImportError:
        print("\n  [WARN] torch not installed — cannot check CUDA")


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    status = "[OK]" if ok else "[WARN]"
    print(f"  {status:15s} Python {v.major}.{v.minor}.{v.micro} (need ≥3.9)")
    return ok


def main():
    print("=" * 60)
    print("  P3 Bio-Signal Forensics — Environment Check")
    print("=" * 60)

    print("\n[Python]")
    check_python()

    print("\n[Required packages]")
    all_ok = all(check_package(name, ver) for name, ver in REQUIRED.items())

    print("\n[Optional packages]")
    for name, ver in OPTIONAL.items():
        check_package(name, ver, optional=True)

    check_cuda()

    print("\n[Face detection backend]")
    try:
        import mediapipe as mp
        _ = mp.solutions.face_mesh.FaceMesh
        print("  [OK]            MediaPipe legacy solutions API  (best precision)")
    except Exception:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode  # noqa
            print("  [OK]            MediaPipe Tasks API (FaceLandmarker) — Kaggle/Colab >= 0.10")
            print("                  face_landmarker.task model will be downloaded on first run (~3MB)")
        except Exception:
            try:
                import mediapipe  # noqa
                print("  [WARN]          MediaPipe installed but neither solutions nor Tasks API available")
                print("                  → will use brightness heuristic fallback (reduced accuracy)")
            except ImportError:
                print("  [WARN]          MediaPipe not installed → brightness heuristic fallback")
    print("  Note: Tasks API backend gives full landmark accuracy on Kaggle (mediapipe 0.10.x)")

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓  Environment OK — ready to run W1 scripts")
    else:
        print("  ✗  Some required packages missing — install via:")
        print("     pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
