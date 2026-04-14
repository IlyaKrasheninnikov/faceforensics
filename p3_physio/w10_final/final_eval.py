"""
W10: One-command final evaluation — runs all evaluation steps and produces the
complete results package for thesis submission.

Usage:
    python w10_final/final_eval.py \
        --ff_root /kaggle/input/.../frames \
        --resume_ckpt /kaggle/input/.../physio_v13_best.pt \
        --rppg_cache /kaggle/input/.../rppg_v2_300 \
        --blink_cache /kaggle/input/.../blink \
        --celebdf_root /kaggle/input/.../crop \
        --out_dir /kaggle/working/final
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"  {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def main(args):
    base = Path(__file__).parent.parent
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = []

    # Step 1: Ablation (W5)
    ablation_json = out / "ablation" / "ablation_results.json"
    if not ablation_json.exists():
        steps.append(("W5 Ablation", [
            sys.executable, str(base / "w5_ablation" / "ablation_runner.py"),
            "--ff_root", args.ff_root,
            "--resume_ckpt", args.resume_ckpt,
            "--rppg_cache", args.rppg_cache or "",
            "--blink_cache", args.blink_cache or "",
            "--rppg_version", "2", "--rppg_dim", "12",
            "--out_dir", str(out / "ablation"),
            "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
        ]))
    else:
        print(f"[SKIP] Ablation results exist: {ablation_json}")

    # Step 2: Cross-dataset (W4)
    cross_json = out / "cross_eval" / "cross_eval_results.json"
    if not cross_json.exists() and args.celebdf_root:
        cmd = [
            sys.executable, str(base / "w4_full_train" / "eval_cross_probe.py"),
            "--ff_root", args.ff_root,
            "--resume_ckpt", args.resume_ckpt,
            "--out_dir", str(out / "cross_eval"),
            "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
        ]
        if args.celebdf_root:
            cmd.extend(["--celebdf_root", args.celebdf_root])
        if args.dfdc_faces_root:
            cmd.extend(["--dfdc_faces_root", args.dfdc_faces_root])
        steps.append(("W4 Cross-dataset", cmd))
    else:
        print(f"[SKIP] Cross-eval results exist or no cross-dataset specified")

    # Step 3: Robustness (W6)
    robust_json = out / "robustness" / "robustness_results.json"
    if not robust_json.exists():
        steps.append(("W6 Robustness", [
            sys.executable, str(base / "w6_robustness" / "compress_test.py"),
            "--ff_root", args.ff_root,
            "--resume_ckpt", args.resume_ckpt,
            "--out_dir", str(out / "robustness"),
            "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
        ]))
    else:
        print(f"[SKIP] Robustness results exist: {robust_json}")

    # Step 4: Calibration (W7)
    cal_json = out / "calibration" / "calibration_results.json"
    if not cal_json.exists():
        steps.append(("W7 Calibration", [
            sys.executable, str(base / "w7_integration" / "calibrate.py"),
            "--ff_root", args.ff_root,
            "--resume_ckpt", args.resume_ckpt,
            "--out_dir", str(out / "calibration"),
            "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
        ]))
    else:
        print(f"[SKIP] Calibration results exist: {cal_json}")

    # Run all steps
    results = {}
    for name, cmd in steps:
        ok = run_step(name, cmd)
        results[name] = ok

    # Step 5: Figures (always re-run, fast)
    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Ablation tables & charts
    if ablation_json.exists():
        run_step("Tables", [sys.executable, str(base / "w5_ablation" / "results_table.py"),
                            "--json_path", str(ablation_json), "--out_dir", str(figures_dir)])
        run_step("Ablation charts", [sys.executable, str(base / "w9_viz" / "plot_ablation_bars.py"),
                                     "--json_path", str(ablation_json), "--out_dir", str(figures_dir)])

    # Robustness figure
    if robust_json.exists():
        run_step("Robustness figure", [sys.executable, str(base / "w9_viz" / "plot_robustness.py"),
                                       "--json_path", str(robust_json), "--out_dir", str(figures_dir)])

    # Blink figure
    if args.blink_cache:
        run_step("Blink figure", [sys.executable, str(base / "w9_viz" / "plot_blink_timeline.py"),
                                  "--ff_root", args.ff_root, "--blink_cache", args.blink_cache,
                                  "--out_dir", str(figures_dir)])

    # Grad-CAM
    run_step("Grad-CAM", [sys.executable, str(base / "w9_viz" / "plot_gradcam.py"),
                          "--ff_root", args.ff_root, "--resume_ckpt", args.resume_ckpt,
                          "--out_dir", str(figures_dir)])

    # Step 6: Comprehensive report (W8)
    report_cmd = [sys.executable, str(base / "w8_eval" / "eval_all.py"),
                  "--out_dir", str(out / "report")]
    if ablation_json.exists():
        report_cmd.extend(["--ablation_json", str(ablation_json)])
    if cross_json.exists():
        report_cmd.extend(["--cross_json", str(cross_json)])
    if robust_json.exists():
        report_cmd.extend(["--robustness_json", str(robust_json)])
    if cal_json.exists():
        report_cmd.extend(["--calibration_json", str(cal_json)])
    run_step("Comprehensive report", report_cmd)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL EVALUATION COMPLETE")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL':>4s}  {name}")
    print(f"\nOutputs:")
    print(f"  Results:  {out}")
    print(f"  Figures:  {figures_dir}")
    print(f"  Report:   {out / 'report' / 'comprehensive_report.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W10: One-command final evaluation")
    p.add_argument("--ff_root", required=True)
    p.add_argument("--resume_ckpt", required=True)
    p.add_argument("--rppg_cache", default=None)
    p.add_argument("--blink_cache", default=None)
    p.add_argument("--celebdf_root", default=None)
    p.add_argument("--dfdc_faces_root", default=None)
    p.add_argument("--out_dir", default="./final")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
