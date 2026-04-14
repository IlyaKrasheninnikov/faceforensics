"""
W8: Comprehensive benchmark runner — consolidates ALL results into one report.

Loads JSON results from W3/W4/W5/W6/W7 and produces a unified summary table
plus failure analysis on the test set.

Usage:
    python w8_eval/eval_all.py \
        --ablation_json /kaggle/working/ablation/ablation_results.json \
        --cross_json /kaggle/working/cross_eval/cross_eval_results.json \
        --robustness_json /kaggle/working/robustness/robustness_results.json \
        --calibration_json /kaggle/working/calibration/calibration_results.json \
        --out_dir /kaggle/working/final_report
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation = load_json(args.ablation_json)
    cross = load_json(args.cross_json)
    robustness = load_json(args.robustness_json)
    calibration = load_json(args.calibration_json)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("P3 BIO-SIGNAL FORENSICS — COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("=" * 80)

    # ─── W5 Ablation ─────────────────────────────────────────────────────────
    if ablation:
        report_lines.append("\n## 1. ABLATION STUDY (W5) — FF++ c23")
        report_lines.append(f"{'Variant':<30s} {'Dim':>5s} {'Val AUC':>8s} {'Test AUC':>9s} "
                           f"{'Test EER':>9s} {'Test ECE':>9s}")
        report_lines.append("-" * 75)
        for vname in sorted(ablation.keys()):
            r = ablation[vname]
            report_lines.append(
                f"{vname:<30s} {r['feat_dim']:>5d} {r.get('val_auc',0):>8.4f} "
                f"{r['test_auc']:>9.4f} {r['test_eer']:>9.4f} {r['test_ece']:>9.4f}"
            )

        # Best variant
        best = max(ablation.items(), key=lambda x: x[1]["test_auc"])
        report_lines.append(f"\nBest variant: {best[0]} (Test AUC={best[1]['test_auc']:.4f})")

        # Key finding
        bb_auc = ablation.get("1_backbone_only", {}).get("test_auc", 0)
        full_auc = ablation.get("4_backbone+rppg+blink", {}).get("test_auc", 0)
        full_eer = ablation.get("4_backbone+rppg+blink", {}).get("test_eer", 0)
        bb_eer = ablation.get("1_backbone_only", {}).get("test_eer", 0)
        report_lines.append(f"Fusion AUC lift: {(full_auc - bb_auc)*100:+.2f}%")
        report_lines.append(f"Fusion EER improvement: {bb_eer:.4f} → {full_eer:.4f} ({(bb_eer-full_eer)*100:.1f}pp)")

    # ─── W4 Cross-dataset ────────────────────────────────────────────────────
    if cross:
        report_lines.append(f"\n## 2. CROSS-DATASET GENERALIZATION (W4)")
        report_lines.append(f"{'Dataset':<20s} {'N':>6s} {'AUC':>8s} {'EER':>8s}")
        report_lines.append("-" * 45)
        for ds_name, m in cross.items():
            if isinstance(m, dict) and "n" in m:
                report_lines.append(f"{ds_name:<20s} {m['n']:>6d} {m['auc']:>8.4f} {m['eer']:>8.4f}")
            elif isinstance(m, dict) and "auc" in m:
                report_lines.append(f"{ds_name:<20s} {'—':>6s} {m['auc']:>8.4f}")

    # ─── W6 Robustness ──────────────────────────────────────────────────────
    if robustness:
        report_lines.append(f"\n## 3. ROBUSTNESS (W6) — FF++ c23 Test")
        clean_auc = robustness.get("clean", {}).get("auc", 0)
        report_lines.append(f"{'Perturbation':<20s} {'AUC':>8s} {'Drop':>8s} {'EER':>8s}")
        report_lines.append("-" * 48)
        for pname, r in robustness.items():
            drop = r["auc"] - clean_auc
            report_lines.append(f"{pname:<20s} {r['auc']:>8.4f} {drop:>+8.4f} {r['eer']:>8.4f}")

        # How many perturbations stay within 5%?
        within_5 = sum(1 for r in robustness.values()
                       if r["auc"] >= clean_auc - 0.05 and r.get("auc") != clean_auc)
        total = len(robustness) - 1  # exclude clean
        report_lines.append(f"\nWithin 5% drop: {within_5}/{total} perturbations")

    # ─── W7 Calibration ─────────────────────────────────────────────────────
    if calibration:
        report_lines.append(f"\n## 4. CALIBRATION (W7)")
        report_lines.append(f"{'Metric':<12s} {'Test raw':>10s} {'Test cal':>10s}")
        report_lines.append("-" * 35)
        raw = calibration.get("test_raw", {})
        cal = calibration.get("test_calibrated", {})
        for m in ["auc", "eer", "ece", "ap"]:
            report_lines.append(f"{m:<12s} {raw.get(m,0):>10.4f} {cal.get(m,0):>10.4f}")
        report_lines.append(f"\nPlatt coef: {calibration.get('platt_coef', '?'):.4f}, "
                           f"intercept: {calibration.get('platt_intercept', '?'):.4f}")

    # ─── Go/No-Go Assessment ─────────────────────────────────────────────────
    report_lines.append(f"\n## 5. GO / NO-GO ASSESSMENT")
    report_lines.append("-" * 40)

    targets = {
        "Cross-dataset AUC ≥ 0.90": False,
        "EER ≤ 10%": False,
        "ECE ≤ 0.08": False,
    }
    if ablation:
        best_eer = min(r["test_eer"] for r in ablation.values())
        best_ece = min(r["test_ece"] for r in ablation.values())
        targets["EER ≤ 10%"] = best_eer <= 0.10
        targets["ECE ≤ 0.08"] = best_ece <= 0.08

    if cross:
        cross_aucs = [m["auc"] for m in cross.values() if isinstance(m, dict) and "auc" in m and "n" in m]
        targets["Cross-dataset AUC ≥ 0.90"] = any(a >= 0.90 for a in cross_aucs) if cross_aucs else False

    for target, met in targets.items():
        status = "PASS" if met else "NOT MET"
        report_lines.append(f"  [{status}] {target}")

    # No-Go criteria check
    if cross:
        cross_aucs = [m["auc"] for m in cross.values() if isinstance(m, dict) and "n" in m]
        all_below_82 = all(a < 0.82 for a in cross_aucs) if cross_aucs else True
        if all_below_82:
            report_lines.append(f"\n  ⚠ NO-GO TRIGGERED: All cross-dataset AUC < 0.82")
            report_lines.append(f"  Recommendation: Emphasize physiological analysis contribution")
            report_lines.append(f"  to explainability rather than standalone AUC; reduce ensemble weight")

    report_lines.append(f"\n{'='*80}")
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Print and save
    report_text = "\n".join(report_lines)
    print(report_text)

    with open(out_dir / "comprehensive_report.txt", "w") as f:
        f.write(report_text)
    print(f"\nReport saved: {out_dir / 'comprehensive_report.txt'}")

    # Also save structured JSON
    summary = {
        "ablation_best": best[0] if ablation else None,
        "ablation_best_auc": best[1]["test_auc"] if ablation else None,
        "cross_dataset": {k: v for k, v in (cross or {}).items() if isinstance(v, dict) and "auc" in v},
        "robustness_within_5pct": f"{within_5}/{total}" if robustness else None,
        "go_no_go": targets,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W8: Comprehensive benchmark report")
    p.add_argument("--ablation_json", default=None)
    p.add_argument("--cross_json", default=None)
    p.add_argument("--robustness_json", default=None)
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--out_dir", default="./final_report")
    main(p.parse_args())
