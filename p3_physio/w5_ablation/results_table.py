"""
W5: Generate publication-ready ablation results table from ablation_results.json.

Produces:
  - LaTeX table (for paper)
  - Markdown table (for README / thesis doc)
  - Console-formatted summary

Usage:
    python w5_ablation/results_table.py --json_path /kaggle/working/ablation/ablation_results.json

    Or with local copy:
    python w5_ablation/results_table.py --json_path ./ablation_results.json --out_dir ./figures
"""

import argparse
import json
from pathlib import Path


VARIANT_LABELS = {
    "1_backbone_only": "Backbone only",
    "2_backbone+rppg": "Backbone + rPPG",
    "3_backbone+blink": "Backbone + Blink",
    "4_backbone+rppg+blink": "Full fusion (ours)",
    "5_rppg_only": "rPPG only",
    "6_blink_only": "Blink only",
    "7_fakecatcher_svm": "FakeCatcher SVM",
}

MANIP_ORDER = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
VARIANT_ORDER = [
    "1_backbone_only", "2_backbone+rppg", "3_backbone+blink",
    "4_backbone+rppg+blink", "5_rppg_only", "6_blink_only", "7_fakecatcher_svm",
]


def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def gen_latex_main(results):
    """Table 1: Main ablation results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: contribution of each physiological signal. "
                 r"Linear probe on frozen EfficientNet-B4 backbone, FF++ c23 test set.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Dim & AUC $\uparrow$ & EER $\downarrow$ & AP $\uparrow$ & ECE $\downarrow$ \\")
    lines.append(r"\midrule")

    best_auc = max(r.get("test_auc", 0) for r in results.values())
    best_eer = min(r.get("test_eer", 1) for r in results.values() if r.get("test_eer", 1) > 0)

    for vname in VARIANT_ORDER:
        if vname not in results:
            continue
        r = results[vname]
        label = VARIANT_LABELS.get(vname, vname)
        dim = r["feat_dim"]
        auc = r["test_auc"]
        eer = r["test_eer"]
        ap = r["test_ap"]
        ece = r["test_ece"]

        auc_str = f"\\textbf{{{auc:.4f}}}" if abs(auc - best_auc) < 1e-5 else f"{auc:.4f}"
        eer_str = f"\\textbf{{{eer:.4f}}}" if abs(eer - best_eer) < 1e-5 else f"{eer:.4f}"

        lines.append(f"{label} & {dim} & {auc_str} & {eer_str} & {ap:.4f} & {ece:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_latex_per_manip(results):
    """Table 2: Per-manipulation AUC breakdown."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-manipulation AUC on FF++ c23 test set.}")
    lines.append(r"\label{tab:per_manip}")
    cols = "l" + "c" * len(MANIP_ORDER)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = "Variant & " + " & ".join(MANIP_ORDER) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for vname in VARIANT_ORDER:
        if vname not in results:
            continue
        r = results[vname]
        label = VARIANT_LABELS.get(vname, vname)
        pm = r.get("test_per_manip", {})
        cells = []
        for m in MANIP_ORDER:
            if m in pm and "auc" in pm[m]:
                cells.append(f"{pm[m]['auc']:.3f}")
            else:
                cells.append("---")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_markdown(results):
    """Markdown version of both tables."""
    lines = []
    lines.append("## Table 1: Ablation Results (FF++ c23 Test)")
    lines.append("")
    lines.append("| Variant | Dim | AUC | EER | AP | ECE |")
    lines.append("|---------|-----|-----|-----|-----|-----|")

    for vname in VARIANT_ORDER:
        if vname not in results:
            continue
        r = results[vname]
        label = VARIANT_LABELS.get(vname, vname)
        lines.append(f"| {label} | {r['feat_dim']} | {r['test_auc']:.4f} | "
                     f"{r['test_eer']:.4f} | {r['test_ap']:.4f} | {r['test_ece']:.4f} |")

    lines.append("")
    lines.append("## Table 2: Per-Manipulation AUC (Test)")
    lines.append("")
    header = "| Variant | " + " | ".join(MANIP_ORDER) + " |"
    sep = "|---------|" + "|".join(["------"] * len(MANIP_ORDER)) + "|"
    lines.append(header)
    lines.append(sep)

    for vname in VARIANT_ORDER:
        if vname not in results:
            continue
        r = results[vname]
        label = VARIANT_LABELS.get(vname, vname)
        pm = r.get("test_per_manip", {})
        cells = []
        for m in MANIP_ORDER:
            if m in pm and "auc" in pm[m]:
                cells.append(f"{pm[m]['auc']:.3f}")
            else:
                cells.append("---")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main(args):
    results = load_results(args.json_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LaTeX
    latex_main = gen_latex_main(results)
    latex_manip = gen_latex_per_manip(results)
    latex_full = latex_main + "\n\n" + latex_manip

    latex_path = out_dir / "ablation_tables.tex"
    with open(latex_path, "w") as f:
        f.write(latex_full)
    print(f"LaTeX tables saved: {latex_path}")

    # Markdown
    md = gen_markdown(results)
    md_path = out_dir / "ablation_tables.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown tables saved: {md_path}")

    # Console
    print("\n" + md)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W5: Generate publication-ready ablation tables")
    p.add_argument("--json_path", required=True, help="Path to ablation_results.json")
    p.add_argument("--out_dir", default="./figures", help="Output directory")
    main(p.parse_args())
