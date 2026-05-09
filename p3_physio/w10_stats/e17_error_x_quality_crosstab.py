"""
E17 — Error-conditional × physiology-quality cross-tab.

Addresses professor v6 review point 4 (verbatim):
   "Strengthen the physiology-quality and error-conditional analysis:
    when does rPPG/blink rescue CLIP errors, and when does it cause
    regressions?"

E16 (v6) already established:
- 100% physiology coverage on the 1758 strict-LODO CelebDF test clips.
- Top-SNR and bottom-SNR quartiles both show negative Δ AUC.
- +rPPG net rescues = +6.6 (55%); +blink net = -13.2; full_fusion -27.6.

What E16 did NOT do: cross-tabulate the error-conditional flips by
quality stratum. The decisive question is whether +rPPG's positive
rescue rate concentrates on high-quality clips (consistent with
"physiology helps when signal is clean"), or whether it is uniform
across quality strata (consistent with "physiology is independent
information at the threshold level, regardless of SNR").

E17 reads the same artefacts that E16 produced, plus the per-seed
score arrays from E14, and produces the cross-tab locally — no
Kaggle needed.

Output files (in --out_dir):
  e17_crosstab.csv          - per (variant × stratum) rescue/regression counts
  e17_summary.json          - same plus seed-level breakdown
  e17_findings.md           - markdown summary for the v7 manuscript

Usage:
    python p3_physio/w10_stats/e17_error_x_quality_crosstab.py \\
        --quality_csv  p3_physio/outputs_and_cfgs/e16_physio_quality_clip/quality_metrics.csv \\
        --scores_dir   p3_physio/outputs_and_cfgs/strict_lodo_bundle/e14_lodo_strict_clip/scores \\
        --out_dir      p3_physio/outputs_and_cfgs/e17_error_x_quality_clip
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import youden_threshold


SEEDS = [0, 1, 42, 1337, 2024]
VARIANTS = ["backbone+rppg", "backbone+blink", "full_fusion"]


def load_quality(quality_csv):
    rows = []
    with open(quality_csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "clip_idx": int(r["clip_idx"]),
                "subject_id": r["subject_id"],
                "label": int(float(r["label"])),
                "rppg_available": int(r["rppg_available"]),
                "blink_available": int(r["blink_available"]),
                "rppg_snr": float(r["rppg_snr"]),
                "blink_intensity": float(r["blink_intensity"]),
            })
    return rows


def main(args):
    quality = load_quality(args.quality_csv)
    n = len(quality)
    print(f"[e17] loaded {n} quality rows")
    snr = np.array([r["rppg_snr"] for r in quality])
    blink_int = np.array([r["blink_intensity"] for r in quality])
    label = np.array([r["label"] for r in quality])

    # Quality strata (mirror E16 §3.6)
    snr_q1, snr_q3 = np.quantile(snr, [0.25, 0.75])
    blink_q1, blink_q3 = np.quantile(blink_int, [0.25, 0.75])

    strata = {
        "ALL":             np.ones(n, dtype=bool),
        "rppg_snr_high":   snr >= snr_q3,
        "rppg_snr_low":    snr <= snr_q1,
        "blink_int_high":  blink_int >= blink_q3,
        "blink_int_low":   blink_int <= blink_q1,
        "real":            label == 0,
        "fake":            label == 1,
    }
    print(f"[e17] strata sizes:")
    for s, m in strata.items():
        pos = int(label[m].sum())
        print(f"  {s:<22s} n={int(m.sum()):>5d}  pos={pos:>4d}")

    # Read score arrays for each (seed, variant) and compute per-clip flips
    scores_dir = Path(args.scores_dir)
    bb_scores_per_seed = {}
    var_scores_per_seed = {v: {} for v in VARIANTS}

    print(f"\n[e17] loading score arrays from {scores_dir}")
    for s in SEEDS:
        d_bb = np.load(str(scores_dir / f"test_celebdf_s{s}_backbone_only.npz"))
        bb_scores_per_seed[s] = d_bb["scores"]
        labels_check = d_bb["labels"]
        if not np.array_equal(labels_check.astype(int), label):
            raise RuntimeError(
                f"label mismatch between quality_csv and seed-{s} score file"
            )
        for v in VARIANTS:
            d_v = np.load(str(scores_dir / f"test_celebdf_s{s}_{v}.npz"))
            var_scores_per_seed[v][s] = d_v["scores"]
    print(f"[e17] loaded {len(bb_scores_per_seed)} seeds x ({len(VARIANTS)+1}) variants")

    # For each (seed, variant): Youden threshold per probe; per-clip flip flags;
    # then aggregate by stratum.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crosstab_rows = []
    detail_per_seed = {v: {} for v in VARIANTS}

    print(f"\n[e17] cross-tab (5-seed mean rescue / regression / net per stratum):")
    header = (f"{'variant':<18s} {'stratum':<18s} {'n':>5s} | "
              f"{'rescue':>7s} {'regress':>8s} {'net':>7s} {'rescue%':>8s}")
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        for stratum_name, stratum_mask in strata.items():
            n_in_stratum = int(stratum_mask.sum())
            if n_in_stratum == 0:
                continue
            rescues_per_seed = []
            regressions_per_seed = []
            for s in SEEDS:
                y = label
                bb_scores = bb_scores_per_seed[s]
                v_scores = var_scores_per_seed[variant][s]
                thr_bb = youden_threshold(y, bb_scores)
                thr_v  = youden_threshold(y, v_scores)
                pred_bb = (bb_scores >= thr_bb).astype(int)
                pred_v  = (v_scores  >= thr_v).astype(int)
                correct_bb = (pred_bb == y).astype(int)
                correct_v  = (pred_v  == y).astype(int)
                # Per-clip flip flags
                rescue_clip = (correct_bb == 0) & (correct_v == 1)
                regr_clip   = (correct_bb == 1) & (correct_v == 0)
                # Restrict to stratum
                rescues_per_seed.append(int((rescue_clip & stratum_mask).sum()))
                regressions_per_seed.append(int((regr_clip & stratum_mask).sum()))
            rescue_mean = float(np.mean(rescues_per_seed))
            regression_mean = float(np.mean(regressions_per_seed))
            net = rescue_mean - regression_mean
            rescue_pct = 100.0 * rescue_mean / max(rescue_mean + regression_mean, 1e-9)
            crosstab_rows.append({
                "variant": variant, "stratum": stratum_name,
                "n_in_stratum": n_in_stratum,
                "rescue_mean": rescue_mean,
                "regression_mean": regression_mean,
                "net_mean": net,
                "rescue_pct": rescue_pct,
            })
            detail_per_seed[variant][stratum_name] = {
                "rescues": rescues_per_seed,
                "regressions": regressions_per_seed,
            }
            print(f"{variant:<18s} {stratum_name:<18s} {n_in_stratum:>5d} | "
                  f"{rescue_mean:>7.1f} {regression_mean:>8.1f} "
                  f"{net:>+7.1f} {rescue_pct:>7.1f}%")
        print()

    # Write CSV
    head = ["variant", "stratum", "n_in_stratum",
            "rescue_mean", "regression_mean", "net_mean", "rescue_pct"]
    with open(out_dir / "e17_crosstab.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in crosstab_rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")
    print(f"[e17] wrote {out_dir/'e17_crosstab.csv'}")

    # Write JSON
    with open(out_dir / "e17_summary.json", "w") as f:
        json.dump({
            "n_test": int(n),
            "snr_quartiles": {"q1": float(snr_q1), "q3": float(snr_q3)},
            "blink_quartiles": {"q1": float(blink_q1), "q3": float(blink_q3)},
            "stratum_sizes": {s: int(m.sum()) for s, m in strata.items()},
            "crosstab": crosstab_rows,
            "detail_per_seed": detail_per_seed,
            "seeds": SEEDS,
            "variants": VARIANTS,
        }, f, indent=2)
    print(f"[e17] wrote {out_dir/'e17_summary.json'}")

    # Write markdown summary
    with open(out_dir / "e17_findings.md", "w", encoding="utf-8") as f:
        f.write("# E17 — Error-conditional × physiology-quality cross-tab\n\n")
        f.write("**Date:** 2026-05-08  \n")
        f.write("**Source data:** "
                "`e16_physio_quality_clip/quality_metrics.csv` + "
                "`strict_lodo_bundle/e14_lodo_strict_clip/scores/`\n\n")
        f.write("Per-clip rescue / regression counts at the Youden threshold, "
                "stratified by physiology quality. 5-seed mean.\n\n")

        for variant in VARIANTS:
            f.write(f"## {variant}\n\n")
            f.write("| Stratum | n | rescue | regr | net | rescue % |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in crosstab_rows:
                if r["variant"] != variant:
                    continue
                f.write(f"| {r['stratum']} | {r['n_in_stratum']} | "
                        f"{r['rescue_mean']:.1f} | {r['regression_mean']:.1f} | "
                        f"{r['net_mean']:+.1f} | {r['rescue_pct']:.1f}% |\n")
            f.write("\n")

        # Decisive question section
        f.write("## Decisive question — does rPPG help more on clean clips?\n\n")
        rppg = {r["stratum"]: r for r in crosstab_rows
                if r["variant"] == "backbone+rppg"}
        if "rppg_snr_high" in rppg and "rppg_snr_low" in rppg:
            net_high = rppg["rppg_snr_high"]["net_mean"]
            net_low  = rppg["rppg_snr_low"]["net_mean"]
            pct_high = rppg["rppg_snr_high"]["rescue_pct"]
            pct_low  = rppg["rppg_snr_low"]["rescue_pct"]
            f.write(f"- +rPPG on **high-SNR quartile**: net = {net_high:+.1f}, "
                    f"rescue % = {pct_high:.1f}%\n")
            f.write(f"- +rPPG on **low-SNR quartile**:  net = {net_low:+.1f}, "
                    f"rescue % = {pct_low:.1f}%\n")
            f.write(f"- Net difference (high − low): {net_high - net_low:+.1f}\n\n")
            if net_high > net_low + 2:
                f.write("**Direction confirms the 'rPPG helps clean clips' "
                        "hypothesis.** rPPG's positive rescue effect is "
                        "larger on the high-SNR quartile.\n")
            elif net_low > net_high + 2:
                f.write("**Direction REJECTS the 'rPPG helps clean clips' "
                        "hypothesis** — rPPG actually rescues more on the "
                        "low-SNR quartile, opposite to the prediction.\n")
            else:
                f.write("**Direction is approximately uniform across SNR strata** "
                        "(|net high − net low| < 2). rPPG's threshold-level "
                        "value is independent of SNR proxy quality at this scale.\n")

        f.write("\n## Real vs fake breakdown\n\n")
        f.write("Did rPPG/blink mostly rescue *real* clips (false positives "
                "corrected) or *fake* clips (false negatives corrected)?\n\n")
        for variant in VARIANTS:
            real = next((r for r in crosstab_rows
                         if r["variant"] == variant and r["stratum"] == "real"),
                        None)
            fake = next((r for r in crosstab_rows
                         if r["variant"] == variant and r["stratum"] == "fake"),
                        None)
            if real and fake:
                f.write(f"- **{variant}**: rescues real {real['rescue_mean']:.1f} "
                        f"(regr {real['regression_mean']:.1f}, net "
                        f"{real['net_mean']:+.1f}); rescues fake "
                        f"{fake['rescue_mean']:.1f} (regr "
                        f"{fake['regression_mean']:.1f}, net "
                        f"{fake['net_mean']:+.1f})\n")
    print(f"[e17] wrote {out_dir/'e17_findings.md'}")
    print(f"\n[e17] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality_csv", required=True)
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
