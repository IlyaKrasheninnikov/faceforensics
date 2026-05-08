"""
E16 — Physiology-quality stratification + error-conditional analysis.

Addresses professor v5 review:
  3. "Run a physiology-quality stratification analysis using existing rPPG SNR,
      blink hit-rate, face tracking, or nonzero-feature information. Under what
      signal-quality and backbone conditions does physiology add residual value?"
  6. "Add error-conditional analysis: among samples where CLIP is wrong, check
      whether rPPG or blink corrects the error or makes it worse."

What this does (one Kaggle run, ~3 min):
  1. Reads the same E14 caches (CLIP, ff/celebdf/dfdc).
  2. For each clip in the CelebDF strict-LODO test partition (n=1758):
       - rppg_available = (rppg vector has any nonzero entry)
       - blink_available = (blink vector has any nonzero entry)
       - rppg_snr = magnitude_max / magnitude_floor of the 12-d rPPG FFT feature
                    (proxy for heart-rate signal strength)
       - blink_intensity = mean(|blink|) over the 16-d blink feature
  3. Re-uses the existing strict-LODO probe scores from
     `e14_lodo_strict_clip/scores/test_celebdf_s*_*.npz` to compute:
       - Per-quality-stratum AUC for backbone_only and full_fusion
       - Δ AUC (full - bb) per stratum
  4. Error-conditional analysis: among samples where backbone_only is wrong,
     check whether full_fusion / +rPPG / +blink flip them to correct, or
     make new errors. Report: "rescues" (bb wrong → variant right) and
     "regressions" (bb right → variant wrong).
  5. Outputs a single JSON + markdown table.

Output:
  /kaggle/working/e16_physio_quality_clip/
    quality_metrics.csv       — per-clip rppg/blink quality
    stratified_auc.csv        — AUC by quality stratum × variant
    error_conditional.csv     — rescue / regression counts
    summary.json              — single-page summary

Usage:
    python w10_stats/e16_physio_quality_stratification.py \\
        --cache_dir   "$CACHE_CLIP" \\
        --celebdf_root "$CDF" \\
        --strict_scores_dir "/kaggle/working/e14_lodo_strict_clip/scores" \\
        --out_dir     "$OUT_E16"
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import roc_auc, eer, tpr_at_fpr
from identity_split_sanity import scan_celebdf_with_subject, subject_aware_split


def rppg_snr(rppg_vec):
    """Proxy SNR for the 12-d rPPG FFT feature.

    The v13 extractor stores a 12-d log-magnitude FFT summary in the
    heart-rate band. SNR proxy: max(magnitude) / mean(magnitude). For an
    informative vector this is > 1.5; for noise it is ≈ 1.0; for an
    all-zero vector it is undefined (return 0).
    """
    v = np.asarray(rppg_vec, dtype=np.float32)
    if not np.any(v != 0):
        return 0.0
    m = np.abs(v)
    floor = m.mean() + 1e-9
    return float(m.max() / floor)


def blink_intensity(blink_vec):
    """Mean absolute value of the 16-d blink feature."""
    v = np.asarray(blink_vec, dtype=np.float32)
    return float(np.mean(np.abs(v)))


def quantile_strata(values, q=(0.0, 0.5, 1.0), labels=("low", "high")):
    """Bin `values` by quantiles. Default: median-split into low/high."""
    qv = np.quantile(values, list(q))
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        for k, label in enumerate(labels):
            if v <= qv[k + 1]:
                out[i] = label
                break
        else:
            out[i] = labels[-1]
    return out, qv


def main(args):
    print(f"[e16] cache_dir={args.cache_dir}")

    # Load CLIP cache for CelebDF (test partition is the strict-LODO 1758)
    cache = {k: v for k, v in np.load(
        Path(args.cache_dir) / "celebdf.npz", allow_pickle=True).items()}
    print(f"[e16] loaded celebdf cache: n={len(cache['labels'])} "
          f"bb_dim={cache['backbone'].shape[1]} "
          f"rppg_dim={cache['rppg'].shape[1]} "
          f"blink_dim={cache['blink'].shape[1]}")

    # Reconstruct the strict-LODO test partition (subject-aware)
    cdf_rows = scan_celebdf_with_subject(args.celebdf_root)
    subject_ids = np.array([r[0] for r in cdf_rows])
    cdf_labels_check = np.array([r[1] for r in cdf_rows])
    if not np.array_equal(cache["labels"], cdf_labels_check):
        raise RuntimeError("CelebDF cache order disagrees with re-scan order")

    cd_subj_tr, cd_subj_te, _, _ = subject_aware_split(subject_ids, seed=42)
    print(f"[e16] subject-aware test partition: n_test={len(cd_subj_te)}")
    assert len(cd_subj_te) == 1758, f"unexpected test size {len(cd_subj_te)}"

    # Compute per-clip quality metrics on the test partition
    rppg_te = cache["rppg"][cd_subj_te]
    blink_te = cache["blink"][cd_subj_te]
    labels_te = cache["labels"][cd_subj_te]

    n_test = len(cd_subj_te)
    rppg_avail = np.array([np.any(r != 0) for r in rppg_te])
    blink_avail = np.array([np.any(b != 0) for b in blink_te])
    rppg_snr_arr = np.array([rppg_snr(r) for r in rppg_te])
    blink_int_arr = np.array([blink_intensity(b) for b in blink_te])

    print(f"\n[e16] Quality metrics on n={n_test} CelebDF test clips:")
    print(f"  rPPG available (nonzero): {rppg_avail.sum()}/{n_test} = "
          f"{rppg_avail.mean()*100:.1f}%")
    print(f"  blink available (nonzero): {blink_avail.sum()}/{n_test} = "
          f"{blink_avail.mean()*100:.1f}%")
    if rppg_avail.any():
        print(f"  rPPG SNR (nonzero clips): "
              f"min={rppg_snr_arr[rppg_avail].min():.2f} "
              f"med={np.median(rppg_snr_arr[rppg_avail]):.2f} "
              f"max={rppg_snr_arr[rppg_avail].max():.2f}")
    if blink_avail.any():
        print(f"  blink intensity (nonzero clips): "
              f"min={blink_int_arr[blink_avail].min():.4f} "
              f"med={np.median(blink_int_arr[blink_avail]):.4f} "
              f"max={blink_int_arr[blink_avail].max():.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-clip quality metrics
    with open(out_dir / "quality_metrics.csv", "w") as f:
        f.write("clip_idx,subject_id,label,rppg_available,blink_available,"
                "rppg_snr,blink_intensity\n")
        for i, te_idx in enumerate(cd_subj_te):
            f.write(f"{te_idx},{subject_ids[te_idx]},{labels_te[i]},"
                    f"{int(rppg_avail[i])},{int(blink_avail[i])},"
                    f"{rppg_snr_arr[i]:.4f},{blink_int_arr[i]:.4f}\n")
    print(f"[e16] wrote {out_dir/'quality_metrics.csv'}")

    # ─────────────────────────────────────────────────────────
    # Stratified AUC analysis using strict-LODO probe scores
    # ─────────────────────────────────────────────────────────
    scores_dir = Path(args.strict_scores_dir)
    if not scores_dir.exists():
        print(f"[e16] WARNING: scores_dir {scores_dir} not found; "
              f"skipping stratified AUC")
        return

    SEEDS = [0, 1, 42, 1337, 2024]
    VARIANTS = ["backbone_only", "backbone+rppg", "backbone+blink", "full_fusion"]

    # Strata: by physiology availability (binary), and by SNR quartile
    snr_finite = rppg_snr_arr[rppg_avail]
    if len(snr_finite) >= 4:
        snr_q1, snr_q3 = np.quantile(snr_finite, [0.25, 0.75])
    else:
        snr_q1 = snr_q3 = np.nan

    strata_def = {
        "ALL":            np.ones(n_test, dtype=bool),
        "rppg_avail":     rppg_avail,
        "rppg_unavail":   ~rppg_avail,
        "blink_avail":    blink_avail,
        "blink_unavail":  ~blink_avail,
        "both_avail":     rppg_avail & blink_avail,
        "both_unavail":   ~rppg_avail & ~blink_avail,
    }
    if not np.isnan(snr_q1):
        strata_def["rppg_snr_high"] = rppg_avail & (rppg_snr_arr >= snr_q3)
        strata_def["rppg_snr_low"]  = rppg_avail & (rppg_snr_arr <= snr_q1)

    print(f"\n[e16] Stratified AUC (5-seed mean, strict LODO CelebDF):")
    print(f"{'stratum':<22s} {'n':>5s} {'pos':>4s} | "
          f"{'bb_only':>10s} {'+rPPG':>10s} {'+blink':>10s} {'fusion':>10s} | "
          f"{'Δ_fusion':>10s}")
    rows = []
    for stratum, mask in strata_def.items():
        n_s = int(mask.sum())
        if n_s == 0 or labels_te[mask].sum() == 0 or labels_te[mask].sum() == n_s:
            continue  # skip empty or single-class strata
        pos_s = int(labels_te[mask].sum())
        per_var = {}
        for v in VARIANTS:
            aucs = []
            for s in SEEDS:
                p = scores_dir / f"test_celebdf_s{s}_{v}.npz"
                if not p.exists():
                    continue
                d = np.load(str(p))
                # Score files were saved with the test-partition order
                # (see lodo_probe_strict.py line 230). Apply mask directly.
                aucs.append(roc_auc(d["labels"][mask], d["scores"][mask]))
            per_var[v] = (float(np.mean(aucs)), float(np.std(aucs, ddof=1)))

        delta_fusion = per_var["full_fusion"][0] - per_var["backbone_only"][0]
        rows.append({
            "stratum": stratum, "n": n_s, "n_pos": pos_s,
            **{f"{v}_mean": per_var[v][0] for v in VARIANTS},
            **{f"{v}_std":  per_var[v][1] for v in VARIANTS},
            "delta_fusion_minus_bb": delta_fusion,
        })
        print(f"{stratum:<22s} {n_s:>5d} {pos_s:>4d} | "
              f"{per_var['backbone_only'][0]:>10.4f} "
              f"{per_var['backbone+rppg'][0]:>10.4f} "
              f"{per_var['backbone+blink'][0]:>10.4f} "
              f"{per_var['full_fusion'][0]:>10.4f} | "
              f"{delta_fusion:>+10.4f}")

    head = ["stratum", "n", "n_pos"] + \
           [f"{v}_mean" for v in VARIANTS] + \
           [f"{v}_std"  for v in VARIANTS] + \
           ["delta_fusion_minus_bb"]
    with open(out_dir / "stratified_auc.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")
    print(f"[e16] wrote {out_dir/'stratified_auc.csv'}")

    # ─────────────────────────────────────────────────────────
    # Error-conditional analysis (Youden threshold per variant per seed,
    # then count rescue / regression flips on bb_only ↔ each variant)
    # ─────────────────────────────────────────────────────────
    print(f"\n[e16] Error-conditional analysis (Youden-threshold flips):")
    print(f"{'variant':<18s} {'rescue':>8s} {'regr':>8s} "
          f"{'net':>8s} {'rescue%':>10s}")
    err_rows = []
    for v in ["backbone+rppg", "backbone+blink", "full_fusion"]:
        rescues = []  # bb wrong → variant correct
        regressions = []  # bb correct → variant wrong
        for s in SEEDS:
            p_bb = scores_dir / f"test_celebdf_s{s}_backbone_only.npz"
            p_v  = scores_dir / f"test_celebdf_s{s}_{v}.npz"
            d_bb = np.load(str(p_bb))
            d_v  = np.load(str(p_v))
            y = d_bb["labels"]
            # Youden threshold per variant on its own seed-0-style score distribution
            from multiseed_and_stats import youden_threshold
            thr_bb = youden_threshold(y, d_bb["scores"])
            thr_v  = youden_threshold(y, d_v["scores"])
            pred_bb = (d_bb["scores"] >= thr_bb).astype(int)
            pred_v  = (d_v["scores"]  >= thr_v).astype(int)
            correct_bb = (pred_bb == y).astype(int)
            correct_v  = (pred_v  == y).astype(int)
            rescues.append(int(((correct_bb == 0) & (correct_v == 1)).sum()))
            regressions.append(int(((correct_bb == 1) & (correct_v == 0)).sum()))
        rescue_mean = np.mean(rescues)
        regression_mean = np.mean(regressions)
        net = rescue_mean - regression_mean
        rescue_pct = 100 * rescue_mean / (rescue_mean + regression_mean + 1e-9)
        err_rows.append({
            "variant": v,
            "rescue_mean": rescue_mean,
            "regression_mean": regression_mean,
            "net_mean": net,
            "rescue_pct": rescue_pct,
            "rescues_per_seed": rescues,
            "regressions_per_seed": regressions,
        })
        print(f"{v:<18s} {rescue_mean:>8.1f} {regression_mean:>8.1f} "
              f"{net:>+8.1f} {rescue_pct:>9.1f}%")

    with open(out_dir / "error_conditional.csv", "w") as f:
        f.write("variant,rescue_mean,regression_mean,net_mean,rescue_pct\n")
        for r in err_rows:
            f.write(f"{r['variant']},{r['rescue_mean']:.2f},"
                    f"{r['regression_mean']:.2f},{r['net_mean']:.2f},"
                    f"{r['rescue_pct']:.2f}\n")
    print(f"[e16] wrote {out_dir/'error_conditional.csv'}")

    # ─────────────────────────────────────────────────────────
    # Summary JSON
    # ─────────────────────────────────────────────────────────
    summary = {
        "n_test": int(n_test),
        "rppg_available_pct": float(rppg_avail.mean()),
        "blink_available_pct": float(blink_avail.mean()),
        "rppg_snr_median": float(np.median(rppg_snr_arr[rppg_avail]))
                           if rppg_avail.any() else None,
        "stratified_auc_rows": rows,
        "error_conditional": err_rows,
        "seeds": SEEDS,
        "variants": VARIANTS,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else
                                    int(o) if isinstance(o, np.integer) else str(o))
    print(f"[e16] wrote {out_dir/'summary.json'}")
    print(f"\n[e16] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--celebdf_root", required=True)
    ap.add_argument("--strict_scores_dir", required=True,
                    help="Path to the strict_lodo bundle's CLIP scores/ subdir")
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
