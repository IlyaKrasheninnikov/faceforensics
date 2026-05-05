"""
E12 — CLIP robustness at low FPR + degraded-input calibration.

Addresses the professor's correction #4:

    "The new CLIP result now becomes the center of the results, so robustness,
    calibration, and operating-point analysis must be rerun for CLIP. The old
    robustness table only tells us that the EfficientNet-B4 probe fails under
    blur and downscaling. We now need to know whether the CLIP probe also
    fails, especially at low false-positive rates."

Two outputs per (variant, perturbation):
  1) TPR@FPR=1% / 5% / 10% with bootstrap 95% CI (seed 0)
  2) ECE before and after Platt scaling (Platt fitted on the CLEAN FF++ val
     set, applied to the degraded test scores) — answers "do operating
     thresholds chosen on clean data still calibrate under degradation?"

Reuses E9's robust feature caches and E6's clean training caches; adds no
new extraction.

Usage:
    python w10_stats/robust_operating_points.py \\
        --robust_cache_dir "$E9_ROBUST_CACHE" \\
        --c23_cache        "$CACHE_CLIP" \\
        --out_dir          /kaggle/working/e12_clip_op_points \\
        --seeds 0 1 42 1337 2024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from multiseed_and_stats import (
    roc_auc, eer, tpr_at_fpr, average_precision,
    train_linear_probe, predict, identity_split_ff, random_split,
    bootstrap_ci_auc,
    VARIANTS, make_features,
)
from optional_experiments import _platt_fit, _platt_apply, _expected_calibration_error


def main(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[op] device={device}")

    c23_dir = Path(args.c23_cache)
    caches = {}
    for tag in ["ff", "celebdf", "dfdc"]:
        p = c23_dir / f"{tag}.npz"
        if p.exists():
            caches[tag] = {k: v for k, v in np.load(p, allow_pickle=True).items()}
    print(f"[op] loaded c23 caches; bb_dim={caches['ff']['backbone'].shape[1]}")

    robust_dir = Path(args.robust_cache_dir)
    perturb_caches = {}
    for npz in robust_dir.glob("ff_*.npz"):
        pname = npz.stem.replace("ff_", "")
        perturb_caches[pname] = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
    print(f"[op] loaded {len(perturb_caches)} perturbation caches: {sorted(perturb_caches)}")

    tr_ff_idx, vl_ff_idx, te_ff_idx = identity_split_ff(caches["ff"], seed=42)
    cd_tr, _ = random_split(len(caches["celebdf"]["labels"]), seed=42)
    df_tr, _ = random_split(len(caches["dfdc"]["labels"]), seed=42) if "dfdc" in caches else (None, None)

    test_src_ids = caches["ff"]["src_id"][te_ff_idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    rows = []
    score_storage = {}

    # ── Train probes on CLEAN c23 mixed pool, then evaluate on each perturbation ──
    for seed in args.seeds:
        print(f"\n[op] ====== seed={seed} ======")
        # Build mixed training pool
        pools_bb = [caches["ff"]["backbone"][tr_ff_idx], caches["celebdf"]["backbone"][cd_tr]]
        pools_rppg = [caches["ff"]["rppg"][tr_ff_idx], caches["celebdf"]["rppg"][cd_tr]]
        pools_blink = [caches["ff"]["blink"][tr_ff_idx], caches["celebdf"]["blink"][cd_tr]]
        pools_y = [caches["ff"]["labels"][tr_ff_idx], caches["celebdf"]["labels"][cd_tr]]
        if df_tr is not None:
            pools_bb.append(caches["dfdc"]["backbone"][df_tr])
            pools_rppg.append(caches["dfdc"]["rppg"][df_tr])
            pools_blink.append(caches["dfdc"]["blink"][df_tr])
            pools_y.append(caches["dfdc"]["labels"][df_tr])
        bb_tr = np.concatenate(pools_bb)
        rppg_tr = np.concatenate(pools_rppg)
        blink_tr = np.concatenate(pools_blink)
        y_tr = np.concatenate(pools_y)

        bb_vl = caches["ff"]["backbone"][vl_ff_idx]
        rppg_vl = caches["ff"]["rppg"][vl_ff_idx]
        blink_vl = caches["ff"]["blink"][vl_ff_idx]
        y_vl = caches["ff"]["labels"][vl_ff_idx]

        for variant in VARIANTS:
            X_tr = make_features(bb_tr, rppg_tr, blink_tr, variant)
            X_vl = make_features(bb_vl, rppg_vl, blink_vl, variant)
            probe = train_linear_probe(X_tr, y_tr, X_vl, y_vl, device,
                                        epochs=args.epochs, lr=args.lr,
                                        bs=args.batch, seed=seed)

            # Fit Platt on CLEAN val scores — that's the "deployment" calibration
            val_scores = predict(probe, X_vl, device)
            a, b = _platt_fit(val_scores, y_vl)

            for pname, pcache in sorted(perturb_caches.items()):
                te_mask = np.isin(pcache["src_id"], test_src_ids)
                idx = np.where(te_mask)[0]
                if len(idx) == 0:
                    continue
                bb_te = pcache["backbone"][idx]
                rppg_te = pcache["rppg"][idx]
                blink_te = pcache["blink"][idx]
                y_te = pcache["labels"][idx]
                X_te = make_features(bb_te, rppg_te, blink_te, variant)
                raw_scores = predict(probe, X_te, device)
                cal_scores = _platt_apply(raw_scores, a, b)

                row = {
                    "seed": seed, "variant": variant, "perturbation": pname,
                    "n": int(len(y_te)),
                    "auc": roc_auc(y_te, raw_scores),
                    "eer": eer(y_te, raw_scores),
                    "tpr1":  tpr_at_fpr(y_te, raw_scores, 0.01),
                    "tpr5":  tpr_at_fpr(y_te, raw_scores, 0.05),
                    "tpr10": tpr_at_fpr(y_te, raw_scores, 0.10),
                    "ece_raw":   _expected_calibration_error(y_te, raw_scores),
                    "ece_platt": _expected_calibration_error(y_te, cal_scores),
                    "platt_a": a, "platt_b": b,
                }
                rows.append(row)
                score_storage[(seed, variant, pname)] = (raw_scores, y_te)
                np.savez(out_dir / "scores" / f"s{seed}_{variant}_{pname}.npz",
                         raw=raw_scores, cal=cal_scores, labels=y_te)
                print(f"  s={seed} {variant:18s} {pname:13s} "
                      f"AUC={row['auc']:.4f} TPR@5={row['tpr5']:.3f} "
                      f"ECE: {row['ece_raw']:.3f}→{row['ece_platt']:.3f}")

    # ── Raw rows ──
    head = ["seed","variant","perturbation","n","auc","eer",
            "tpr1","tpr5","tpr10","ece_raw","ece_platt","platt_a","platt_b"]
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(head) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in head) + "\n")

    # ── Aggregate ──
    from collections import defaultdict
    agg_buckets = defaultdict(list)
    for r in rows:
        agg_buckets[(r["variant"], r["perturbation"])].append(r)

    agg_rows = []
    print("\n[op] AGGREGATE (mean ± std across seeds)")
    print(f"{'variant':<18s} {'perturbation':<13s}  AUC             TPR@1%       TPR@5%       TPR@10%      ECE_raw       ECE_platt")
    for (var, pert), rs in sorted(agg_buckets.items()):
        def ms(k):
            xs = [r[k] for r in rs]
            return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs)>1 else 0)
        am, asd = ms("auc")
        t1m, t1s = ms("tpr1"); t5m, t5s = ms("tpr5"); t10m, t10s = ms("tpr10")
        em, es = ms("ece_raw"); pm, ps = ms("ece_platt")
        agg_rows.append({
            "variant": var, "perturbation": pert, "n": rs[0]["n"],
            "auc_mean": am, "auc_std": asd,
            "tpr1_mean": t1m, "tpr1_std": t1s,
            "tpr5_mean": t5m, "tpr5_std": t5s,
            "tpr10_mean": t10m, "tpr10_std": t10s,
            "ece_raw_mean": em, "ece_raw_std": es,
            "ece_platt_mean": pm, "ece_platt_std": ps,
            "delta_ece": pm - em,
        })
        print(f"{var:<18s} {pert:<13s}  "
              f"{am:.4f}±{asd:.4f}  "
              f"{t1m:.3f}±{t1s:.3f}  {t5m:.3f}±{t5s:.3f}  {t10m:.3f}±{t10s:.3f}  "
              f"{em:.3f}±{es:.3f}  {pm:.3f}±{ps:.3f}")

    head_agg = list(agg_rows[0].keys()) if agg_rows else []
    with open(out_dir / "aggregate.csv", "w") as f:
        f.write(",".join(head_agg) + "\n")
        for r in agg_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in head_agg) + "\n")

    # ── Bootstrap CIs (seed 0, every perturbation × variant) ──
    boot_rows = []
    for (seed, variant, pert), (s, y) in score_storage.items():
        if seed != args.seeds[0]:
            continue
        ci = bootstrap_ci_auc(y, s, n_boot=args.n_boot, seed=42)
        boot_rows.append({
            "variant": variant, "perturbation": pert, "n": int(len(y)),
            "auc": roc_auc(y, s),
            "auc_ci_lo": ci["auc_ci"][0], "auc_ci_hi": ci["auc_ci"][1],
            "tpr1_ci_lo": ci["tpr_ci"][0.01][0], "tpr1_ci_hi": ci["tpr_ci"][0.01][1],
            "tpr5_ci_lo": ci["tpr_ci"][0.05][0], "tpr5_ci_hi": ci["tpr_ci"][0.05][1],
            "tpr10_ci_lo": ci["tpr_ci"][0.10][0], "tpr10_ci_hi": ci["tpr_ci"][0.10][1],
        })
    head_boot = list(boot_rows[0].keys()) if boot_rows else []
    with open(out_dir / "bootstrap_cis.csv", "w") as f:
        f.write(",".join(head_boot) + "\n")
        for r in boot_rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                             for k in head_boot) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"agg": agg_rows, "bootstrap": boot_rows, "raw": rows},
                  f, indent=2, default=str)
    print(f"\n[op] wrote {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLIP robustness at low FPR + Platt-on-degraded")
    p.add_argument("--robust_cache_dir", required=True)
    p.add_argument("--c23_cache", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 1337, 2024])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n_boot", type=int, default=1000)
    main(p.parse_args())
