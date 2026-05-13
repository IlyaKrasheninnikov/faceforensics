# E19 — Fusion-fairness sanity check

**Date:** 2026-05-10  
**Setting:** strict LODO CelebDF (n=1758), CLIP backbone

Three probe families compared:
- `raw_concat_adamw`: linear AdamW probe on raw concatenated features (the v6/v7 default)
- `zscored_adamw`: same probe, but each block (backbone, rppg, blink) is z-scored on the training set
- `l2_logistic`: scikit-learn LogisticRegression(C=1.0) on z-scored features (no probe-init noise)

## 5-seed aggregate AUC by (variant × family)

| variant | family | AUC mean ± std | TPR@5% |
|---|---|---|---|
| backbone_only | raw_concat_adamw | 0.7493 ± 0.0096 | 0.2377 |
| backbone_only | zscored_adamw | 0.7349 ± 0.0146 | 0.1537 |
| backbone_only | l2_logistic | 0.6230 ± 0.0000 | 0.1414 |
| backbone+rppg | raw_concat_adamw | 0.7485 ± 0.0097 | 0.2379 |
| backbone+rppg | zscored_adamw | 0.7325 ± 0.0142 | 0.1511 |
| backbone+rppg | l2_logistic | 0.6232 ± 0.0000 | 0.1284 |
| backbone+blink | raw_concat_adamw | 0.7455 ± 0.0089 | 0.2346 |
| backbone+blink | zscored_adamw | 0.7217 ± 0.0149 | 0.1492 |
| backbone+blink | l2_logistic | 0.6299 ± 0.0000 | 0.1433 |
| full_fusion | raw_concat_adamw | 0.7453 ± 0.0092 | 0.2318 |
| full_fusion | zscored_adamw | 0.7199 ± 0.0145 | 0.1484 |
| full_fusion | l2_logistic | 0.6273 ± 0.0000 | 0.1485 |

## Decisive question — does the physiology-variant ordering hold under fair probes?

### raw_concat_adamw

- backbone_only: 0.7493 ± 0.0096
- backbone+rppg: 0.7485 ± 0.0097
- backbone+blink: 0.7455 ± 0.0089
- full_fusion: 0.7453 ± 0.0092

  Δ (full_fusion − backbone_only) = -0.0041

### zscored_adamw

- backbone_only: 0.7349 ± 0.0146
- backbone+rppg: 0.7325 ± 0.0142
- backbone+blink: 0.7217 ± 0.0149
- full_fusion: 0.7199 ± 0.0145

  Δ (full_fusion − backbone_only) = -0.0149

### l2_logistic

- backbone_only: 0.6230 ± 0.0000
- backbone+rppg: 0.6232 ± 0.0000
- backbone+blink: 0.6299 ± 0.0000
- full_fusion: 0.6273 ± 0.0000

  Δ (full_fusion − backbone_only) = +0.0043


If Δ stays in the [-0.005, +0.005] band for all three families, the v7 conclusion is robust to probe choice. If the z-scored or L2 family flips Δ to positive, the v7 framing must be restricted to the raw-concat AdamW probe.
