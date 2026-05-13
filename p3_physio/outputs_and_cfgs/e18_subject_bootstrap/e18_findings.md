# E18 — Subject-cluster bootstrap for strict-LODO CelebDF

**Date:** 2026-05-10  
**n_clips:** 1758  
**n_subjects:** 72

## 5-seed aggregate (clip-level vs subject-cluster CI)

| Variant | mean AUC | mean clip CI width | mean subject CI width | inflation |
|---|---|---|---|---|
| backbone_only | 0.7493 | 0.0712 | 0.1399 | 1.97x |
| backbone+rppg | 0.7485 | 0.0713 | 0.1394 | 1.96x |
| backbone+blink | 0.7455 | 0.0709 | 0.1384 | 1.96x |
| full_fusion | 0.7453 | 0.0714 | 0.1378 | 1.93x |

## Headline CI for backbone-only (seed 0)

- Clip-level 95% CI: [0.7236, 0.7946]
- Subject-cluster 95% CI: [0.6900, 0.8197]

**The subject-cluster CI is materially wider than the clip-level CI.** This means the clip-level CI we have been quoting underestimates the true sampling uncertainty. The headline CelebDF strict-LODO AUC is still 0.749, but the 95% CI should be quoted as the subject-cluster width.
