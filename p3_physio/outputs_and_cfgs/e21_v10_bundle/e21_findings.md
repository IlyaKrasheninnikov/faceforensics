# E21 - Hierarchical paired bootstrap + threshold CIs + B4 vs CLIP

**Date:** 2026-05-10  
**N_BOOT:** 2000, alpha = 0.05

## 1. CLIP - hierarchical paired (subject+seed) bootstrap on delta AUC

Single CI per (variant, stratum) computed jointly over subject and seed resampling (NOT averaged across seeds).

| Variant | Stratum | n_subj | mean delta | 95% CI | excludes 0? |
|---|---|---|---|---|---|
| backbone+rppg | ALL | 72 | -0.0011 | [-0.0170, +0.0139] | no |
| backbone+rppg | rppg_snr_high_Q | 21 | -0.0021 | [-0.0285, +0.0163] | no |
| backbone+rppg | rppg_snr_low_Q | 37 | +0.0007 | [-0.0074, +0.0076] | no |
| backbone+rppg | blink_int_high_Q | 28 | +0.0001 | [-0.0119, +0.0105] | no |
| backbone+rppg | blink_int_low_Q | 35 | -0.0022 | [-0.0283, +0.0217] | no |
| backbone+blink | ALL | 72 | -0.0037 | [-0.0093, +0.0020] | no |
| backbone+blink | rppg_snr_high_Q | 21 | -0.0007 | [-0.0134, +0.0157] | no |
| backbone+blink | rppg_snr_low_Q | 37 | -0.0033 | [-0.0142, +0.0078] | no |
| backbone+blink | blink_int_high_Q | 28 | -0.0048 | [-0.0138, +0.0034] | no |
| backbone+blink | blink_int_low_Q | 35 | -0.0047 | [-0.0140, +0.0061] | no |
| full_fusion | ALL | 72 | -0.0040 | [-0.0214, +0.0125] | no |
| full_fusion | rppg_snr_high_Q | 21 | -0.0018 | [-0.0327, +0.0236] | no |
| full_fusion | rppg_snr_low_Q | 37 | -0.0017 | [-0.0164, +0.0142] | no |
| full_fusion | blink_int_high_Q | 28 | -0.0040 | [-0.0174, +0.0092] | no |
| full_fusion | blink_int_low_Q | 35 | -0.0049 | [-0.0339, +0.0221] | no |

## 2. B4 - hierarchical paired bootstrap on delta AUC

Same procedure for EfficientNet-B4 v13 strict-LODO scores. If the B4 deltas show CIs that exclude zero on the positive side, the representation-dependent claim is statistically supported.

| Variant | Stratum | n_subj | mean delta | 95% CI | excludes 0? |
|---|---|---|---|---|---|
| backbone+rppg | ALL | 72 | +0.0162 | [-0.0151, +0.0506] | no |
| backbone+rppg | rppg_snr_high_Q | 21 | -0.0023 | [-0.0367, +0.0282] | no |
| backbone+rppg | rppg_snr_low_Q | 37 | -0.0266 | [-0.0491, -0.0072] | **YES** |
| backbone+rppg | blink_int_high_Q | 28 | -0.0047 | [-0.0453, +0.0441] | no |
| backbone+rppg | blink_int_low_Q | 35 | +0.0200 | [-0.0246, +0.0604] | no |
| backbone+blink | ALL | 72 | +0.0008 | [-0.0041, +0.0073] | no |
| backbone+blink | rppg_snr_high_Q | 21 | -0.0015 | [-0.0079, +0.0055] | no |
| backbone+blink | rppg_snr_low_Q | 37 | +0.0023 | [-0.0067, +0.0122] | no |
| backbone+blink | blink_int_high_Q | 28 | +0.0001 | [-0.0056, +0.0062] | no |
| backbone+blink | blink_int_low_Q | 35 | -0.0022 | [-0.0057, +0.0013] | no |
| full_fusion | ALL | 72 | +0.0155 | [-0.0145, +0.0476] | no |
| full_fusion | rppg_snr_high_Q | 21 | -0.0028 | [-0.0357, +0.0296] | no |
| full_fusion | rppg_snr_low_Q | 37 | -0.0257 | [-0.0477, -0.0053] | **YES** |
| full_fusion | blink_int_high_Q | 28 | -0.0057 | [-0.0489, +0.0432] | no |
| full_fusion | blink_int_low_Q | 35 | +0.0206 | [-0.0229, +0.0594] | no |

## 3. CLIP - threshold-level rescue/regression with subject-cluster CIs

**Diagnostic interpretation only.** The FPR thresholds are set directly on the test partition for each probe per seed; this is an oracle calibration, not a deployment-validatable threshold. Recalibration on a source-domain validation set would shift the thresholds.

| Variant | Stratum | FPR | rescue [CI] | regression [CI] | net [CI] |
|---|---|---|---|---|---|
| backbone+rppg | ALL | 1% | 11.8 [0.0, 41.0] | 10.4 [0.0, 36.0] | +1.4 [-32.0, +40.0] |
| backbone+rppg | ALL | 5% | 25.6 [2.0, 80.0] | 18.6 [0.0, 67.0] | +7.0 [-51.0, +78.0] |
| backbone+rppg | ALL | 10% | 22.3 [4.0, 67.0] | 30.9 [4.0, 90.0] | -8.6 [-76.0, +51.0] |
| backbone+rppg | rppg_snr_high_Q | 1% | 3.1 [0.0, 14.0] | 3.4 [0.0, 13.0] | -0.3 [-13.0, +13.0] |
| backbone+rppg | rppg_snr_high_Q | 5% | 9.9 [0.0, 38.0] | 8.9 [0.0, 40.0] | +1.0 [-40.0, +38.0] |
| backbone+rppg | rppg_snr_high_Q | 10% | 7.5 [0.0, 31.0] | 13.9 [0.0, 55.0] | -6.4 [-55.0, +26.0] |
| backbone+rppg | blink_int_high_Q | 1% | 4.2 [0.0, 12.0] | 4.3 [0.0, 15.0] | -0.1 [-14.0, +11.0] |
| backbone+rppg | blink_int_high_Q | 5% | 8.2 [1.0, 23.0] | 5.5 [0.0, 18.0] | +2.7 [-14.0, +21.0] |
| backbone+rppg | blink_int_high_Q | 10% | 6.2 [0.0, 16.0] | 9.4 [1.0, 22.0] | -3.2 [-17.0, +13.0] |
| backbone+rppg | blink_int_low_Q | 1% | 2.4 [0.0, 10.0] | 1.3 [0.0, 6.0] | +1.1 [-5.0, +10.0] |
| backbone+rppg | blink_int_low_Q | 5% | 6.1 [0.0, 23.0] | 2.8 [0.0, 9.0] | +3.2 [-8.0, +23.0] |
| backbone+rppg | blink_int_low_Q | 10% | 6.0 [0.0, 23.0] | 5.7 [0.0, 27.0] | +0.3 [-26.0, +22.0] |
| backbone+rppg | real | 1% | 0.2 [0.0, 2.0] | 0.2 [0.0, 2.0] | +0.0 [-2.0, +2.0] |
| backbone+rppg | real | 5% | 0.4 [0.0, 3.0] | 0.4 [0.0, 3.0] | +0.0 [-2.0, +2.0] |
| backbone+rppg | real | 10% | 1.9 [0.0, 5.0] | 2.0 [0.0, 6.0] | -0.0 [-4.0, +4.0] |
| backbone+rppg | fake | 1% | 11.7 [0.0, 38.0] | 10.2 [0.0, 33.0] | +1.4 [-30.0, +38.0] |
| backbone+rppg | fake | 5% | 24.7 [2.0, 73.0] | 19.1 [0.0, 62.0] | +5.5 [-51.0, +70.0] |
| backbone+rppg | fake | 10% | 20.9 [2.0, 63.0] | 28.0 [4.0, 83.0] | -7.1 [-73.0, +51.0] |
| backbone+blink | ALL | 1% | 2.9 [0.0, 15.0] | 22.1 [6.0, 45.0] | -19.2 [-45.0, +4.0] |
| backbone+blink | ALL | 5% | 22.5 [5.0, 50.0] | 20.5 [4.0, 51.0] | +2.0 [-41.0, +29.0] |
| backbone+blink | ALL | 10% | 26.6 [8.0, 61.0] | 30.5 [13.0, 52.0] | -3.9 [-31.0, +36.0] |
| backbone+blink | rppg_snr_high_Q | 1% | 0.4 [0.0, 3.0] | 6.0 [0.0, 14.0] | -5.6 [-14.0, +2.0] |
| backbone+blink | rppg_snr_high_Q | 5% | 5.1 [0.0, 12.0] | 6.3 [0.0, 19.0] | -1.2 [-14.0, +6.0] |
| backbone+blink | rppg_snr_high_Q | 10% | 4.6 [0.0, 14.0] | 10.3 [2.0, 21.0] | -5.7 [-19.0, +9.0] |
| backbone+blink | blink_int_high_Q | 1% | 0.4 [0.0, 3.0] | 14.4 [4.0, 29.0] | -14.0 [-29.0, -2.0] |
| backbone+blink | blink_int_high_Q | 5% | 0.4 [0.0, 3.0] | 15.4 [4.0, 33.0] | -15.0 [-32.0, -4.0] |
| backbone+blink | blink_int_high_Q | 10% | 1.1 [0.0, 4.0] | 22.0 [9.0, 39.0] | -20.9 [-38.0, -8.0] |
| backbone+blink | blink_int_low_Q | 1% | 0.4 [0.0, 3.0] | 1.4 [0.0, 6.0] | -1.0 [-6.0, +2.0] |
| backbone+blink | blink_int_low_Q | 5% | 10.4 [2.0, 26.0] | 1.0 [0.0, 5.0] | +9.4 [+0.0, +26.0] |
| backbone+blink | blink_int_low_Q | 10% | 12.5 [2.0, 33.0] | 1.0 [0.0, 5.0] | +11.5 [-1.0, +33.0] |
| backbone+blink | real | 1% | 0.4 [0.0, 3.0] | 0.4 [0.0, 3.0] | -0.0 [-2.0, +2.0] |
| backbone+blink | real | 5% | 1.0 [0.0, 4.0] | 1.0 [0.0, 4.0] | +0.0 [-3.0, +3.0] |
| backbone+blink | real | 10% | 1.4 [0.0, 5.0] | 1.4 [0.0, 4.0] | +0.0 [-4.0, +3.0] |
| backbone+blink | fake | 1% | 2.3 [0.0, 13.0] | 21.8 [6.0, 39.0] | -19.4 [-39.0, +4.0] |
| backbone+blink | fake | 5% | 21.7 [5.0, 45.0] | 19.7 [4.0, 45.0] | +2.0 [-39.0, +27.0] |
| backbone+blink | fake | 10% | 24.9 [8.0, 52.0] | 28.8 [13.0, 46.0] | -3.9 [-28.0, +34.0] |
| full_fusion | ALL | 1% | 7.5 [0.0, 19.0] | 26.4 [7.0, 57.0] | -18.9 [-50.0, +4.0] |
| full_fusion | ALL | 5% | 24.3 [4.0, 75.0] | 36.2 [7.0, 88.0] | -11.9 [-73.0, +60.0] |
| full_fusion | ALL | 10% | 29.9 [9.0, 69.0] | 49.2 [15.0, 120.0] | -19.3 [-98.0, +34.0] |
| full_fusion | rppg_snr_high_Q | 1% | 1.7 [0.0, 8.0] | 6.3 [0.0, 19.0] | -4.6 [-18.0, +6.0] |
| full_fusion | rppg_snr_high_Q | 5% | 6.8 [0.0, 30.0] | 11.6 [0.0, 40.0] | -4.8 [-39.0, +28.0] |
| full_fusion | rppg_snr_high_Q | 10% | 8.3 [0.0, 34.0] | 19.3 [4.0, 57.0] | -11.0 [-57.0, +22.0] |
| full_fusion | blink_int_high_Q | 1% | 1.0 [0.0, 4.0] | 15.5 [3.0, 33.0] | -14.5 [-32.0, -1.0] |
| full_fusion | blink_int_high_Q | 5% | 0.4 [0.0, 3.0] | 21.2 [5.0, 46.0] | -20.8 [-45.0, -5.0] |
| full_fusion | blink_int_high_Q | 10% | 1.6 [0.0, 5.0] | 27.2 [10.0, 50.0] | -25.6 [-49.0, -8.0] |
| full_fusion | blink_int_low_Q | 1% | 3.0 [0.0, 9.0] | 2.4 [0.0, 7.0] | +0.6 [-6.0, +8.0] |
| full_fusion | blink_int_low_Q | 5% | 10.1 [1.0, 32.0] | 2.9 [0.0, 12.0] | +7.2 [-8.0, +31.0] |
| full_fusion | blink_int_low_Q | 10% | 13.1 [2.0, 31.0] | 3.5 [0.0, 21.0] | +9.6 [-15.0, +29.0] |
| full_fusion | real | 1% | 0.6 [0.0, 3.0] | 0.6 [0.0, 3.0] | +0.0 [-2.0, +2.0] |
| full_fusion | real | 5% | 0.8 [0.0, 3.0] | 0.8 [0.0, 3.0] | -0.0 [-3.0, +2.0] |
| full_fusion | real | 10% | 2.0 [0.0, 6.0] | 2.0 [0.0, 6.0] | +0.0 [-4.0, +4.0] |
| full_fusion | fake | 1% | 6.7 [0.0, 18.0] | 26.4 [9.0, 50.0] | -19.7 [-46.0, +3.0] |
| full_fusion | fake | 5% | 23.7 [5.0, 66.0] | 34.6 [7.0, 78.0] | -10.9 [-65.0, +53.0] |
| full_fusion | fake | 10% | 28.7 [9.0, 64.0] | 47.0 [16.0, 110.0] | -18.3 [-91.0, +38.0] |

## 4. Cross-backbone comparison

Direct B4-vs-CLIP comparison on the ALL stratum:

| Variant | B4 mean delta [CI] | CLIP mean delta [CI] | B4-CLIP gap |
|---|---|---|---|
| backbone+rppg | +0.0162 [-0.0151, +0.0506] | -0.0011 [-0.0170, +0.0139] | +0.0173 |
| backbone+blink | +0.0008 [-0.0041, +0.0073] | -0.0037 [-0.0093, +0.0020] | +0.0045 |
| full_fusion | +0.0155 [-0.0145, +0.0476] | -0.0040 [-0.0214, +0.0125] | +0.0196 |
