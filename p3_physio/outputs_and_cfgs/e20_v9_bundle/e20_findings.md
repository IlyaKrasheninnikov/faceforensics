# E20 - Paired bootstrap + reliability test + threshold curves

**Date:** 2026-05-10  
**Bootstrap iterations:** 2000; alpha = 0.05

## 1. Paired subject-cluster bootstrap on delta AUC

CIs that exclude zero indicate statistically reliable differences at the subject-cluster level (not just clip-level). 5-seed mean +/- paired-bootstrap CI.

### backbone+rppg vs backbone-only

| Stratum | n_subj | mean delta AUC | 95% CI | excludes 0? |
|---|---|---|---|---|
| ALL | 72 | -0.0008 | [-0.0066, +0.0048] | no |
| rppg_snr_high_Q | 21 | -0.0023 | [-0.0157, +0.0094] | no |
| rppg_snr_low_Q | 37 | +0.0008 | [-0.0048, +0.0070] | no |
| blink_int_high_Q | 28 | +0.0001 | [-0.0093, +0.0101] | no |
| blink_int_low_Q | 35 | -0.0023 | [-0.0135, +0.0076] | no |

### backbone+blink vs backbone-only

| Stratum | n_subj | mean delta AUC | 95% CI | excludes 0? |
|---|---|---|---|---|
| ALL | 72 | -0.0037 | [-0.0092, +0.0017] | no |
| rppg_snr_high_Q | 21 | -0.0005 | [-0.0132, +0.0166] | no |
| rppg_snr_low_Q | 37 | -0.0035 | [-0.0132, +0.0066] | no |
| blink_int_high_Q | 28 | -0.0051 | [-0.0135, +0.0033] | no |
| blink_int_low_Q | 35 | -0.0046 | [-0.0140, +0.0061] | no |

### full_fusion vs backbone-only

| Stratum | n_subj | mean delta AUC | 95% CI | excludes 0? |
|---|---|---|---|---|
| ALL | 72 | -0.0040 | [-0.0125, +0.0043] | no |
| rppg_snr_high_Q | 21 | -0.0020 | [-0.0225, +0.0183] | no |
| rppg_snr_low_Q | 37 | -0.0016 | [-0.0136, +0.0118] | no |
| blink_int_high_Q | 28 | -0.0041 | [-0.0158, +0.0086] | no |
| blink_int_low_Q | 35 | -0.0048 | [-0.0191, +0.0104] | no |


## 2. Reliability-conditioned physiology test

From part 1 (rppg_snr_high_Q stratum, n=440 / 440 clips):

- backbone+rppg on high-SNR rPPG quartile: delta = -0.0023, CI [-0.0157, +0.0094], excludes 0: no
- backbone+blink on high-SNR rPPG quartile: delta = -0.0005, CI [-0.0132, +0.0166], excludes 0: no
- full_fusion on high-SNR rPPG quartile: delta = -0.0020, CI [-0.0225, +0.0183], excludes 0: no

For the reliability hypothesis to be confirmed, the high-SNR row for +rPPG (or any variant) would need to show a positive delta with a CI that excludes zero. If no variant achieves this, the redundancy argument is strengthened.

## 3. Threshold-level rescue/regression curves

Per-seed mean rescue/regression counts at FPR in {1, 5, 10}%.

### backbone+rppg

| Stratum | FPR=1% | FPR=5% | FPR=10% |
|---|---|---|---|
| ALL | net=+1.8 (rescue% 54) | net=+6.8 (rescue% 58) | net=-8.6 (rescue% 42) |
| rppg_snr_high_Q | net=+0.2 (rescue% 52) | net=+0.4 (rescue% 51) | net=-5.2 (rescue% 38) |
| blink_int_high_Q | net=+0.0 (rescue% 50) | net=+2.6 (rescue% 59) | net=-3.0 (rescue% 40) |
| blink_int_low_Q | net=+1.0 (rescue% 63) | net=+2.8 (rescue% 66) | net=+0.4 (rescue% 52) |
| real | net=+0.0 (rescue% 50) | net=+0.0 (rescue% 50) | net=+0.0 (rescue% 50) |
| fake | net=+1.8 (rescue% 54) | net=+6.8 (rescue% 58) | net=-8.6 (rescue% 41) |

### backbone+blink

| Stratum | FPR=1% | FPR=5% | FPR=10% |
|---|---|---|---|
| ALL | net=-19.2 (rescue% 11) | net=+1.4 (rescue% 52) | net=-3.8 (rescue% 47) |
| rppg_snr_high_Q | net=-5.6 (rescue% 6) | net=-1.4 (rescue% 44) | net=-5.6 (rescue% 31) |
| blink_int_high_Q | net=-14.0 (rescue% 3) | net=-15.0 (rescue% 3) | net=-21.2 (rescue% 4) |
| blink_int_low_Q | net=-1.0 (rescue% 22) | net=+9.6 (rescue% 91) | net=+11.4 (rescue% 93) |
| real | net=+0.0 (rescue% 50) | net=+0.0 (rescue% 50) | net=+0.0 (rescue% 50) |
| fake | net=-19.2 (rescue% 10) | net=+1.4 (rescue% 52) | net=-3.8 (rescue% 46) |

### full_fusion

| Stratum | FPR=1% | FPR=5% | FPR=10% |
|---|---|---|---|
| ALL | net=-19.2 (rescue% 22) | net=-11.2 (rescue% 41) | net=-19.2 (rescue% 38) |
| rppg_snr_high_Q | net=-4.4 (rescue% 22) | net=-4.8 (rescue% 37) | net=-11.2 (rescue% 30) |
| blink_int_high_Q | net=-14.2 (rescue% 6) | net=-21.2 (rescue% 2) | net=-25.4 (rescue% 6) |
| blink_int_low_Q | net=+0.6 (rescue% 56) | net=+7.2 (rescue% 77) | net=+10.0 (rescue% 79) |
| real | net=+0.0 (rescue% 50) | net=+0.0 (rescue% 50) | net=+0.0 (rescue% 50) |
| fake | net=-19.2 (rescue% 21) | net=-11.2 (rescue% 40) | net=-19.2 (rescue% 37) |

