# F4 summary

Eval draw: seed 111, n=500,000 (10,000 blocks), realized mean exposure 49.5058. bp = (J_comp - J_other) / mean exposure * 1e4. Draw CI: paired block bootstrap, 1000 resamples, seed 20260930, 50-payment blocks. Row CI: percentiles of per-rep differences (200 reps, mean 7.3 s/rep). B4_F4 = (k=0, a=0.01250, b=0.62500).

## 1. Band and calibration (eval split)

| quantity | value |
|---|---|
| band [0.1625, 0.63125) share of eval rows | 0.1152% (82 rows) |
| band fraud rate vs mean pi0 | 0.6829 vs 0.6194 (56 frauds) |
| zeta_hat [95% row-bootstrap CI] | 0.8958 [0.7788, 1.0411] (200 reps) |
| band share in eval draw (payments) | 0.1118%; row-boot mean 0.1156% [0.0924, 0.1448] |

Fixed-edge bins:

| bin | n | mean pi0 | fraud rate | frauds |
|---|---|---|---|---|
| [0, 0.05) | 71121 | 0.00044 | 0.00053 | 38 |
| [0.05, 0.1625) | 0 | - | - | 0 |
| [0.1625, 0.3) | 0 | - | - | 0 |
| [0.3, 0.63125) | 82 | 0.61935 | 0.68293 | 56 |
| [0.63125, 0.9) | 0 | - | - | 0 |
| [0.9, 1] | 0 | - | - | 0 |

## 2. Payoff differences, Comp minus X (bp)

| comparison | point (draw) [draw CI] | row-boot median [row CI] |
|---|---|---|
| Comp - B4_F4 | +4.49 [+3.34, +5.80] | +4.48 [+4.44, +4.49] |
| Comp - Cutoff | +4.63 [+3.46, +5.93] | +4.87 [+4.55, +5.30] |
| Comp - Comp_nv | +0.15 [-0.05, +0.36] | +0.40 [+0.08, +0.82] |

J ($ per payment): Comp 16.854890, Comp_nv 16.854155, Cutoff 16.831993, B4_F4 16.832680

## 3. Misuse and verify

| policy | misuse pass rate (%) [draw CI] | row-boot mean [row CI] | misuse exposure ($ total) | per payment ($) | grant % |
|---|---|---|---|---|---|
| Comp | 38.808 [35.302, 42.326] | 40.171 [30.159, 50.850] | 15010.23 | 0.03002 | 99.842 |
| Comp_nv | 38.808 [35.302, 42.326] | 40.171 [30.159, 50.850] | 15010.23 | 0.03002 | 99.842 |
| Cutoff | 38.808 [35.302, 42.326] | 40.186 [30.159, 50.850] | 15010.23 | 0.03002 | 99.888 |
| B4_F4 | 38.808 [35.302, 42.326] | 40.186 [30.159, 50.850] | 15010.23 | 0.03002 | 99.888 |

Misuse payments in the draw: 688 (rate 0.1376%). Comp verify share at arrival: 0.0986% [0.0898, 0.1072]. Comp first actions (counts): {'grant': 499011, 'reject': 66, 'verify': 493, 'wait': 430}.

## 4. Classifier and split

ECE (fixed bins) 0.000167; AUC 0.8267, AP 0.4286 (calibrated; raw scores AUC 0.7327, AP 0.4387). Base rate (eval) 0.00132.

| split | rows | frauds | Time range |
|---|---|---|---|
| train | 142,403 | 269 | 0 - 84691 |
| calib | 71,201 | 129 | 84692 - 139320 |
| eval | 71,203 | 94 | 139320 - 172792 |

## 5. Reading rules (docs/spec-F4.md, section 6)

- zeta: R-zeta-1: calibration error inside the first grid point of the distortion-sensitivity curve (zeta_hat 0.8958, CI [0.7788, 1.0411]).
- Comp vs B4_F4: R-b4-1: equal within the margin (row CI inside [-10, +10] bp).
- Channel: R-ch-1: channel hardly used in F4 (|Comp - Comp_nv| = 0.15 bp <= 2).
