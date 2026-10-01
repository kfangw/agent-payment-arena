# Experiment HF: automatic summary

5,663,400 evaluation payments (E-fitted, F2, main-run evaluation draw), paired block bootstrap with 1000 resamples, 95% intervals. bp is of $49.50. Exposure is in dollars per input payment.

## Metrics by h

| h | B4_h (k, a, b) | (a) J_h(Comp_h) - J_h(B4_h), bp | (b) J_1(Comp_h) | J_1(Comp_h) - J_1(Comp_1), bp | J_1(B4_h) |
|---|---|---|---|---|---|
| 0.25 | (0, 0.3875, 0.8) | 5.49 [4.93, 6.12] | 5.740598 | -645.13 [-649.88, -640.38] | 5.531434 |
| 0.5 | (0, 0.2188, 0.8) | 6.52 [5.69, 7.30] | 8.190438 | -150.21 [-153.16, -147.17] | 8.130201 |
| 1 | (0, 0.1125, 0.8) | 4.26 [3.70, 4.76] | 8.933994 | 0.00 [0.00, 0.00] | 8.912930 |
| 2 | (0, 0.06875, 0.8) | 6.14 [4.72, 7.53] | 8.666688 | -54.00 [-56.12, -51.97] | 8.713349 |
| 4 | (0, 0.0375, 0.8) | 3.54 [2.13, 5.02] | 8.339498 | -120.10 [-122.44, -117.59] | 8.402737 |

| h | policy | (c) misuse granted, % | (c) misuse exposure, $ per payment | (d) honest released, % (grant + executed after an answer) | (d) queries, % |
|---|---|---|---|---|---|
| 0.25 | Comp_0.25 | 79.75 [79.68, 79.82] | 7.2954 [7.2711, 7.3195] | 98.52 (grant 94.49, after answer 4.03) | 8.27 |
| 0.25 | B4_0.25 | 76.03 [75.95, 76.10] | 7.5422 [7.5184, 7.5666] | 98.32 (grant 93.41, after answer 4.91) | 10.06 |
| 0.5 | Comp_0.5 | 41.93 [41.84, 42.02] | 3.4641 [3.4476, 3.4800] | 92.98 (grant 73.11, after answer 19.87) | 32.68 |
| 0.5 | B4_0.5 | 35.51 [35.42, 35.59] | 3.5221 [3.5049, 3.5390] | 92.03 (grant 68.60, after answer 23.43) | 38.01 |
| 1 | Comp_1 | 14.21 [14.15, 14.28] | 1.0769 [1.0671, 1.0866] | 85.00 (grant 42.41, after answer 42.58) | 62.42 |
| 1 | B4_1 | 9.81 [9.76, 9.87] | 0.9763 [0.9663, 0.9855] | 83.67 (grant 35.62, after answer 48.05) | 69.53 |
| 2 | Comp_2 | 3.46 [3.43, 3.49] | 0.2480 [0.2434, 0.2524] | 78.95 (grant 19.38, after answer 59.58) | 82.71 |
| 2 | B4_2 | 3.24 [3.21, 3.28] | 0.3255 [0.3199, 0.3312] | 79.51 (grant 19.24, after answer 60.27) | 83.95 |
| 4 | Comp_4 | 0.70 [0.69, 0.72] | 0.0485 [0.0463, 0.0504] | 75.85 (grant 7.61, after answer 68.24) | 92.53 |
| 4 | B4_4 | 0.75 [0.74, 0.77] | 0.0758 [0.0730, 0.0784] | 76.73 (grant 8.22, after answer 68.50) | 93.26 |

## The ten curve points (misuse exposure in $ per payment, J_1)

| point | misuse exposure, $ per payment | J_1 | misuse granted, % | queries, % |
|---|---|---|---|---|
| Comp_0.25 | 7.2954 | 5.740598 | 79.75 | 8.27 |
| B4_0.25 | 7.5422 | 5.531434 | 76.03 | 10.06 |
| Comp_0.5 | 3.4641 | 8.190438 | 41.93 | 32.68 |
| B4_0.5 | 3.5221 | 8.130201 | 35.51 | 38.01 |
| Comp_1 | 1.0769 | 8.933994 | 14.21 | 62.42 |
| B4_1 | 0.9763 | 8.912930 | 9.81 | 69.53 |
| Comp_2 | 0.2480 | 8.666688 | 3.46 | 82.71 |
| B4_2 | 0.3255 | 8.713349 | 3.24 | 83.95 |
| Comp_4 | 0.0485 | 8.339498 | 0.70 | 92.53 |
| B4_4 | 0.0758 | 8.402737 | 0.75 | 93.26 |

## J_1 gap between the Comp_h curve and each B4_h point (same misuse exposure, curve interpolated linearly)

| h | B4_h exposure, $ per payment | Comp curve J_1 - B4_h J_1 (bp) |
|---|---|---|
| 0.25 | 7.5422 | outside the curve range |
| 0.5 | 3.5221 | 4.67 [3.98, 5.33] |
| 1 | 0.9763 | -2.30 [-3.03, -1.66] |
| 2 | 0.3255 | -4.38 [-5.33, -3.44] |
| 4 | 0.0758 | -3.75 [-4.61, -2.83] |

## Reading rules (spec section 3)

- Monotonicity: the misuse exposure of Comp_h decreases with h. J_1(Comp_h) does not decrease with h (it is largest at h = 1 and falls on both sides). The two conditions of the rule do not hold together, so the rule sentence is not used and this fact is reported instead.
- B4_h: misuse exposure decreasing, J_1 largest at h = 1, falling on both sides.
- Position: the Comp_h curve has a higher J_1 than the B4_h point for h = 0.5 and a lower or equal J_1 for h = 1, 2, 4. The two cross between these.
- B4_h points outside the curve range: h = 0.25 (no interpolation; compare by point values in the tables).

## Checks (h = 1 against experiment D2)

- table hash equal: True
- J_1(Comp_1) = 8.933993969941707, D2 8.933993969941707, equal: True
- J_1(B4_1) = 8.91293001429241, D2 8.91293001429241, equal: True
- B4_1 (k, a, b, tuning mean) = [0, 0.1125, 0.8, 8.872650462538848], D2 [0, 0.1125, 0.8, 8.872650462538848], equal: True
