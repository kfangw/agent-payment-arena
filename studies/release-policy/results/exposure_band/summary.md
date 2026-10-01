# Exposure-band B4 (E-fitted, 161-point grid)

bp is relative to the mean exposure of $49.50; brackets are 95% paired block-bootstrap intervals (blocks of 50, 10,000 resamples).

## Difference to the compiled policy

| flow | compiled − B4 | compiled − B4-band(4) | compiled − B4-band(3) | compiled − B4-band(5) |
|---|---|---|---|---|
| F1 | +3.8 [+3.5, +4.1] | +3.8 [+3.5, +4.1] | +3.8 [+3.5, +4.2] | +4.2 [+3.8, +4.5] |
| F2 | +4.3 [+3.7, +4.8] | +4.6 [+3.7, +5.5] | +5.5 [+4.6, +6.5] | +4.6 [+3.7, +5.5] |
| F3 | +1.4 [+1.0, +1.8] | +1.1 [+0.7, +1.5] | +1.1 [+0.7, +1.5] | +1.1 [+0.6, +1.5] |

## B4-band − B4

| flow | 3 bands | 4 bands | 5 bands |
|---|---|---|---|
| F1 | -0.0 [-0.1, +0.0] | +0.0 [-0.0, +0.0] | -0.4 [-0.5, -0.2] |
| F2 | -1.3 [-2.3, -0.2] | -0.3 [-1.3, +0.7] | -0.4 [-1.4, +0.7] |
| F3 | +0.4 [+0.3, +0.5] | +0.4 [+0.3, +0.5] | +0.4 [+0.3, +0.5] |

## Verdict (main specification: 4 bands, margin 10 bp)

- F1: interval [+3.5, +4.1], inside the margin
- F2: interval [+3.7, +5.5], inside the margin
- F3: interval [+0.7, +1.5], inside the margin

## Per-band choice (k, a, b)

- F1 B4: (0, 0.03125, 0.875)
  - 3 bands (inner edges 19.52, 46.13 USD): (0, 0.6375, 0.7125); (0, 0.0312, 0.8500); (0, 0.6250, 0.8750)
  - 4 bands (inner edges 15.34, 30.01, 58.86 USD): (0, 0.6438, 0.7125); (0, 0.0250, 0.7188); (0, 0.5500, 0.8500); (0, 0.6063, 0.8750)
  - 5 bands (inner edges 12.97, 23.35, 38.66, 69.63 USD): (0, 0.6438, 0.7125); (0, 0.0250, 0.6688); (0, 0.0312, 0.8313); (0, 0.6250, 0.8500); (0, 0.6750, 0.8750)
- F2 B4: (0, 0.1125, 0.8)
  - 3 bands (inner edges 19.52, 46.17 USD): (0, 0.1438, 0.6000); (0, 0.1313, 0.8625); (0, 0.1000, 0.7250)
  - 4 bands (inner edges 15.27, 30.02, 58.91 USD): (0, 0.1438, 0.5813); (0, 0.1313, 0.8625); (0, 0.1188, 0.7938); (0, 0.1000, 0.7500)
  - 5 bands (inner edges 12.92, 23.31, 38.63, 69.45 USD): (0, 0.1438, 0.5813); (0, 0.1250, 0.6750); (0, 0.1313, 0.8625); (0, 0.1188, 0.8000); (0, 0.1000, 0.7375)
- F3 B4: (0, 0.00625, 0.9625)
  - 3 bands (inner edges 19.54, 46.28 USD): (0, 0.6125, 0.8938); (0, 0.0063, 0.9500); (0, 0.0063, 0.9688)
  - 4 bands (inner edges 15.30, 30.04, 59.16 USD): (0, 0.6125, 0.6750); (0, 0.5312, 0.9250); (0, 0.0063, 0.9375); (0, 0.0063, 0.9688)
  - 5 bands (inner edges 12.95, 23.29, 38.73, 70.02 USD): (0, 0.5125, 0.6750); (0, 0.5312, 0.9000); (0, 0.6063, 0.9313); (0, 0.0063, 0.9625); (0, 0.0063, 0.9688)

## Security: misuse payments released

| flow | policy | share of misuse payments released | released misuse exposure per payment (USD) |
|---|---|---|---|
| F1 | B4 | 39.628% | 0.09822 |
| F1 | B4-band(3) | 39.889% | 0.09875 |
| F1 | B4-band(4) | 39.825% | 0.09851 |
| F1 | B4-band(5) | 40.082% | 0.10055 |
| F2 | B4 | 9.813% | 0.97631 |
| F2 | B4-band(3) | 12.529% | 0.94990 |
| F2 | B4-band(4) | 12.149% | 0.93906 |
| F2 | B4-band(5) | 12.180% | 0.96538 |
| F3 | B4 | 3.759% | 0.09345 |
| F3 | B4-band(3) | 3.799% | 0.09367 |
| F3 | B4-band(4) | 3.797% | 0.09367 |
| F3 | B4-band(5) | 3.786% | 0.09375 |

## Reproduction check

- F1: one band reproduces the b4ref (k, a, b) and tuning mean: True; largest gap of the B4 evaluation block sums to b4ref: 1.14e-12
- F2: one band reproduces the b4ref (k, a, b) and tuning mean: True; largest gap of the B4 evaluation block sums to b4ref: 9.09e-13
- F3: one band reproduces the b4ref (k, a, b) and tuning mean: True; largest gap of the B4 evaluation block sums to b4ref: 1.36e-12
