# Experiment L: automatic summary

Generated from summary.json. E-fitted, F2, seed 8. Unit: bp of the mean exposure $49.50; intervals are paired block bootstrap 95%. Comp_lambda is not built (the compiler does not take an answer hazard per intent).

| lambda | rho_m | Comp - B4_lambda | Comp - Cutoff | Comp - B4_main | misuse granted, Comp / B4_lambda / Cutoff (%) | share of Comp verify payments without an answer (all / misuse / honest) | B4_lambda (k, a, b) |
|---|---|---|---|---|---|---|---|
| 1 | 0.12945 | 4.26 [3.70, 4.78] | 622.84 [618.68, 627.07] | 4.26 [3.70, 4.78] | 14.21 / 9.81 / 45.29 | 0.2500 / 0.2506 / 0.2498 | (0, 0.1125, 0.8) |
| 0.5 | 0.06472 | 4.50 [3.95, 5.02] | 604.31 [600.18, 608.54] | 4.50 [3.95, 5.02] | 14.21 / 9.81 / 45.29 | 0.3207 / 0.5130 / 0.2498 | (0, 0.1125, 0.8) |
| 0.25 | 0.03236 | 4.67 [4.11, 5.19] | 592.38 [588.28, 596.61] | 4.66 [4.11, 5.18] | 14.21 / 9.81 / 45.29 | 0.3766 / 0.7203 / 0.2498 | (0, 0.1125, 0.75625) |
| 0 | 0.00000 | 4.85 [4.29, 5.36] | 578.25 [574.19, 582.45] | 4.86 [4.31, 5.37] | 14.21 / 9.81 / 45.29 | 0.4520 / 1.0000 / 0.2498 | (0, 0.1125, 0.75625) |

## Reading-rule sentences
- Comp - B4_lambda does not fall below -10 bp at any lambda (lambda = 0 included).
- Inside the margin at lambda = 0 too. Check: at lambda = 0, 0.000% of misuse payments answer within tau (the misuse answer hazard is clipped to 1e-9; minimum t_ans = 619). Of the 952652 misuse payments that Comp verified, 952652 got no answer and 0 were granted after the verify. A query that times out ends without a release, so it filters misuse. In the replay rule an answered misuse payment is not released either (a grant after an answer goes to honest intents only), so lambda changes only the waiting cost of misuse verify payments (at most tau ticks).
  - lambda = 1: of 952652 misuse payments verified by Comp, 0 were granted after the verify; misuse no-answer share 0.2506.
  - lambda = 0.5: of 952652 misuse payments verified by Comp, 0 were granted after the verify; misuse no-answer share 0.5130.
  - lambda = 0.25: of 952652 misuse payments verified by Comp, 0 were granted after the verify; misuse no-answer share 0.7203.
  - lambda = 0: of 952652 misuse payments verified by Comp, 0 were granted after the verify; misuse no-answer share 1.0000.
- Misuse granted (Comp) as lambda goes from 1 to 0: 14.211% -> 14.211% -> 14.211% -> 14.211%; it stays the same.
- Misuse granted (B4_lambda) as lambda goes from 1 to 0: 9.813% -> 9.813% -> 9.813% -> 9.813%; it stays the same.
- Misuse granted (Cutoff) as lambda goes from 1 to 0: 45.294% -> 45.294% -> 45.294% -> 45.294%; it stays the same.

## Checks
- lambda = 1 draw against the D2 F2 draw: J(Comp) equal, J(Cutoff) equal, J(B4_main) equal, B4_1 tuning result equals the D2 B4: yes.
- lambda = 1 block sums (15 per-episode arrays compared element by element with the D2 npz): all equal.
