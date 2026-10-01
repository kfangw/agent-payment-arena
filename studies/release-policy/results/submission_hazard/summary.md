# Experiment F0: automatic summary

Generated from summary.json. Unit: bp of the mean exposure $49.50; intervals are paired block bootstrap 95%.

## Reading rule applied
- Window A_all90d: revert share 0.405% = 0.81 times the declared value, inside the range 1/2 to 2 times.
- Window B_cb90d: revert share 1.216% = 2.43 times the declared value, outside the range 1/2 to 2 times.
- Rule: (b) applies (window outside the range: B_cb90d). Recompiled f0 = 0.01216147.

## Metrics by flow
| flow | (1) W_m: Comp_m - Comp_d | (2) W_d: Comp_d - Comp_m | (3) W_m: Comp_m - B4_m | W_m: Comp_d - B4_m |
|---|---|---|---|---|
| F1 | 61.71 [60.21, 63.22] | 9.11 [8.12, 10.05] | 4.22 [3.91, 4.56] | -57.49 [-59.00, -55.96] |
| F2 | 20.28 [19.43, 21.08] | 2.92 [2.38, 3.46] | 25.99 [24.80, 27.26] | 5.71 [4.70, 6.74] |
| F3 | 59.37 [57.94, 60.79] | 9.40 [8.46, 10.25] | 5.03 [4.60, 5.44] | -54.33 [-55.82, -52.80] |

Misuse granted (%):
| flow | W_m Comp_d | W_m Comp_m | W_m B4_m | W_m Cutoff_m | W_d Comp_d | W_d Comp_m |
|---|---|---|---|---|---|---|
| F1 | 39.62 | 39.21 | 39.23 | 39.63 | 39.62 | 39.44 |
| F2 | 14.21 | 14.04 | 7.59 | 43.86 | 14.21 | 14.14 |
| F3 | 3.76 | 3.71 | 3.71 | 3.76 | 3.76 | 3.74 |

First-action shares of Comp_m (%; of all payments, by arrival regime running/halted):
- F1: running grant 0.000, reject 0.275, verify 0.081, wait 99.567; halted grant 0.000, reject 0.000, verify 0.000, wait 0.077. wait in total 99.644%.
  - wait (running): 5638871 payments, pi0 in [0.0000, 0.0277], v in [0.500, 2000.000]
  - wait (halted): 4357 payments, pi0 in [0.0000, 0.0175], v in [1.058, 819.671]
- F2: running grant 0.000, reject 0.781, verify 62.385, wait 36.758; halted grant 0.000, reject 0.025, verify 0.000, wait 0.052. wait in total 36.810%.
  - wait (running): 2081766 payments, pi0 in [0.0000, 0.2550], v in [0.500, 2000.000]
  - wait (halted): 2918 payments, pi0 in [0.0024, 0.3134], v in [0.922, 1136.177]
- F3: running grant 0.000, reject 5.518, verify 0.142, wait 94.261; halted grant 0.000, reject 0.004, verify 0.000, wait 0.074. wait in total 94.335%.
  - wait (running): 5338400 payments, pi0 in [0.0000, 0.0012], v in [0.500, 2000.000]
  - wait (halted): 4188 payments, pi0 in [0.0000, 0.0003], v in [0.500, 1305.757]

## Checks
- F1: J(Comp_d) on W_d equals the D2 J_comp (difference 0); each world's own table is ahead: W_m yes, W_d yes.
- F2: J(Comp_d) on W_d equals the D2 J_comp (difference 0); each world's own table is ahead: W_m yes, W_d yes.
- F3: J(Comp_d) on W_d equals the D2 J_comp (difference 0); each world's own table is ahead: W_m yes, W_d yes.

## Rule (c) sentence
- The loss exceeds the 10 bp margin: F1 (1) W_m: comp_m - comp_d = 61.71 bp; F2 (1) W_m: comp_m - comp_d = 20.28 bp; F3 (1) W_m: comp_m - comp_d = 59.37 bp. State this as it is and narrow the scope of "rail constants may be rough".
