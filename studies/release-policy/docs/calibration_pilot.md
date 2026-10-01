# Calibration re-estimated from the pilot (design file, fixed before the run)

Scripts: `scripts/run_G.py` (plan / smoke / eval), `scripts/analyze_G.py` (metrics, bootstrap, reading rule), `scripts/fastreplay_G.py` (vectorized replay, checked against the repository replay: per-payment difference 0 on the full seed-108 evaluation set and on the first 500,000 payments of eight policies of the main run), `scripts/tests_G.py` (information-contract tests), `scripts/indep_G.py` (independent recomputation from the block sums; max difference from `summary.json` 4.3e-11). Results: `results/calibration_pilot/` (summary.json, G_results.npz, manifest.json, plan/, logs). Per-policy block-sum files (1,720 npz, 1.4 GB) are not included; `run_G.py all --workers 2` regenerates them in about 100 minutes on two cores.

World: E-fitted, flow F2, constants of the main run (m=0.35, h=1, C=0.5, c_w=0.006/tick, tau=10, q=0.75, answers independent of intent). The policy reads x = min(zeta*pi, 1) in place of pi; the simulator keeps pi and the true intent. True zeta in {0.5, 0.7, 1, 1.4, 2, 3}.

Evaluation: seed 110 (never used before), 5,663,400 payments in 113,268 blocks of 50, shared by all policies (paired). Learning: 30 seeds, default_rng([110, 500, s]), 5,000 payments each; pilot n in {130, 500, 2000, 5000} is the first n.

Policies: pi_ref (true q, true pi); Comp_naive (q_hat = K/n, reads x as is); Comp_cal (same table, reads min(x/zeta_hat, 1)); B4_n (two-threshold watch rule tuned on the same n payments with hindsight payoffs and labels, 161-point grid over x); B4_200k (tuned on 200,000 payments, default_rng([110, 700])).

Estimator: zeta_hat = argmax over the grid exp(0.01k), k=-200..200, of the likelihood of P(misuse | x) = min(x/zeta, 1) on the answered payments (t_ans <= tau); ties to the smallest zeta; no misuse labels -> grid top, all misuse -> grid bottom (counted; both occurred zero times). Information contract enforced in code: Comp_cal reads t_ans, x, and the intent label of answered payments only.

Metrics (bp of $49.50 mean exposure): L_cal = J(pi_ref) - J(Comp_cal); L_naive; L_B4; D_cal = J(Comp_cal) - J(B4_n). Two-way block bootstrap (blocks x seeds, 1,000 draws, seed 20260930; seed resampling shared across n and zeta).

Reading rule, fixed in advance: n* is the smallest n at which, for every true zeta, mean L_cal <= 10 bp and at most 3 of 30 pilots exceed 10 bp. n* <= 500 -> "the same few hundred questions"; 500 < n* <= 5000 -> "the same pilot, made larger"; none -> negative result. Runtime rule: full evaluation if projected replays x 9.4 s / 2 workers <= 5 h (it was 1.5 h; full size used).

Result: n* = 2,000 (n=130 fails at every zeta; n=500 fails at zeta=3 only, 4 pilots above the margin; n=2,000 and 5,000 pass). See summary.json.
