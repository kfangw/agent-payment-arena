# Experiment F0 spec: the measured stage-0 submission failure f0

Fixed after the data were obtained and before any policy was run. Script: `scripts/run_F0.py`; data: `data/x402_amounts/revert_share.json`; results: `results/submission_hazard/` (`summary.json`, `summary.md`, `plan/`). The draw and the replay of experiments D2 and C are reused.

Motivation: the rail constants f0 and f_i of the E-fitted world are declared values, and f0 may violate the assumption under which the structure results hold.

## 1. Question

The stage-0 submission failure probability of E-fitted is the declared 0.005, which is 0.83 times c_w = 0.006. What is the measured value of f0 if it is taken to be the revert share of the transactions that real x402 facilitators send on Base, and how different is a policy compiled with that value from the policy compiled with the declared value?

## 2. Data (`data/x402_amounts/revert_share.json`)

From the facilitator address list of the public Dune dashboard "x402 Facilitators", the `success` flag of the transactions in `base.transactions` with `from` in that list was counted. Two windows (the same as in experiment E): A, all listed facilitators, last 90 days: 147,059 transactions, 596 reverted (0.405%); B, the Coinbase facilitator, its last 90 active days: 44,276,970 transactions, 538,473 reverted (1.216%). Meaning: if a settle transaction sent by a facilitator reverts, that payment has failed, which corresponds to the stage-0 failure (absorption in FAILED) of the model. Limit: whether a reverted payment is submitted again is unknown (if the resubmission succeeds, the real failure rate is lower than this). Both points are stated with the result.

## 3. Reading rules (fixed after the data were seen, before any policy run)

* (a) If the measured value is within 1/2 to 2 times the declared 0.005 in both windows: state the measured value and "x times the declared value" in one sentence, no recompilation (cite the loss bound of the input-error analysis).
* (b) If either window is outside that range (window B, 1.216%, is 2.4 times the declared value and is): recompile the E-fitted table with that f0 (all other constants unchanged) and replay it on the main-run evaluation draw (F1, F2, F3; seeds 7, 8, 9). Report (1) in the world with f0 = 0.01216, J(Comp_measured) - J(Comp_declared): what the declared-value policy loses when the world has the measured f0 (cost of the misspecification); (2) in the declared world, J(Comp_declared) - J(Comp_measured) (the reverse direction); (3) the difference to B4 retuned in the measured-f0 world; in bp of the mean exposure 49.50, with paired block bootstrap 95% intervals. Since the measured f0 = 0.01216 exceeds c_w = 0.006, the condition of Theorem 3.2 is reversed (for f0 > c_w, granting on arrival need not be the unique optimum). The first-action shares of Comp_measured (the wait share in particular) must be reported.
* (c) In either case the result is reported with one summary table. If a loss exceeds the margin (10 bp), state it as it is and narrow the scope of the phrase "rail constants may be rough".

## 4. Design for (b)

* World W_m: E-fitted with only p_sub (f0) replaced by 0.01216. World W_d: the main-run E-fitted (f0 = 0.005).
* Policies: Comp_d (main-run table), Comp_m (compiled for W_m), B4_m (tuned on the W_m tuning split of 200,000 payments, 161-point grid, the existing tuner).
* Evaluation: for each flow, in W_m with seeds 7, 8, 9: the 200,000 tuning payments first, then 5,663,400 evaluation payments (the method of `run_D2.draw_main` with W_m as the environment). The W_d replay uses the same draw.
* Metrics: (1), (2), (3) above, misuse granted, first-action shares of Comp_m (grant, reject, verify, wait), and the range of pi0 and v of the payments that wait.
* Checks: J(Comp_d) in W_d equals the D2 value J_comp (difference 0). The sign of Comp_m replayed in W_d and of Comp_d replayed in W_m agrees with the parallel-lines result (each world's own table must be ahead).

## 5. Run time

3 compilations x 3 s, 3 draws x 3 min, 9 replays x 10 s, 3 B4 tunings x 2 min, about 20 minutes.

## Change log

* Value of f0. f[0] of W_m is the window-B revert share rounded to 0.01216147 (data value 0.0121614690...). All other constants are as in E-fitted (only `f` was replaced, with `dataclasses.replace`).
* Exercise rule. The exercise rule `ex` of the replay (whether a grant follows a verify answer, `ex_pos` of `replay_outage`) is part of the world, so it was computed in each world with that world's `window_AD` (`ex_m` for W_m, the main-run `ex` for W_d). The two differ (`ex_m_differs` in the check output).
* Draw. The repository's draw reads only the length of `f`, not its values, so the W_m draw and the W_d draw of the same seed are identical arrays (all fields compared in the check phase). The draw was therefore made once per flow and replayed in both worlds. That J(Comp_d) in W_d equals D2's J_comp confirms that the draw is the D2 draw.
* Reference policies added. The Cutoff of W_m uses the threshold pihat(0) of the W_m constants (f0 included). B4_d (the retuned main-run B4 of D2, `results/no_channel/plan/b4main_*.json`) was replayed in both worlds, so B4_m - B4_d is also reported. The reading rules were not changed.
