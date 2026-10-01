# Experiment L spec: misuse intents that answer less (a curve for violating H2)

Fixed before any result was seen. Script: `scripts/run_L.py`; results: `results/misuse_answer/` (`summary.json`, `summary.md`, `plan/`).

Motivation: only one value, lambda = 1/4, had been run, and only in a reduced sample. Code used: `draw_outage_batch(..., rho_m=)` of the repository (it has a separate answer hazard for misuse intents), the vectorized replay of experiments D2 and H, and the B4 tuner.

## 1. Question

(H2) is the assumption that a misusing principal answers with the same probability as an honest delegator. Let the answer hazard of misuse intents be rho_m = lambda rho with lambda in {0, 1/4, 1/2, 1}. How much does the policy compiled under (H2) (Comp, the main-run table unchanged) lose in that world, and how far does it hold against B4 tuned on the results of that world?

## 2. Design

* World: E-fitted, F2 (the flow in which verify is active), main-run constants. For each lambda the draw has seed 8, 200,000 tuning payments first, then 5,663,400 evaluation payments (the method of D2, with rho_m = lambda rho; the lambda = 1 draw must equal the D2 draw, checked).
* Policies: Comp (main-run table, not recompiled), B4_lambda (tuned on the tuning split of that world, 161 points, existing tuner), Cutoff (pihat(0)), and Comp_lambda (the table compiled knowing lambda) if possible. If the compiler does not accept a misuse answer hazard, Comp_lambda is omitted and this is stated.
* Metrics: for each lambda, J(Comp) - J(B4_lambda) in bp with 95% interval, J(Comp) - J(Cutoff), misuse granted (Comp, B4_lambda, Cutoff), and the share of Comp's verify payments for which no answer arrived. At lambda = 0 every misuse query times out.

## 3. Reading rules (fixed in advance)

* Report the largest lambda at which Comp - B4_lambda falls below -10 bp ("even if (H2) fails down to lambda = x, the compiled policy stays inside the margin"). If it never falls below, say so. If it is inside the margin at lambda = 0 too, state that and check and state the reason (after a query times out, reject filters misuse).
* Report in which direction the share of misuse granted moves as lambda falls (expectation: misuse queries time out and are rejected, so the share falls; if the opposite happens, report the opposite).
* Whichever way the result goes: one summary table, which replaces the single lambda = 1/4 value.

## 4. Run time

4 values of lambda x (1 min draw + 2 min tuning + 4 replays of 40 s in total), about 15 minutes.

## Change log

* Comp_lambda omitted. `compile_outage` and `value_labels` take the answer hazard as the single `env.rho` and use it for both intents (the D of `window_AD` does not depend on the intent). There is no argument for a separate misuse answer hazard, so Comp_lambda was omitted as the spec allows.
* lambda = 0 passes `rho_m = 0.0`. The code accepts it (`retime_answers_geom` clips the hazard to 1e-9, so misuse answer times lie far beyond tau = 10; the minimum t_ans is checked in the check phase). The other fields and the t_ans of honest intents are the same as in the lambda = 1 draw (common random numbers).
* Reference policy B4_main (the D2 B4 of F2, the lambda = 1 tuning) was replayed for every lambda. "Share of verify payments without an answer" was counted as the share of verified payments with t_ans > min(tau, remaining ticks); payments that ended earlier through a chain failure inside the window are not separated. The reading rules were not changed.
* The run was interrupted by a container restart after lambda = 1 and 0.5 had finished, and `run` was started again. Existing per-lambda npz files and B4 plans are skipped, so only lambda = 0.25 and 0 ran anew (same seed, same results). In `analyze`, the lambda = 1 check gained an element-wise comparison of the block-sum arrays with the D2 npz file. The reading rules were not changed.
