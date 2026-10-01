# Experiment D2 spec: the compiled policy without a verify action

Fixed before any result was seen. Script: `scripts/run_D2.py`; results: `results/no_channel/` (`summary.json`, `checks.json`, `plan/`).

Motivation: today's x402 has no question channel. The main table needs a row for the compiled policy in that setting.

## 1. Question

In today's x402, which has no question channel, what is the compiled policy, and how do its payoff and its misuse release differ from the policy with the channel and from B4? The numbers of the three main-rail flows give one added row of the main-result table.

## 2. Design

* World: E-fitted, F1, F2, F3, main-run constants, evaluation set as in the main run (see the change log for the seeds).
* Policy Comp_nv: the table compiled with the same constants but without the verify action (the BUILD option that excludes verify if there is one; otherwise remove verify from the compiler's action set and redo the backward induction). wait, grant and reject stay. The halt regime stays.
* Comparison: Comp (with channel), B4 (main-run value), Cutoff (the threshold pihat(0) = 0.255 at arrival, no waiting).
* Metrics: J(Comp_nv) - J(B4) and J(Comp) - J(Comp_nv) in bp, with paired block bootstrap 95% intervals. Misuse pass rate and misuse exposure in dollars. The first-action shares of Comp_nv (does wait occur only at halt arrivals?).
* Check: do Comp_nv and Cutoff give the same action at running arrivals (Theorem 3.2(d): with verify inactive and c_w >= f_i, the decision is made at arrival)? If not, state the number and the reason.

## 3. Reading rules (fixed in advance)

* Whatever the result, the main-result table gets a "Comp, no channel" row (Comp_nv - B4 for F1, F2, F3).
* If the check confirms the sentence of "Today's payment sizes", "Without a question channel ... that cutoff at every amount outside halts", it stays; if Comp_nv and Cutoff differ at running arrivals, the sentence is corrected to the numbers.
* In F2, Comp - Comp_nv is the value of the channel. State that value in one phrase of the text.

## 4. Run time

3 compilations x 3 s + draw 3 min + 6 replays x 10 s, about 5 min.

## Change log

* Evaluation set. The first version of this spec said "the main run's seed 8". The main-run data use different seeds per flow (F1 7, F2 8, F3 9; `SEEDS` in `scripts/run_queue.py`). The main run draws with `default_rng(seed)` first the 200,000 tuning payments and then the 5,663,400 evaluation payments (the repro step of `scripts/run_D.py` reproduces the main-run block sums exactly this way). This design was used. Check: B4 retuned on the same tuning split, minus Comp, equals the main-run values of `lead_cells.json` (3.8 [3.5, 4.1], 4.3 [3.7, 4.8], 1.4 [1.0, 1.8]) to the first decimal in all three flows, and the F2 B4 (0, 0.1125, 0.8) and its tuning mean equal the repro record of experiment D.
* B4. For J(Comp_nv) - J(B4) in section 2, the main-run B4 was retuned on its tuning split with the existing tuner (161-point grid) and replayed on the same draw (check above). The original block-sum files of the main run are not part of this package.
* Comp_nv was built with `compile_outage(..., drop=(VERIFY,))` (an option of the repository; the continuation is recomputed with the masked action set). The Cutoff threshold is pihat(0) = 0.2551795 (`run_H.pihat0`), not the rounded 0.255.
* Check result. The 2,101 mismatches at F2 running arrivals all have pi0 in [0.25500, 0.25518); they come from the rounding of pi to the table's grid (0.002).
