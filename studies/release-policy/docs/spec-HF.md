# Experiment HF spec: the harm-weight curve (payoff against misuse released)

Fixed before any result was seen. Script: `scripts/run_HF.py`; results: `results/harm_frontier/` (`summary.json`, `summary.md`, `plan/`).

Motivation: the model says that the operator sets the amount of misuse released with one number, the harm weight h. Until this experiment only three values of h had been run (0.5, 1, 2) on the E-outage rail. The claim needs a curve. Code used: `compile_outage` (only `env.h` changes), the draw and vectorized replay of experiment D2 (`run_D2.draw_main`), and the B4 tuner of D2.

## 1. Question

Is it true that one number h sets how much misuse is allowed? When the policy is compiled for different h, what curve does it draw in the plane (payoff, misuse released), and how does it lie relative to the points of B4 tuned with the same h?

## 2. Design

* World: E-fitted, flow F2, main-run constants, main-run evaluation draw (seed 8, `run_D2.draw_main` unchanged). The draw does not depend on h, so it is made once.
* h in {0.25, 0.5, 1, 2, 4}. For each h: Comp_h (compiled with `env.h = h`) and B4_h (tuned with the payoff accounted at weight h: 161-point grid, tuning split of 200,000 payments).
* Metrics for each h: (a) J_h(Comp_h) - J_h(B4_h) in bp, with interval, both scored at weight h; (b) J_1(Comp_h), the payoff at the reference weight h = 1 of the policy compiled for h: the cost of choosing another h (a cost of the choice, not of a misspecification); (c) share of misuse payments granted and misuse exposure in dollars per input payment, for Comp_h and B4_h; (d) share of honest payments released (granted, or executed after an answer) and share of payments queried. The curve is drawn in the plane (misuse exposure in dollars per payment, J_1) with the five Comp_h points and the five B4_h points.
* Check: Comp_1 and B4_1 must equal the values of experiment D2 (J_comp 8.933994, B4 (0, 0.1125, 0.8)).

## 3. Reading rules (fixed in advance)

* If the misuse exposure decreases with h and J_1 decreases with h, write "one number h sets the amount of misuse allowed, and that amount is priced in a loss of J_1". If monotonicity breaks, report what happens.
* If the Comp_h curve lies inside the B4_h points (higher J_1 at equal misuse exposure, or lower misuse exposure at equal J_1), say so; if outside, say so. With five points the statement goes only as far as where the curves cross.
* Whichever way the result goes, the curve is reported with one table (one figure if possible).

## 4. Run time

5 compilations x 3 s, 5 B4 tunings x 2 min, 10 replays x 10 s, about 15 minutes.

## Change log

* Accounting of the replay. The harm weight enters the payoff in one place: -v h for a settled misuse grant (`replay_outage`; the vectorized replay is the same). The exercise rule `ex` of the verify window does not depend on h (checked: the arrays are identical for all five h). Each policy was therefore replayed twice, once at the accounting of h and once at h = 1 (direct replay instead of rescoring from counts: same numbers, easier to check). For every h and policy, the payoff difference of the two replays was checked to be exactly zero outside the misuse grants.
* Metric definitions. Misuse granted = share of misuse payments that received a grant; misuse exposure = sum of v over granted misuse payments divided by the number of input payments (as in D2). Honest released = share of honest payments that were granted or that received an answer after a query and were executed (failed settlements included). Queried = payments for which verify was played divided by input payments. "Executed after an answer" was measured with a copy of the vectorized verify window that only adds a record; its payoff equals exactly that of the replay without the record and that of the repository's `replay_outage` on the first 20,000 payments.
* Reading of the monotonicity rule. "J_1 decreases" was read as decreasing in h. Comp_1 is optimal at the accounting h = 1, so J_1(Comp_h) is expected to be largest at h = 1; in that case the summary does not use the rule sentence and reports the actual shape (largest at 1, falling on both sides). The rule itself was not changed.
* Reading of the position rule. The position of the curve relative to the points is measured by the linear interpolation of the five Comp_h points ordered by misuse exposure, evaluated at the misuse exposure of each B4_h (J_1 difference in bp, interval from the same bootstrap resamples). A B4_h point outside the exposure range of the curve is not interpolated; only its point values are given.
