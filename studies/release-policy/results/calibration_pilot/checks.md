# Experiment G smoke checks (spec sec 9, 2026-09-30 14:44 to 14:55)

Code: the repository of this package with the E-fitted patch, Python 3.11.15, numpy 2.4.4, scipy 1.17.1. Records: `smoke/progress.log`, `smoke/checks.json`, `tests_G.json`, `repro/repro_D.json`, `repro/check_vec.json`.

| Check | Content | Numbers | Verdict |
|---|---|---|---|
| G0 | Reproduction of D, repository `replay_outage`, full seed 108, true q | J = 8.938881202421832 (summed as in analyze_D). One replay 80.1 s, draw 55 s, compile 4.1 s. Table hash b80233bd... = D's ref_hash | PASS |
| 9.1 | G code path (zeta = 1, zeta-hat = 1, true q, vectorized replay), full seed 108 | J = 8.938881202421832, difference 0. Replay 9.4 s | PASS |
| extra | Vectorized replay = repository replay | Full seed 108, pi_ref: per-payment maximum difference 0. On 300,000 payments (zeta, k-hat) = (1, 0), (2, +50), (0.5, -80): difference 0. 12 forced policies (k = 0, 1, 4, 30 x G/R/V) on 100,000 payments: difference 0. In the smoke evaluation, 4 direct replays (pi_ref, Comp_cal zeta = 2, B4 zeta = 1, 2): block-sum difference 0 | PASS |
| 9.2 | Comp_cal table hash at zeta = 1 = D's table hash for the same q-hat | of 7 smoke q-hat values, the 5 that overlap D's qmap are all equal | PASS |
| 9.3 | Estimator recovery, 100,000 synthetic payments (`default_rng([110, 900])`), 74,946 with an answer (15,069 misuse) | zeta = 0.7: zeta-hat = 0.6977 (k = -36, true -35.67), \|ln error\| = 0.0033. zeta = 1.4: zeta-hat = 1.3910 (k = 33, true 33.65), \|ln error\| = 0.0065. Criterion 0.02 | PASS |
| 9.4 | Information contract | Fields read by Comp_cal = {pi0, t_ans, theta}, number of theta indices = number of answers (1,497 / 2,000). Six deliberate violations (one unanswered payment, scalar index, slice of the first 100, the whole array, iteration, an estimator using labels of all payments) all raise AssertionError. Reading v fails the field-set assertion | PASS |
| 9.5 | Smoke run (100,000 evaluation payments, seeds 1 to 3, n = 130, 2000, zeta = 1, 2, 2 workers) | plan 3 seeds, 5 s each; B4_200k 278 s (two zeta); 41 evaluated policies (Comp 25, B4 13, forced 3) 0.2 min; 0 errors; ALLDONE; `smoke/summary.md` written | PASS |

Smoke numbers are not results. The replay time used in the run-time rule is the full-size replay of 9.1, 9.4 s (see the change log of the spec).
