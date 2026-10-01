# Regularized small-sample B4 (`scripts/run_E1.py`, `scripts/analyze_E1.py`)

Design notes, fixed before any run of this experiment (E1). Constants below are the ones coded in `run_E1.py`; they were not changed after results were seen. If a constant is changed later, the affected run must be labelled exploratory.

## 0. Status of the run

* The evaluation set is the one of experiment D (`scripts/run_D.py`, seed 108). D already produced results for Comp_n and B4_n on it, so E1 on seed 108 is a pre-specified exploratory follow-up, and its summary says so.
* For a confirmatory status, `--confirm` uses evaluation seed 109 and learning root 109 (`default_rng([109, 500, s])`), with the same design.

## 1. Question

B4_n of experiment D is an unregularized grid search, so a large loss at small n can be attributed to a weak baseline. With the same n payments and the same information (hindsight counterfactual payoffs and intent labels), does a regularized tuner close the gap to Comp_n? In particular, at n = 130, does it come within the 10 bp margin?

## 2. World

As in D: environment E-fitted (`envs_for("mid")["E-fitted"]`), flow F2, true answer rate q = 0.75 (tau = 10 ticks, rho = 0.129449), all other constants nominal. Payoffs are always computed in the true world.

## 3. Evaluation set

* Default: `draw_outage_batch(world, F2, 5,663,400, default_rng(108), payments_per_episode=50)`, 113,268 blocks. The payoff J of the reference policy pi_ref (compiled at q = 0.75) must equal D's value 8.938881202421832 per payment (checked by `analyze_E1.py`).
* With `--confirm`: seed 109, otherwise identical.
* All policies are applied to the same evaluation draws (paired).

## 4. Learning sets

* Learning seeds s = 1..30 of D, `default_rng([108, 500, s])`, 5,000 payments each; n in {130, 500} uses the first n payments (prefix). 30 repetitions is fixed.
* n = 130 and n = 500 are dependent within a seed; the two-way bootstrap uses the same seed resample for both.
* The plan phase checks K, q-hat and R0 (= D's B4_n: k, a, b and tuning mean) bit for bit against D's `plan/seed_XX.json` for every seed and stops on a mismatch (`--no-d-check` overrides).

## 5. Tuners

All tuners use the policy family of D's B4: observe k ticks, then grant if pi0 < a, verify if a <= pi0 <= b, reject if pi0 > b. The tuning objective is the mean payoff J_n(k, a, b) over the n payments; the forced payoff of each (k, action) is replayed once on the 5,000 learning payments and the first n are used.

| Name | Role | Candidates | Selection rule |
|---|---|---|---|
| R0 `R0_b4` | D's B4_n (unregularized), reproduction reference | k in {0,1,2,4,8,16,30}, 161-point grid, 12,880 pairs a < b | existing `tune_watch_policy`; first maximum (k ascending, then a, then b) |
| R1 `R1_coarse` | primary: fewer candidates | k in {0,1,2,4}, 21-point grid (`suspicion_grid(21)`, 210 pairs) | maximum J_n; candidates within 1e-9 USD per payment of the best are tied; ties go to the smallest k, then the pair closest to the centre c, then the smaller a, then the smaller b |
| R2 `R2_shrink` | primary: shrinkage | same as R0 | maximize J_n(k,a,b) - lambda_n * ||(a,b) - c|| / d_max; ties as in R1; c_lambda = 1 |
| R3 `R3_bag` | primary: bagging | same as R0 | on each of B = 20 resamples (with replacement) of the n payments apply R0's rule; k by majority (ties: smaller k); a and b are the means of the 20 picks, rounded to the 161-point values (exact midpoint: lower value) |
| `R2_shrink_c0.5`, `R2_shrink_c2` | sensitivity (not used for the verdict) | same as R0 | R2 with c_lambda = 0.5 and 2 |
| `Rc_centre` | reference (no tuning) | - | k = 0 and the grid pair closest to c, (0.33125, 0.66875); the limit of R2 as c_lambda grows |

### 5.1 Centre c = (1/3, 2/3)

c enters only the tie rule of R1 and the penalty of R2. It is the centroid of the valid region {0 <= a < b <= 1}, i.e. the mean of a uniform prior on (a, b); it is determined by the grid geometry alone and does not look at any payoff. The optimum found by B4 on 200,000 payments is deliberately not used, because it comes from hindsight results on a large sample and would leak into a small-sample tuner. (0.25, 0.75) was considered and rejected, since it is closer to that optimum than the centroid is.

### 5.2 Penalty strength of R2

* lambda_n = c_lambda * sigma_n / sqrt(n), with c_lambda = 1 primary. The distance is divided by d_max (the largest distance from c among grid pairs), so that the pair farthest from c is penalized by one standard error of the tuning mean. It shrinks as 1/sqrt(n).
* sigma_n is the sample standard deviation (ddof = 1) of the per-payment payoff of the centre policy (k = 0, pair nearest c) on the same n payments. No evaluation data are used. sigma_centre and lambda_ are recorded per seed and n in the plan json.
* No penalty on k (smaller k only breaks ties).
* The dependence on strength is shown by the sensitivity rows (c_lambda = 0.5, 2) and the limit `Rc_centre`.

### 5.3 R1's k restriction and R3's resampling

* R1 restricts k to {0,1,2,4}; R2 and R3 use the whole k grid. The verdict uses any one of the three (section 9).
* R3 resamples payments (`default_rng([learn_root, 600, s, n])`); n = 130 is only 2.6 episodes, so episode-level resampling is not meaningful. Dependence inside an episode is ignored.
* If a >= b after rounding, b moves up one grid point (or a moves down one point if b is already the top point).

## 6. Information contract

| Policy | What it reads for learning | What it does not know |
|---|---|---|
| Comp_n | Only the indicator "answer arrived within tau" of the first n pilot payments (checked by `FieldGuard`: the fields read are exactly `{t_ans}`); the compiler receives q-hat only. Prior knowledge: the rail model and all constants except q | payoffs, intent theta, v, pi0 |
| R0, R1, R2, R3 | For each of the n payments, the forced counterfactual payoff of every (k, action) (a hindsight replay, so it contains intent label, settlement outcome and answer time) and pi0. R2 also computes sigma_n from these payoffs. R3 uses resamples of the same n payments | the model and its constants; the evaluation data |
| `Rc_centre` | nothing | everything |
| all | No policy reads the evaluation data. Grids, c, c_lambda, B, tie tolerance and rng tags are fixed here | |

The B4 side receives hindsight counterfactual payoffs and therefore knows more than Comp in information terms, while Comp knows the model. E1 compares the two methods as wholes under the stated information, not learning efficiency in isolation.

## 7. Evaluation method

* pi_ref (q = 0.75) and every distinct Comp q-hat (up to 37 policies) are compiled and replayed once each. The SHA-256 of the compiled table is compared with D's `qmap.json` when that file exists.
* Tuner policies (k, a, b) are not replayed; for every selected k the three forced policies (grant, reject, verify) are replayed once on the evaluation set and each payment takes the payoff selected by pi0 (same argument as `xband.select`; B4 does not look at the rail state, so the per-payment payoff equals one of the forced payoffs).
* Replays use `xband.replay_release`, which also records whether the resource was released.
* Automatic checks:
  1. pi_ref: payoff of `replay_release` against the repository `replay_outage`, maximum difference <= 1e-9, otherwise stop. If `replay_release` fails for a compiled policy, that policy's payoff comes from `replay_outage` and the released statistics are left empty (recorded in the manifest).
  2. Selection against direct replay: one tuner policy (three in a smoke run) is replayed directly as `OutageWatchBandPolicy`; maximum block-sum difference <= 1e-9.
  3. Reproduction of D (seed 108, full evaluation, 30 seeds): pi_ref J difference <= 1e-9; per-seed L(Comp_n) and L(R0) agree with D's summary (L and L_B4) within 1e-6 bp.

## 8. Metrics (fixed in advance)

Unit: bp of the mean exposure 49.50 USD. T is a tuner.

* L_T(s, n) = J(pi_ref) - J(T_n); L_Comp(s, n) = J(pi_ref) - J(Comp_n).
* Delta_T(s, n) = J(Comp_n) - J(T_n) = L_T - L_Comp. Positive means Comp is better (same sign as D).
* Overfitting: J_tune(T_n) - J(T_n), where J_tune is the unpenalized tuning mean on the n payments; and J(pi_ref) - J_tune.
* Misuse: share of misuse (theta = 1) payments released, released misuse exposure per payment, release rate, share of legitimate payments refused; averaged over the 30 seeds.
* Summary per T and n: mean, median, minimum, maximum over the 30 seeds, number of seeds with Delta <= 10, distribution of chosen k, mean of a and b, number of distinct (k, a, b).
* Intervals: same bootstrap as `analyze_D.py` (2,000 resamples, seed 20260929, the same block resample for all policies, seed resample after the block resample). CI_block (learning seeds fixed) and CI_2way (blocks and seeds together, the main interval).
* No tests; the verdict is the point-estimate rule of section 9.

## 9. Decision rule (fixed in advance)

* The primary tuners are R1, R2 (c_lambda = 1) and R3.
* If at n = 130 any primary tuner has a 30-seed mean Delta <= 10 bp, the claim "small-sample B4 loses much more than Comp" is replaced by "a regularized tuner comes within 10 bp of Comp". A CI_2way upper end above 10 does not prevent this reading if the mean is <= 10.
* If all three have mean Delta > 10 bp, the statement stands, and a robustness paragraph reports that the three regularized tuners still trail by x to y bp at n = 130 (labelled exploratory when the evaluation seed is 108).
* Because the rule is "any of three", it is lenient against the claim; no multiplicity correction is applied.
* n = 500, the sensitivity rows and `Rc_centre` are descriptive only.
* Claims not made: generalization to regularization methods as a class, "no small-sample tuner can match Comp", any advantage in pure learning efficiency. The comparison is limited to the tuners above, this world and this flow.

## 10. Compute budget

About 35 minutes on 2 workers (worst case about 50 minutes if large k are chosen): plan 3 to 5 minutes; evaluation-set draw about 1 minute; compiling and replaying up to 37 Comp policies about 72 s each; forced replays for the selected k; selection and checks 3 to 4 minutes; analysis 2 to 3 minutes. Memory: about 1.2 to 1.5 GB for the evaluation set, about 150 MB per selected k for the forced arrays, about 100 MB per worker. Disk: forced-array cache about 150 MB per k, npz files a few hundred MB. Smoke run (first 100,000 evaluation payments, seeds 1 and 2): 2 to 3 minutes; `repro`: about 1 minute.

## 11. Checks before the full run

1. `--dry-run`: prints configuration and plan size; does not import arena.
2. `repro`: plan for seed 1 with n in {50, 130, 500, 5000}; K, q-hat, R0 (k, a, b) and tuning mean must equal D's `plan/seed_01.json`, and the vectorized scorer must pick the same first maximum as `tune_watch_policy`. No evaluation data are drawn.
3. Smoke run: `--tag smoke --n-eval 100000 --seeds 1,2`; all Comp replays are compared with `replay_outage`, and selection against direct replay is checked for three policies. Smoke numbers are not used.
4. Full run after the three checks pass.

## 12. Out of scope

* A third baseline tuned only on the outcomes of released payments (selective labels) has a different information condition and is a separate experiment.
* Grid-resolution convergence (for example 641 points) and repeated tuning splits of the 200,000-payment B4.

## Running

    export PYTHONPATH=$PWD/src
    python3 scripts/run_E1.py --dry-run
    python3 scripts/run_E1.py repro
    python3 scripts/run_E1.py all --tag smoke --n-eval 100000 --seeds 1,2 --workers 2
    python3 scripts/run_E1.py all --workers 2
    python3 scripts/run_E1.py analyze            # or scripts/analyze_E1.py

Output: `results/regularized_b4/` (`summary.md`, `summary.json`, `plan/seed_01..30.json`, `manifest.json`, `E1_results.npz`, `progress.log`; large intermediate files in `npz/` and `cache/`). An interrupted run is resumed by repeating the same command. `--confirm` writes to `results/regularized_b4/confirm109/`. Requires scipy (via `run_D.py` and `analyze_D.py`).
