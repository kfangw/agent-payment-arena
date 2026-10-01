# Exposure-band B4 (`scripts/xband.py`, `scripts/analyze_xband.py`)

Design notes, fixed before the full run.

## Question

B4 does not see the exposure v of a payment, although the optimal decision depends on v. A tuning rule that does see v is therefore a missing comparison. B4-band tunes the thresholds (k, a, b) separately in each exposure band. The question is whether the compiled policy stays within the 10 bp equivalence margin of this stronger baseline.

## Reference setting (matches the main result table)

* World E-fitted; flows F1, F2, F3 with seeds 7, 8, 9 (as in the main run).
* B4 is the refined-grid B4 of the `b4ref` files: 161 threshold grid points (12,880 pairs with a < b), k in {0, 1, 2, 4, 8, 16, 30}. The E-fitted rows of the main result table are the `b4ref` values.
* Compiled policy: the `true` block sums of the wrong-constants experiment (`results/wrong_constants/`). They reproduce the mean A2 minus B4 of `b4ref` exactly.
* 200,000 tuning payments, 5,663,400 evaluation payments (blocks of 50, 113,268 blocks). Draws follow the order of `b4.py` (tuning batch, then evaluation batch, from one `default_rng(seed)`), so all differences are paired.

## Definition of B4-band

* Bands are set from the tuning sample only: quantiles of the tuning exposures. Main specification: 4 bands (quartiles). Sensitivity: 3 and 5 bands. Band j holds e_{j-1} < v <= e_j.
* In each band the triple (k, a, b) of the same grid as B4 is chosen to maximize the total tuning gain over the band's tuning payments (hindsight gain with intent label, as in B4). Ties: smallest k first, then first in grid order (the B4 tie rule).
* Everything else is as in B4: the policy does not observe halts; when it asks, it follows the answer.
* Implementation: for each k and each terminal action (grant, reject, verify) the forced policy is replayed once on the tuning draws; every (a, b) is scored per band with cumulative sums over payments sorted by pi0. For evaluation, the three forced policies are replayed on the evaluation draws for every k chosen in any band configuration, and each payment takes the payoff of the action its band policy selects (same argument as the B4 tuning in `b4.py`).

## Outputs (per flow)

* Gain per payment of B4-band (3, 4, 5 bands), the per-band choice (k, a, b), tuning-sample gain.
* Compiled minus B4-band and B4-band minus B4, in bp of the declared mean exposure 49.50 USD, with 95% paired block-bootstrap intervals (blocks of 50, 10,000 resamples, `stats.boot_ci`) and a sign-flip p-value.
* Security: share of misuse payments released and released misuse exposure per payment (definition of `security_replay`), for B4 and for B4-band.
* Reproduction checks: one band must reproduce the `b4ref` (k, a, b) and tuning mean; the evaluation block sums built from the B4 choice must reproduce the `b4ref` block sums.

## Decision rule (fixed in advance)

* Main specification, 4 bands: if the 95% interval of compiled minus B4-band lies inside (-10, +10) bp, the compiled policy is equivalent to a tuning rule that sees exposure. If the lower end is below -10, the equivalence claim fails against this baseline.
* 3 and 5 bands are sensitivity results only.

## Running

    PYTHONPATH=src python3 scripts/xband.py --flow F1 --seed 7
    PYTHONPATH=src python3 scripts/xband.py --flow F2 --seed 8
    PYTHONPATH=src python3 scripts/xband.py --flow F3 --seed 9
    python3 scripts/analyze_xband.py

Per-flow output goes to `results/exposure_band/runs/` (`--out`), the forced-replay cache to `results/exposure_band/cache/` (`--cache`). Memory: about 1.5 GB per flow for the evaluation draws; run F1 and F2 in parallel on two cores and F3 afterwards. Measured wall time: about 6 minutes per flow on 2 cores (F1 and F2 in parallel, then F3). `--smoke` runs a slice and compares the one-band tuner with the repository tuner. Requires `results/reference_b4/` (shipped) and `results/wrong_constants/wrong_E-fitted_F*_s*.json` (from `scripts/wrong_constants.py`). The shipped summary is `results/exposure_band/{summary.json,summary.md}`.
