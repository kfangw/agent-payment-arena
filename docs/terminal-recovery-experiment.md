# Terminal-accounting experiment

Run from the repository root with the locked environment (`uv sync --locked`).
This runner is separate from the historical `b5` command.

```bash
uv run python -m arena.experiments.settlement.recovery \
  --flow F2 --seed 380901 \
  --n-tune 2000 --n-eval 5000 --repeats 5 \
  --conditions normal outage --recovery 0 0.5 1 \
  --out results/recovery-F2-pilot
```

Use F1 and F3 with separate output paths to run the other flows. A seed identifies
the entire sequence of independent train/evaluation repeats. Reusing it across
recovery values pairs the draws. The default seed is distinct from historical
F1/F2/F3 seeds 7/8/9. Normal and outage use the same endpoint, stage hazards,
60-second decision interval, and 30-tick release cutoff; normal disables only
the running-to-halted transition. This is a controlled outage comparison, not
a reproduction of the original E-fast environment and not a time-step
convergence experiment.

## Alternative accounting

- Stage 0 means the authorization has not executed. Expiry before the first
  successful stage transition prevents payment.
- Stages 1 through N mean execution has occurred but FINAL has not been reached.
- Recovery 0 preserves the old full-loss-at-cutoff accounting.
- Recovery 1 permits an executed transfer backing an earlier resource release
  to complete after the cutoff, subject to the remaining stage failure hazards.
- Intermediate recovery is a conditional probability of retaining this eventual
  settlement opportunity, sampled using an independent saved uniform per payment.
  It is not a fractional refund amount.
- An uncommitted request still closes without release at the cutoff. The
  continuation payoff belongs only to a grant already made. A rejected request
  receives the historical zero refund-reference payoff.

The first successful stage transition as the execution event is an explicit
assumption. Eventual recovery from a halt is assumed (`p10 > 0`); the tail uses
the remaining independent stage uniforms, so there is no arbitrary simulation
tail horizon. There are no added post-release financing costs, penalties, refund
fees, or terminal finality-latency measurements. This sensitivity model is not
calibrated to a real chain and does not establish ERC-3009 deployment correctness.

## Policies and fitting

For every repeat, condition, and recovery value:

1. Draw independent tuning and evaluation samples, in independent simulated
   settlement episodes of `--block-size` payments.
2. Estimate the response rate on the tuning sample and solve A again under the
   specified recovery accounting. A uses 41 exposure bins by default and the
   existing 501 suspicion bins; it remains a numerical policy approximation.
3. Retune B3 suspicion thresholds and B4's watch length and thresholds using
   tuning payoff only. The default watch grid is 0 through N+1, inclusive.
4. Apply B5's frozen rule: reject a public halt at arrival, otherwise use that
   repeat's B3 thresholds. B5 does not add a separate fitted state-aware class.
5. Replay all policies against the same held-out paths, labels, and response times.

All policies use the simulator's same verification exercise semantics, as in
the historical experiment. This is not a comparison of independently deployed
verification protocols.

## Outputs and denominators

Each outcome matrix has one row per payment. `summary.json` records column names.
The metrics include payoff, resource-release indicator, misuse-release indicator
and exposure, unpaid released exposure, legitimate refusal, query indicator,
release time, successful settlement after the cutoff, and unexecuted expiry
encountered when tracing a grant. The last indicator is not an all-request
authorization-expiry rate.

Unqualified summary fields are means per input payment, including exposure in
dollars per input payment. Conditional misuse-grant and legitimate-refusal rates
use the number of misuse and legitimate payments respectively. Conditional mean
release time uses released payments; absent denominators produce JSON null.
The clock follows the historical tick-boundary convention: an answer checked in
the first verification tick can release at the current tick boundary, although
one verification waiting-cost tick has been charged. These are discretized
release times, not measured wall-clock latencies.

The runner saves tuning/evaluation draws, recovery uniforms, per-policy outcome
matrices, fitted thresholds/watch lengths, response calibration, resolved
environment parameters, seed, Git revision, Python/NumPy versions, source-file
hashes and copies, and file checksums. Existing output directories are rejected
to avoid overwriting a run. An interrupted directory can be inspected, but a
complete run must contain `summary.json` and `sha256.json`.

## Inference

Per-repeat intervals resample evaluation episodes and condition on the fitted
policies. Across-repeat intervals resample independent full tuning/evaluation
repeats and therefore include their combined variability. The latter are omitted
when fewer than five repeats exist; five is still only a pilot. Intervals are
unadjusted and exploratory. There are no automatic equivalence verdicts. Define
the final comparison family and multiplicity procedure before a publication run.

For a larger run, increase sample counts, threshold resolution, and repeats after
checking runtime and the number of actual outage episodes. Rare halted arrivals
will be poorly represented by the pilot. Full per-payment artifacts consume disk
space in proportion to samples, conditions, recovery values, and repeats.

## Halt duration

The cell constant is a mean halt of 60 minutes (`p10 = 1/60` at the 60-second
tick). `--halt-minutes M` sets the mean halt to `M` minutes. By default the halt
start rate `p01` is unchanged, so shorter halts also lower the stationary halt
share; `--keep-halt-share` rescales `p01` so that only the duration moves. The
override is recorded in `config.json` (`halt_minutes`, `keep_halt_share`, and
the resulting `base_environment`).

The analytic settlement probability for an arrival during a halt, as a function
of the mean halt duration, needs no sampling:

```bash
uv run python -m arena.experiments.settlement.halt_curve --recovery 0
uv run python -m arena.experiments.settlement.halt_curve --recovery 1 --minutes 10 30 60
```

## Halted-arrival subset and additive rescoring

`halt_subset` reads the saved outcomes of one or more runs and reports, per
condition and recovery value, the halted-arrival count, the compiled policy's
release rate among halted arrivals (all, legitimate, misuse), the per-halted-
payment and pooled A minus comparator differences with a block-bootstrap
interval, and the shift each policy takes under an additive loss convention
that charges `h*v` on a release that is both misuse and unpaid.

```bash
uv run python -m arena.experiments.settlement.halt_subset \
  --run results/halt-10 results/halt-60 results/recovery-F2-pilot \
  --conditions outage --recovery 0 1 --out results/halted-subset.json
```

## Quick execution check

```bash
uv run python -m arena.experiments.settlement.recovery \
  --n-tune 100 --n-eval 200 --repeats 2 --n-boot 40 \
  --n-v 3 --grid-points 5 --max-watch 2 --recovery 0 1 \
  --out results/recovery-smoke
uv run pytest tests/test_settlement_recovery.py tests/test_settlement_b5.py -q
```

The quick check deliberately uses coarse policy grids and tiny samples. Its
payoff values cannot be substituted for a full run.

## Rescoring a finished run under a different accounting

The runner fits and scores every policy under the same recovery value. To see
what a policy fitted under one accounting loses when the world follows another,
rescore the saved evaluation paths without refitting:

```bash
uv run python -m arena.experiments.settlement.misspec \
  --run results/recovery-F2-pilot \
  --believed 0 --actual 1 \
  --conditions outage \
  --out results/recovery-F2-misspec
```

`--believed` selects the saved fit (A recompiled from that setting's response
estimate, B3/B4 at their saved thresholds and watch length, B5 at its frozen
halt rule). `--actual` is the accounting applied when scoring the same paths and
recovery uniforms. Nothing is redrawn and nothing is refitted.

The module first replays the run at its own accounting and stops unless every
saved outcome matrix is reproduced exactly. Each row then records, per policy,
`misspecification_cost` = payoff of the run's matched fit at `--actual` minus
payoff of the `--believed` fit scored at `--actual`, plus the usual A minus B4
and A minus B5 comparisons. Both conventions for the verification exercise rule
are written (`--exercise-at policy` keeps it at the believed accounting, `world`
recomputes it at the actual one); `both` is the default.

```bash
uv run pytest tests/test_settlement_misspec.py -q
```
