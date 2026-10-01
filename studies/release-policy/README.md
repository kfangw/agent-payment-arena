# Release policies from declared rail constants

This directory holds the experiment scripts, the measured data and the summary
results of the release-policy study: when an x402 server should grant, wait,
reject, or ask the delegator, with the policy compiled from declared rail
constants. The model, the policy compiler and the simulators are the
repository's `src/arena/experiments/settlement/` package; the replication
runners and result files of the main run are `scripts/replication/` and
`results_replication/` at the repository root. Random seeds that look like
dates (for example 20260929) are ordinary integers and are kept because
changing them would change the results.

## Requirements

* Python 3.12 or later (the repository's requirement). Run with `PYTHONPATH` pointing at the repository's `src`; installing is not required.
* numpy 2.4.4 (`numpy>=2.4`). scipy (any recent version) is needed only by
  `scripts/run_D.py`, `scripts/analyze_D.py`, `scripts/run_E1.py`, `scripts/analyze_E1.py`,
  `scripts/run_H.py` and the scripts that import it (`scripts/run_C.py`, `scripts/run_D2.py`,
  `scripts/run_E.py`, `scripts/analyze_E.py`, `scripts/run_HF.py`, `scripts/run_F0.py`, `scripts/run_L.py`, `scripts/run_F4.py`, `scripts/analyze_F4.py`) and
  `scripts/replication/analysis/commitment_premium_flow.py` (repository root).
* pandas and scikit-learn (tested with scikit-learn 1.8.0) are needed only by `scripts/prep_F4.py`, which also needs the raw
  data file described in `data/card_fraud/README.md`; `scripts/run_F4.py` and `scripts/analyze_F4.py` need neither.
* No network access, API keys or GPU are needed to run anything except
  `data/halt/collect_batches.py`, which re-downloads the halt data and needs an
  Etherscan API key in the environment variable `ETHERSCAN_API_KEY` (the key is
  never stored; the collected output is already shipped).

All commands below are run from this directory (`studies/release-policy/`):

    export PYTHONPATH=$PWD/../../src

The main-run commands of step 1 and the diagnostics of step 4 are run from the
repository root with `PYTHONPATH=src`.

The scripts write under `results/<experiment>/`; set `ARENA_RESULTS=<dir>` to
move that tree (`scripts/paths.py`).

## Layout

| Path | Content |
|---|---|
| `../../src/arena/experiments/settlement/` | Model and experiment code: environments (`gate.py`), flows, chain and halting-rail simulators (`simulate.py`, `outage.py`), policy compilers and baselines B1 to B4 (`policies.py`, `watch.py`), single-cell runner (`run.py`), B4 tuning (`b4.py`, `grid.py`), aggregation and tests of the code (`aggregate.py`, `validate_*.py`) |
| `../../src/arena/experiments/{statistics,artifacts}.py` | Shared bootstrap and result-envelope helpers |
| `scripts/*.py` | The added experiments, one script each (see below); `paths.py` holds all locations |
| `docs/` | Design notes of the exposure-band experiment (`exposure_band.md`), of the regularized small-sample B4 experiment (`regularized_b4.md`), of the calibration and payment-size experiments (`calibration_pilot.md`, `exposure_law.md`) and the specs of the tuning-variation, no-channel, measured-sizes, harm-weight, submission-failure, misuse-answer and real-label-flow experiments (`spec-C.md`, `spec-D2.md`, `spec-E.md`, `spec-HF.md`, `spec-F0.md`, `spec-L.md`, `spec-F4.md`) |
| `../../scripts/replication/` | Runners and table generators for the security, transfer, two-threshold (B6) and commitment-premium diagnostics |
| `data/x402_amounts/` | Histogram of measured x402 payment amounts (`hist.json`), the revert share of facilitator transactions (`revert_share.json`) and a description of their sources (`README.md`) |
| `data/card_fraud/` | Experiment F4: description of the public card-fraud data set (`README.md`; the raw file `creditcard.csv` is not shipped) and the calibration report of the classifier (`F4_calibration.json`) |
| `data/halt/` | Measured halt law: `gaps.csv` (all gaps between successive batch postings above 60 s over 360 days), `gap-histogram.json`, `collect-meta.json`, the collector `collect_batches.py` and the fitted-rail cell `fitted_rail.py` |
| `results/` | Summary results of the added experiments (including `exposure_band/`), the archived block sums (`archived_blocks/`), the B4 reference files of the fitted-halt rail (`reference_b4/`) and the answer-count experiment (`response_records/`); `run_E1.py` writes to `regularized_b4/`; `run_C.py`, `run_D2.py` and `run_E.py` write to `tuning_variation/`, `no_channel/` and `measured_streams/`; `run_HF.py`, `run_F0.py` and `run_L.py` write to `harm_frontier/`, `submission_hazard/` and `misuse_answer/`; `run_F4.py` writes to `real_label_flow/`; compiled-table caches go to `cache/` (regenerated, not shipped) |
| `scripts/prep_F4.py`, `run_F4.py`, `analyze_F4.py` | Real-label flow F4: split, classifier and calibration of the public card-fraud data, the policy comparison on the flow built from it, and its summary |
| `scripts/make_lead_rows.py`, `make_halt_table.py`, `make_wrong_rows.py` | Generators of the main-result, measured-halt and wrong-constants table rows (write to `results/generated/`) |
| `../../results_replication/` | Fitted parameters, seeds, sample sizes and archived mean payoffs of the main run per cell (`fits/`), and the security, transfer, B6 and diagnostic result files |

The archived per-cell block sums of the main run (`results/archived_blocks/`, 18 MB) are shipped with their metadata stripped of commit hashes and dates. The per-policy block sums of the wrong-constants experiment (32 MB) are regenerated, not shipped.

## Reproducing the results

Sizes used everywhere: 200,000 tuning payments, 5,663,400 evaluation payments
(113,268 episodes of 50 payments; 5,664 episodes of 1,000 on the chain rail
E-slow), seeds 1 to 9 for the nine main cells (E-fast: 1-3, E-slow: 4-6,
E-outage: 7-9 for flows F1, F2, F3). Time estimates are wall-clock on 2 CPU
cores of one x86_64 machine.

### 1. Main run (rails table, main-result table)

For each rail and flow (`E-fast`, `E-slow`, `E-outage` by `F1`, `F2`, `F3`):

    python -m arena.experiments.settlement.run --env E-outage --flow F2 --n-eval 5663400 --n-tune 200000 --seed 8 --out results_main
    python -m arena.experiments.settlement.b4  --env E-outage --flow F2 --seed 8 --results results_main --out results_main
    python -m arena.experiments.settlement.grid --env E-outage --flow F2 --seed 8 --results results_main --out results_main
    python -m arena.experiments.settlement.aggregate --results results_main --out tables

About 2.6 h per cell on one core (the two cores run two cells at a time). The
constants of every rail are in `gate.py` (`envs_for`). `results_replication/fits/`
records the fitted parameters and the archived mean payoffs of each cell, so a
regenerated cell can be checked against them; the runners under
`scripts/replication/` do this check before they report anything.

### 2. Measured halt law and the fitted rail E-fitted (measured-halt table)

* `data/halt/gaps.csv` lists every gap above 60 s between successive
  batch-inbox postings; gaps above 5 minutes are read as halts (18 halts, mean
  22.8 min, 0.0792% stationary share over 359.99 days). The two fitted rates
  are `p01 = 18 / 517975.2` and `p10 = 1 / 22.8`, hard-coded in `gate.py` as
  environment `E-fitted` (E-outage with only these two constants replaced).
* Run E-fitted at main-run size with seeds 7, 8, 9 for F1, F2, F3:

      python -m arena.experiments.settlement.run --env E-fitted --flow F2 --n-eval 5663400 --n-tune 200000 --seed 8 --out results_fitted
      python -m arena.experiments.settlement.b4  --env E-fitted --flow F2 --seed 8 --results results_fitted --out results_fitted --b3-n 161 --name b4ref

  About 2.6 h per cell on one core. `results/reference_b4/b4ref_E-fitted_F*_mid_s*.json`
  are the B4 outputs of this step (3.5 MB each); the scripts below read them, so
  they are shipped and need not be regenerated.
* `data/halt/fitted_rail.py` runs the terminal-accounting diagnostic of
  `recovery.py` on the fitted constants at small size (minutes).
* `data/halt/collect_batches.py` re-downloads the postings (about 1,000 API
  requests, about 6 minutes at the free tier's rate limit); not needed to
  reproduce anything else.

### 3. Added experiments

Run them in this order; each later script reads files written by an earlier one.

| Script | What it does | Reads | Writes | Time on 2 cores |
|---|---|---|---|---|
| `scripts/wrong_constants.py --flow F2 --seed 8` (also F1/7, F3/9) | Policies compiled from wrong constants (declared halt law, and hazards and answer rate times a lognormal factor with scale 0.1 and 0.2, 20 replicates) replayed in the E-fitted world | nothing | `results/wrong_constants/wrong_E-fitted_F*_s*.json` | about 45 min per flow (one core each) |
| `scripts/r_cross.py run` then `analyze` | Mirror image: compile on the fitted halt law, replay in the declared E-outage world | `results/archived_blocks/blocks_E-outage_F*_mid_s*.npz` (see below) | `results/r_cross/summary.json` | about 10 min in total |
| `scripts/run_queue.py [--workers 2]` then `scripts/analyze.py` | Per-draw input-error experiments in the E-fitted world: joint hazard and answer-rate error (J), answer-rate-only and hazard-only (C), cost-constant error (K); 439 items, each one compiled policy replayed on the full evaluation draw | `results/wrong_constants/` (reproduction check), `results/reference_b4/` | `results/draw_injection/npz/`, `summary.json`, `summary.md` | about 77 s per item per core, about 5 h for the whole queue |
| `scripts/h_eslow.py repro`, `spec`, `run`, `analyze` | Hazard-input sensitivity on the slow rail E-slow x F2: all stage hazards multiplied by k in {0.1, 0.3, 0.5, 0.8, 0.9, 1.1, 1.2, 2.0} | `results/archived_blocks/blocks_E-slow_F2_mid_s5.npz` | `results/h_eslow/` | about 1 min (repro), 5 min (run) |
| `scripts/run_D.py all --workers 2` (or `plan`, `eval`, `repro` separately), then `scripts/analyze_D.py` | Loss of the compiled policy and of tuned B4 as a function of the number n of observed answers (n in {50, 130, 500, 5000}, 30 learning seeds), E-fitted x F2 | `results/wrong_constants/wrong_E-fitted_F2_s8.json`, `results/reference_b4/b4ref_E-fitted_F2_mid_s8.json` | `results/D_answers/` (shipped summary: `results/response_records/`) | 94 minutes for `all` on 2 cores (214 replayed policies) |
| `scripts/measure.py` | Compile time, table size and lookup latency of the compiled policy on E-fast, E-slow, E-outage, E-fitted | nothing | `results/compute_cost/result.json` | under 5 min |
| `scripts/xband.py --flow F1 --seed 7` (also F2/8, F3/9), then `scripts/analyze_xband.py` | B4 tuned separately in 3, 4 and 5 exposure bands (B4-band), compared with B4 and the compiled policy on the same draws, with security statistics (design notes: `docs/exposure_band.md`) | `results/reference_b4/`, `results/wrong_constants/wrong_E-fitted_F*_s*.json` | `results/exposure_band/runs/`, `summary.{json,md}` | about 6 min per flow (F1 and F2 in parallel), about 1.5 GB memory per flow |
| `scripts/run_E1.py repro`, then `all --workers 2` (`--dry-run`, `--tag smoke --n-eval 100000 --seeds 1,2` for checks; `analyze` re-runs the summary via `scripts/analyze_E1.py`) | Regularized small-sample B4 (coarse grid, shrinkage toward a neutral centre, bagging) against the compiled policy compiled from n observed answers, n in {130, 500}, 30 learning seeds, E-fitted x F2 (design notes: `docs/regularized_b4.md`) | `results/response_records/plan/` (or `results/D_answers/plan/` after step `run_D.py`), imports `scripts/run_D.py` and `scripts/xband.py` | `results/regularized_b4/` | about 35 min on 2 cores. `results/regularized_b4/` holds the exploratory run on the answer-records evaluation sample (seed 108), which experiment D already used; `results/regularized_b4/confirm109/` holds the confirmatory run (`--confirm`) on a fresh evaluation seed (109) and fresh learning draws (learning root 109) |

Shipped summaries: `results/r_cross/summary.json`,
`results/draw_injection/summary.{json,md}`,
`results/h_eslow/{summary,repro,spec_tables}.json`,
`results/compute_cost/result.json`,
`results/exposure_band/{summary.json,summary.md}`,
`results/regularized_b4/{summary.json,summary.md}` (exploratory run on the answer-records evaluation sample),
`results/regularized_b4/confirm109/{summary.json,summary.md}` (confirmatory run on a fresh evaluation seed and fresh learning draws).
`results/draw_injection/summary.json` covers the 277 finished items (the F2
items); the F1 and F3 joint-error items are not in it.
`results/response_records/` holds `summary.json`, `summary.md` and the 30 per-seed
plan files (`plan/seed_*.json`) of experiment D.

`results/archived_blocks/` holds the per-policy block sums of the main run
(keys `A2`, `B1`, `B3`, `B4_declared`, `B4_refined`, `block_counts`, plus a `meta`
string), for the three E-outage cells and E-slow x F2. Only the metadata was
edited (commit hashes and timestamps replaced by `redacted`); the arrays are
bit-identical to the originals. `r_cross.py` and `h_eslow.py` assert that their
own replay reproduces the `A2` array before they compute anything else.

Table rows (run after steps 1 to 3; the generators read `results/archived_blocks/`,
`results/reference_b4/`, `data/halt/` and the E-fitted settlement files
`settlement_E-fitted_F*_mid_s*.json` written by step 2 into `results/fitted_runs/`):

    python3 scripts/make_lead_rows.py
    python3 scripts/make_halt_table.py
    python3 scripts/make_wrong_rows.py

### 4. Diagnostics of `scripts/replication/`

    python3 scripts/replication/transfer_cell.py --help
    python3 scripts/replication/b6_cell.py --help
    python3 scripts/replication/adversarial_struct.py <draws> <seed>
    python3 scripts/replication/analysis/wait_interval_sweep.py [n] [seed]
    python3 scripts/replication/analysis/commitment_premium.py
    python3 scripts/replication/tables/table_rows.py results_replication <out_dir>

The runners rebuild every policy from `results_replication/fits/` and check the
replayed mean payoffs against the archived ones. Result files record the
SHA-256 of the module that produced them (`module_sha256`).

In the result files of this directory, recorded commit hashes and creation
timestamps were replaced by the string `redacted`. Content hashes of parameters
(`params_hash`, `base_hash`) are kept.


## Calibration re-estimated from the pilot

Design and result summary: `docs/calibration_pilot.md`; scripts `scripts/run_G.py`, `scripts/analyze_G.py`, `scripts/fastreplay_G.py`, `scripts/tests_G.py`, `scripts/indep_G.py`; results `results/calibration_pilot/`. Run from this directory with `python3 scripts/run_G.py all --workers 2` (about 100 minutes on two cores), then `python3 scripts/analyze_G.py`.


## Today's payment sizes

Design and result summary: `docs/exposure_law.md`; scripts `scripts/run_H.py`, `scripts/analyze_H.py`; results `results/exposure_law/`. Run from this directory with `python3 scripts/run_H.py run` (about 20 minutes on two cores), then `python3 scripts/analyze_H.py`.


## Tuning variation of B4

Design: `docs/spec-C.md`; script `scripts/run_C.py`; results `results/tuning_variation/` (`summary.json`, `plan/b4_F{1,2,3}_s{1..5}.json`). What it does: B4 is tuned on five independent learning samples of 200,000 payments (`default_rng([112, 700, s])`, s = 1 to 5) in the E-fitted world for each of F1, F2, F3, every B4_s is replayed on the main-run evaluation draw of the flow (5,663,400 payments) next to the compiled policy, and the difference Comp - B4_s is reported with its spread over s and with a two-way (block times tuning sample) bootstrap interval. Run from this directory with `python3 scripts/run_C.py run --workers 2` (about 18 minutes on two cores, resumable; `run_C.py smoke` runs F2 with 100,000 evaluation and 20,000 learning payments in a few minutes), then `python3 scripts/run_C.py analyze` re-creates the summary. If `results/no_channel/npz/` exists (after `run_D2.py`), Comp's block sums are checked against it.


## Compiled policy without a question channel

Design: `docs/spec-D2.md`; script `scripts/run_D2.py`; results `results/no_channel/` (`summary.json`, `checks.json`, `plan/b4main_F{1,2,3}.json`). What it does: the compiler is run without the verify action (`compile_outage(..., drop=(VERIFY,))`), giving Comp_nv, which is replayed on the main-run draws of F1, F2, F3 (E-fitted world) with Comp, the one-threshold Cutoff and the main-run B4 (retuned on the main run's tuning split); payoff differences, misuse release, the first-action shares of Comp_nv and the check against Cutoff at running arrivals are reported. Run from this directory with `python3 scripts/run_D2.py run` (about 10 minutes on two cores, resumable; `run_D2.py smoke` replays 100,000 payments of F2 and checks the vectorized replay against the repository's `replay_outage`; `run_D2.py analyze` re-creates the summary). `checks.json` is the output of the smoke run.


## Measured x402 payment sizes

Design: `docs/spec-E.md`; scripts `scripts/run_E.py`, `scripts/analyze_E.py`; data `data/x402_amounts/` (histogram and source description); results `results/measured_streams/` (`summary.json`, `summary.md`, `manifest.json`, `plan/b4_L{0,6,7}.json`). What it does: the exposure law of experiment H is replaced by two empirical laws taken from the on-chain amount histogram, L6 (all listed Base facilitators, last 90 days) and L7 (Coinbase facilitator, its last 90 active days); Comp, Cutoff, B4_L and B4_L0 are replayed on 5,663,400 payments of F2 per law. It reads `results/exposure_law/plan/b4_L0.json` (shipped) and imports `scripts/run_H.py`. Run from this directory with `python3 scripts/run_E.py run` (about 7 minutes on two cores; `run_E.py smoke` uses 100,000 evaluation payments and writes to `results/measured_streams/smoke/`), then `python3 scripts/analyze_E.py` (about 5 seconds; `--tag smoke --nb 100` for the smoke run).


## Harm-weight curve

Design: `docs/spec-HF.md`; script `scripts/run_HF.py`; results `results/harm_frontier/` (`summary.json`, `summary.md`, `plan/b4_h{0.25,0.5,1,2,4}.json`). What it does: the harm weight h of the payoff is set to 0.25, 0.5, 1, 2 and 4 (E-fitted world, F2, main-run constants and evaluation draw). For each h, the policy Comp_h is compiled with that weight and B4_h is tuned with the payoff at that weight on the 200,000-payment tuning split; both are replayed on the same 5,663,400 evaluation payments and scored at weight h and at weight 1. The summary reports J, the misuse released (share of misuse payments and dollars per payment), the honest payments released, the query share, and the position of the Comp_h curve against the B4_h points in the plane (misuse exposure, J at weight 1). At h = 1 the table, the B4 parameters and both J values are checked against experiment D2. Run from this directory with `python3 scripts/run_HF.py run` (about 15 minutes on two cores, resumable per h; `run_HF.py check` compares the vectorized replay with the repository's `replay_outage` on a 20,000-payment prefix; `run_HF.py smoke` uses 100,000 evaluation and 20,000 tuning payments and writes to `results/harm_frontier/smoke/`; `run_HF.py analyze` re-creates `summary.json` and `summary.md` from the per-h block sums, which `run` writes to `results/harm_frontier/npz/` and which are not shipped). It imports the helpers of `scripts/run_D2.py`; compiled tables go to `results/cache/`.


## Measured submission failure f0

Design: `docs/spec-F0.md`; script `scripts/run_F0.py`; data `data/x402_amounts/revert_share.json` (source description in `data/x402_amounts/README.md`); results `results/submission_hazard/` (`summary.json`, `summary.md`, `plan/b4m_F{1,2,3}.json`). Note: the revert share is a proxy for f0, not a measurement or an upper bound of it; "measured" in the design file and the result names refers to the on-chain query, not to f0. What it does: the stage-0 submission failure probability of E-fitted (declared 0.005) is replaced by the revert share of facilitator transactions on Base (window B, 0.01216147), all other constants unchanged. The policy compiled for the measured value (Comp_m), the main-run policy (Comp_d), B4 retuned in the measured world (B4_m) and the one-threshold Cutoff are replayed on the main-run evaluation draws of F1, F2, F3 (5,663,400 payments each); the summary gives the payoff differences in both directions, misuse granted, the first-action shares of Comp_m and the range of the payments for which Comp_m waits. The check is J(Comp_d) in the declared world against experiment D2. It reads `results/no_channel/plan/b4main_F*.json` and `results/no_channel/summary.json` (shipped). Run from this directory with `python3 scripts/run_F0.py run` (about 20 minutes on two cores, resumable per flow; `run_F0.py check` compares the vectorized replay with `replay_outage` on a prefix; `run_F0.py smoke` replays 100,000 payments of F2 and writes to `results/submission_hazard/smoke/`; `run_F0.py analyze` re-creates the summary from `results/submission_hazard/npz/`, which is not shipped).


## Misuse intents that answer less

Design: `docs/spec-L.md`; script `scripts/run_L.py`; results `results/misuse_answer/` (`summary.json`, `summary.md`, `plan/b4_lam{0p00,0p25,0p50,1p00}.json`). What it does: the answer hazard of misuse intents is set to lambda times the common answer hazard, lambda in {1, 0.5, 0.25, 0} (E-fitted world, F2, main-run constants and seed). The main-run policy Comp (not recompiled), B4 tuned on the tuning split of each world (B4_lambda), the reference B4 of experiment D2 and the Cutoff are replayed on the 5,663,400 evaluation payments of each lambda; the summary gives Comp - B4_lambda and Comp - Cutoff in bp, misuse granted, and the share of Comp's verify payments that get no answer (overall, misuse, honest). Comp_lambda is not built because the compiler takes one answer hazard for both intents. The lambda = 1 draw is checked against experiment D2 (J values, B4 parameters; block sums too if `results/no_channel/npz/` exists). Run from this directory with `python3 scripts/run_L.py run` (about 15 minutes on two cores, resumable per lambda; `run_L.py check` tests the draws (lambda = 1 equals the default draw, lambda = 0 pushes every misuse answer beyond the deadline) and the vectorized replay on a prefix; `run_L.py smoke` uses 100,000 evaluation payments and writes to `results/misuse_answer/smoke/`; `run_L.py analyze` re-creates the summary from `results/misuse_answer/npz/`, which is not shipped).

## Real-label flow

Design: `docs/spec-F4.md`; scripts `scripts/prep_F4.py`, `scripts/run_F4.py`, `scripts/analyze_F4.py`; data `data/card_fraud/` (description of the public data set and `F4_calibration.json`); results `results/real_label_flow/` (`summary.json`, `summary.md`, `plan/b4_F4.json`, `verify/recheck.json`). What it does: flow F4 is built from the public ULB card-fraud data (284,807 transactions, 492 frauds). The transactions are split by time (50/25/25 by the rule of the spec), a gradient-boosted classifier is trained on the first part, its scores are calibrated by isotonic regression on the second part, and the (score, label) pairs of the third part (71,203 rows, 94 frauds) are the flow: a payment takes a row at random, uses its calibrated score as the suspicion score and its actual label as the intent; exposure is the declared lognormal of the main run. The report gives the share of payments in the verify band and the calibration inside it, the fitted distortion zeta-hat with a row-bootstrap interval, the fixed-bin calibration table, and, on E-fitted with 500,000 evaluation payments (seed 111), Comp minus B4 tuned on this flow, Comp minus Cutoff, the value of the channel (Comp minus Comp without verify), and the misuse released, with a draw-level interval (paired block bootstrap) and a row-level interval (200 resamples of the evaluation rows, B4 not retuned). The reading rules were fixed before the run (spec section 6). Shipped result: the band holds 82 of 71,203 rows (0.115%), zeta-hat = 0.896 [0.779, 1.041], Comp - B4_F4 = +4.49 bp (draw interval [+3.34, +5.80]), channel +0.15 bp, and the four policies release the same 38.8% of misuse. The row-level interval of Comp - B4_F4 ([+4.44, +4.49]) is narrow because all repetitions use the same evaluation seed and only the rows are resampled; the draw-level interval is the honest width for that comparison.

The raw file `creditcard.csv` is not shipped (150 MB, public data; see `data/card_fraud/README.md`). Place it at `data/card_fraud/creditcard.csv`, then, from this directory:

    export PYTHONPATH=$PWD/../../src
    python3 scripts/prep_F4.py             (about 2 min; writes data/card_fraud/F4_scores.npz and F4_calibration.json)
    python3 scripts/run_F4.py smoke        (100,000 evaluation payments, 3 row resamples; checks the vectorized replay against replay_outage on a 20,000-payment prefix; writes to results/real_label_flow/smoke/)
    python3 scripts/run_F4.py run          (500,000 evaluation payments; resumable)
    python3 scripts/run_F4.py rowboot      (200 row resamples, about 7 s each; resumable; `--reps N` for fewer)
    python3 scripts/analyze_F4.py          (re-creates summary.json and summary.md; `--tag smoke --nb 100` for the smoke run)

`run_F4.py` reads the shipped `results/real_label_flow/plan/b4_F4.json` instead of retuning if it exists, and uses the main-run table from `results/cache/` (compiled on first use, as in `run_H.py`). `F4_scores.npz` and the evaluation outputs (`results/real_label_flow/npz/`) are regenerated, not shipped. `verify/recheck.json` is an independent recomputation from the raw file and the stored block sums.
