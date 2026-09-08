# Repeated settlement robustness evaluation

Run from the repository root with Python 3.12 and the locked environment. No LLM, network service, credentials, or blockchain transaction is used. All outcomes are synthetic exact expectations. Existing single-request and repeated-request experiments remain unchanged.

## Validation

```sh
uv run --no-sync pytest tests/test_settlement_learning.py tests/test_settlement_robustness.py -q
uv run --no-sync ruff check --no-cache --select F,E9,I src/arena/experiments/settlement_learning/robustness.py src/arena/experiments/settlement_learning/run_robustness.py tests/test_settlement_robustness.py
```

## Inventory, pilot, and full run

Choose a fresh output directory for each run. The following commands are separate choices; the full run includes both grid resolutions by default.

```sh
uv run --no-sync python -m arena.experiments.settlement_learning.run_robustness --output results/robustness-plan --plan-only
uv run --no-sync python -m arena.experiments.settlement_learning.run_robustness --output results/robustness-pilot --case-prefix uncertain-persistent-h1-matched --limit 1
uv run --no-sync python -m arena.experiments.settlement_learning.run_robustness --output results/robustness-main
```

The pilot measures one two-request case. Three-request cases and large numerical tie sets can take longer, so its time is not a guaranteed full-run bound. Each case prints elapsed seconds. Start with the pilot and inspect its manifest and result file before allocating a larger run.

Resume an interrupted run with the same selections and source files:

```sh
uv run --no-sync python -m arena.experiments.settlement_learning.run_robustness --output results/robustness-main --resume
```

The runner rejects changed source or design hashes, preserves completed cases, and never overwrites an unrelated output directory. A `--plan-only` directory can be executed using `--resume`. If a result file is corrupt, use a fresh output directory rather than editing it manually.

## Design and interpretation

The inventory contains 18 bases: two exposure schedules, three settlement chains, and three harm weights. Each has a correctly specified model, one-at-a-time input changes, and additive-accounting refit and fixed-policy evaluation. Identical no-hazard variants share tags. The deduplicated suite has 168 cases. There are seven model policies and six threshold variants at each of two grid resolutions, giving 19 rows per case.

B1 and B2 use loss and amount thresholds. B3 and B4 have fixed-prior and posterior variants; B4 can wait first. Tuning maximizes exact assumed-model reward across the whole relation. This supplies model knowledge equally to planner and comparators. It is not learning thresholds from held-out data. Every grid tuple is retained, but tuples with identical actions over all possible public risks share an evaluation. All numerical ties are stored in compressed files. The selected tuple has the largest computed reward; the tie reporting tolerance does not change selection.

Actual and assumed models remain distinct throughout evaluation. The policy's normal-response release decision uses assumed settlement survival. Labels update the policy's assumed posterior; the evaluator uses the actual posterior. Private outcomes never enter policy inputs. Accounting refits and held-policy scores are separate cases. Correct-response verification never releases misuse, so only the unverified grant payoff changes under additive accounting.

`design.json` records the complete models and grids before execution. `manifest.json` records source hashes, runtime versions, revision, and completion. Each case has relation reward, queries, expected normal counts, normal nonrelease, misuse release amount, unpaid release amount, oracle regret, exposure, selected parameters, and timing. `*-ties.json.gz` contains all numerically optimal grid tuples. Regret is against the actual-model optimum, even for a misspecified policy. Exact expectations have no Monte Carlo confidence intervals. Existing replay files are not new evidence for this suite.

The scope excludes changing relationship types, selective or wrong labels, relationship exit, run/halt states, and shrinking operational deadlines. Tests cover reference agreement, actual versus assumed probabilities, response-before-failure ordering, the normal-answer action under misspecification, additive losses, action removal, and grouped versus exhaustive grid search.
