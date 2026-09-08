# Repeated release decisions

This module extends the stage-chain release calculation across a finite sequence of related requests. The declared latent type stays fixed. Only a timely binary intent reply updates the public label counts. Settlement observations, refusal and timeout do not reveal intent. No LLM, wallet, live payment service or API key is used.

The implementation is a research reference, not a deployed intent-authentication service. It assumes reliable request identity and authentic, accurate replies. It supports independent finite request distributions or a fixed schedule, a full geometric reply window, and the mutually exclusive unpaid-loss accounting. Relation exit, correlated request settings, reply errors and operational deadline residues are outside this version.

## Components

- `model.py`: immutable settings and public decision inputs.
- `solver.py`: reuse of `settlement.core` execution and delay values, actual observation kernel, count-based posterior and cached exact backward recursion. There is no belief grid or interpolation.
- `simulate.py`: relation replay with named random streams shared across policies. Private realized harm never enters `PublicState`.
- `run.py`: strict configuration loading, exact policy evaluation, optional replay, source hashes and output files.
- `tests/test_settlement_learning.py`: reference-action regression, independent tick-tree recursion, observation-contract and seeded-replay checks.

The policy names are `optimal`, `fixed` (fixed prior single-request rule), `myopic` (updated prior single-request rule), `query_first` (query the first request, then use the updated single-request rule), and `q_assumed` (plan as if replies were not censored by settlement). All policy rewards are evaluated under the same actual kernel. Each policy observes replies from its own queries. The exact evaluator may condition on observed history even when the fixed policy deliberately ignores it.

## Run locally

From the repository root, prepare dependencies if needed:

```sh
uv sync --group dev
```

Run deterministic verification first:

```sh
uv run --no-sync pytest tests/test_settlement_learning.py -q
uv run --no-sync python -m arena.experiments.settlement_learning.run \
  --config configs/settlement_learning/uncertain.json \
  --output artifacts/settlement-learning/exact-01 --relations 0 --split e0
```

Then run a small pilot separately:

```sh
uv run --no-sync python -m arena.experiments.settlement_learning.run \
  --config configs/settlement_learning/pilot.json \
  --output artifacts/settlement-learning/pilot-01 --relations 200 --seed 731 --split pilot
```

Output directories must be new. The exact check is a numerical calculation even when `--relations 0`; it is not just configuration validation. The tests include tiny synthetic replays. Use the `evaluation` split only after fixing the evaluation protocol; it has independent random streams from `pilot` and `e0`.

For the supplied two-request uncertain setting, algebra gives exact expected total rewards of 0.399849 for `myopic`, 0.404678985651 for `query_first`, and 0.405126060751 for `optimal`. The optimal initial action is waiting, not immediate querying. These are reference expectations, not reported execution results. Accept action-value and independent-tree agreement within the tolerances in the tests. Do not use pilot sampling error to excuse a deterministic test failure.

## Result files

- `manifest.json`: declared setting, split, seed, source hashes, revision, dirty flag and completion state.
- `summary.json`: exact values, sample means, reward standard errors and paired differences by relation.
- `public-events.jsonl`: decisions and eligible labels, without latent intent or reward.
- `private-metrics.jsonl`: evaluator-only relation rewards and loss metrics. Never feed this file to a policy.

A completed manifest means the runner reached its end; it does not establish statistical equivalence or validate all scientific assumptions. A `running` manifest without a summary indicates an incomplete run. Keep it for diagnosis. A 200-relation pilot checks execution and variance, not a small performance advantage. Exact and replay values are separate outputs. Rates of normal refusal must use the count of normal requests as the denominator.

The initial command-line solver caps relations at 12 requests each. The settings and memoized solver are intended for small finite experiments. Name collisions, signature verification, concurrency and production log ingestion require separate integration work.

## Verification status

Syntax parsing and a selected static lint profile were checked during preparation. Numerical tests, exact checks and pilot runs are prepared for separate execution. No test pass or measured performance is claimed by this README.
