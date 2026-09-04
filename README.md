# agent-payment-arena

[![CI](https://github.com/kfangw/agent-payment-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/kfangw/agent-payment-arena/actions/workflows/ci.yml)

An evaluation environment that measures what breaks when an LLM agent is given
the authority to spend money.

Agent payment standards settle the question of whether a signature is valid.
They do not settle the question this repository is about: a payment whose
signature is valid and whose purpose is outside what the delegator authorized.
An autonomous agent holding a payment credential can produce exactly that, and
whether it does is a measurable property of the agent, the defense it runs
under, and the policy the gateway applies. This repository is the instrument
that measures it.

## Status

The repository contains two related evaluation components.

- `src/arena` is the generic agent-payment evaluation environment. Its minimum
  offline suite includes signed mandates and payments, an in-memory gateway,
  baseline agents and policies, a prompt-injection scenario with a benign twin,
  repeated execution, scoring, and interval reports.
- `arena.experiments.settlement` is the implemented settlement-policy
  experiment. It contains the domain model and command-line entry points while
  reusing the generic experiment infrastructure under `src/arena/experiments`.

Install the project and run its fast checks with:

```bash
uv sync --all-extras --dev
uv run arena contract
uv run arena demo
uv run pytest -m "not live and not http_gateway"
```

Run a reproducible evaluation and generate report artifacts:

```bash
uv run arena run --suite minimum --repetitions 20 --seed 1 --out results/minimum.json
uv run arena run --suite attack-catalog --repetitions 20 --seed 1 \
  --out results/attack-catalog.json
uv run arena report results/minimum.json \
  --json-out reports/minimum.json \
  --markdown-out reports/minimum.md
```

Run the policy grid and write its complete interval estimates, Pareto points,
and a standalone SVG trade-off plot:

```bash
uv run arena frontier \
  --limits 25,50,100 \
  --ask-thresholds off,20,50 \
  --bond-thresholds off,50 \
  --repetitions 20 \
  --json-out reports/frontier.json \
  --svg-out reports/frontier.svg
```

Pass `--otlp-endpoint http://localhost:4318/v1/traces` to export run and
payment spans. This requires installing the `telemetry` extra.

Result files are never overwritten. The minimum suite is offline and uses no
provider key or network service. The demo first exercises fetch, payment,
delegator confirmation, and retry through the MCP-compatible tool surface,
then prints the repeated evaluation report.

The optional HTTP backend targets the pinned
[`stablecoin-x402-gateway`](https://github.com/kfangw/stablecoin-x402-gateway)
Compose stack. With that stack listening on port 8402, run the differential
contract check with:

```bash
ARENA_HTTP_GATEWAY_URL=http://localhost:8402 \
  uv run pytest -m http_gateway tests/test_gateway_differential.py
```

## Settlement-policy experiment

The settlement-policy experiment compares `grant`, `reject`, `verify`, and `wait` policies
under payment-channel delay and failure. A run draws separate tuning and
evaluation samples, replays each policy on the same evaluation payments, and
writes a parameter-hashed JSON envelope. Generated results are ignored by
Git.

Run one environment-by-flow cell:

```bash
uv run python -m arena.experiments.settlement.run \
  --env E-outage \
  --flow F2 \
  --cw mid \
  --n-tune 50000 \
  --n-eval 200000 \
  --seed 8 \
  --out results
```

Aggregate completed confirmatory cells:

```bash
uv run python -m arena.experiments.settlement.aggregate --results results --out tables
```

The batch drivers in `scripts/` declare independent jobs or ordered pipelines.
They skip outputs that already exist, pin numerical libraries to one thread per
process, and write one log per job to `reports/_logs/`. Pass the desired worker
count as the first argument:

```bash
uv run python scripts/run_s1.py 8
```

The main validation programs can be run independently:

```bash
uv run python -m arena.experiments.settlement.validate_harness
uv run python -m arena.experiments.settlement.validate_outage
uv run python -m arena.experiments.settlement.validate_stats
uv run python -m arena.experiments.settlement.validate_run
```

## Repository layout

```text
src/arena/     Generic agent-payment contracts and CLI
src/arena/experiments/settlement/  Settlement model, simulation, policies, and CLIs
scripts/       Resumable batch definitions using shared experiment utilities
tests/         Fast CI regression tests
docs/          Scope, threat model, and generic arena design notes
```

The settlement-policy modules separate the model from the experiment plumbing:

- `arena.experiments.settlement.core` defines the exact decision model.
- `arena.experiments.settlement.simulate` and
  `arena.experiments.settlement.outage` generate and replay stochastic channels.
- `arena.experiments.settlement.policies` compiles and tunes comparison policies.
- `arena.experiments.settlement.watch` contains the reusable
  settlement-observation policies and their threshold-and-horizon tuner.
- `arena.experiments.settlement.run` executes one comparison cell.
- `arena.experiments.settlement.aggregate` turns completed cells into tables.
- `arena.experiments.settlement.stats` and
  `arena.experiments.settlement.report` adapt the shared experiment utilities
  to the settlement result schema.
- `arena.experiments.statistics` implements paired block inference.
- `arena.experiments.artifacts` creates reproducible metadata envelopes and
  refuses to overwrite completed results.
- `arena.experiments.runner` provides shared process, pipeline, logging, and
  concurrency code for every batch script and future experiment suites.

Reusable infrastructure belongs under `arena.experiments`; experiment-specific
models and entry points belong under `arena.experiments.settlement`. This
boundary keeps new evaluation suites from depending on the current
experiment's names or internal modules.

## Generic arena results

`arena report` writes this comparison from a repeated result. Each primary
metric includes a bootstrap interval; raw per-scenario records remain in the
input JSON artifact.

| policy | agent | unauthorized spend | benign tasks blocked | escalations | tokens | latency |
| ------ | ----- | ------------------ | -------------------- | ----------- | ------ | ------- |

A single run of an LLM evaluation is a number, not a result. Nothing is
reported here without repetitions and an interval.

## Design

Two halves, kept apart on purpose. The *subject* is the agent, the gateway
backend, and the accept policy: everything that would exist in a real
deployment. The *instrument* is the scenario, the metrics, and the report:
everything that knows the ground truth of a run and therefore may not be
visible to the subject.

### Backends

The generic arena is designed to run against two gateway backends.

`FakeGateway` is in-memory and the default. It reproduces the x402
contract, not the gateway implementation, so a clean checkout runs with no
chain, no node, and no Go toolchain. `HttpGateway` talks to a running
[stablecoin-x402-gateway](https://github.com/kfangw/stablecoin-x402-gateway)
over HTTP.

The schemas and signatures are aligned to reference gateway revision
`ddd20fe3ec7c2109a006e1112bcac9fdeabf9b32`. A future differential test will
run the same cases against the HTTP backend and assert that both agree.

### Decision space

Both backends and both repositories share one set of outcomes for an accept
policy: `approve`, `reject`, `defer`, `ask`, `bond`. Sharing it is what lets a
policy written for the gateway be evaluated here, and a finding here be
carried back as a change there.

Only `approve` moves money. The other four are different kinds of refusal, and
separating them is most of the measurement: a policy that answers `reject` to
everything scores perfectly against attacks and cannot be deployed.

### What keeps the numbers honest

Every attack ships with a benign twin, so over-blocking is priced. The
delegator is modeled as a person who answers late, answers wrongly, and tires
of being asked, so escalation is priced. LLM responses are recorded once and
replayed from cassettes, so CI is free and deterministic and a reported figure
can be reproduced.

## Documentation

Scope first, then design, then plan.

- [What this measures](docs/what-this-measures.md), including what it does not
- [Threat model](docs/threat-model.md): what counts as compromise, and what is out of scope
- [Design notes](docs/design/README.md): one note per part, each stating what is built and what is not
- [Roadmap](ROADMAP.md)

### Design notes

- [Contract](docs/design/contract.md): what both gateway backends are held to, stated apart from either one
- [Backends](docs/design/backends.md): the in-memory and HTTP gateways, and the differential test between them
- [Agents](docs/design/agents.md): the four subjects and the defense each represents
- [Attacks](docs/design/attacks.md): the catalog, its benign twins, and where an adversary may write
- [Delegator](docs/design/delegator.md): the person the agent escalates to, modeled rather than assumed
- [Metrics](docs/design/metrics.md): the five quantities, and why none is reported alone
- [Reproducibility](docs/design/reproducibility.md): cassettes, seeds, repetitions, version records

## License

MIT
