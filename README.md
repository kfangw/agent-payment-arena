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

The repository contains two related components at different stages.

- `src/arena` is the generic agent-payment evaluation environment. Its shared
  decision contract and command-line interface are implemented. The agents,
  attacks, gateway backends, and end-to-end report remain on the
  [roadmap](ROADMAP.md).
- `duel` is the implemented settlement-policy experiment. It includes the exact
  model, stochastic replay, policy families, paired statistical analysis,
  robustness experiments, and validation programs.

Install the project and run its fast checks with:

```bash
uv sync --all-extras --dev
uv run arena contract
uv run pytest -m "not live and not http_gateway"
```

## Settlement-policy experiment

The settlement-policy experiment compares `grant`, `reject`, `verify`, and `wait` policies
under payment-channel delay and failure. A run draws separate tuning and
evaluation samples, replays each policy on the same evaluation payments, and
writes a parameter-hashed JSON envelope. Generated results are ignored by
Git.

Run one environment-by-flow cell:

```bash
uv run python -m duel.run \
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
uv run python -m duel.aggregate --results results --out tables
```

The batch drivers in `scripts/` skip outputs that already exist and write one
log per process to `reports/_logs/`. Pass the desired worker count as the
first argument:

```bash
uv run python scripts/run_s1.py 8
```

The main validation programs can be run independently:

```bash
uv run python -m duel.validate_harness
uv run python -m duel.validate_outage
uv run python -m duel.validate_stats
uv run python -m duel.validate_run
```

## Repository layout

```text
src/arena/     Generic agent-payment contracts and CLI
duel/          Settlement model, simulation, policies, statistics, and reports
scripts/       Resumable batch definitions using shared experiment utilities
tests/         Fast CI regression tests
docs/          Scope, threat model, and generic arena design notes
```

The settlement-policy modules separate the model from the experiment plumbing:

- `duel.core` defines the exact decision model.
- `duel.simulate` and `duel.outage` generate and replay stochastic channels.
- `duel.policies` compiles and tunes comparison policies.
- `duel.run` executes one comparison cell.
- `duel.stats`, `duel.aggregate`, and `duel.report` perform inference and
  serialize results.
- `arena.experiments.runner` provides shared process, logging, and concurrency
  code for batch scripts and future experiment suites.

## Generic arena results

Not yet populated. When it is, this table is written by `arena report` and
carries the model identifier, the run date, the repetition count, and an
interval for every figure.

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

### Planned backends

The generic arena is designed to run against two gateway backends.

`FakeGateway` will be in-memory and the default. It will reproduce the x402
contract, not the gateway implementation, so a clean checkout runs with no
chain, no node, and no Go toolchain. `HttpGateway` will talk to a running
[stablecoin-x402-gateway](https://github.com/kfangw/stablecoin-x402-gateway)
over HTTP.

A contract test will run the same scenarios against both and assert that they agree.
That test is what makes a result produced against the fake backend worth
reading.

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
