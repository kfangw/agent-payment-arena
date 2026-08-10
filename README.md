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

Early. The scaffolding, the shared decision space, and the contract tests are
in place; the agents, the attacks, and the report are not. The results table
below is empty on purpose and will be filled by generated output rather than
by hand.

```bash
uv sync
uv run arena contract   # the decision space and refusal codes both backends are held to
uv run pytest
```

## Results

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

### Backends

The arena runs against either of two gateway backends.

`FakeGateway` is in-memory and is the default. It reproduces the x402
contract, not the gateway implementation, so a clean checkout runs with no
chain, no node, and no Go toolchain. `HttpGateway` talks to a running
[stablecoin-x402-gateway](https://github.com/kfangw/stablecoin-x402-gateway)
over HTTP.

A contract test runs the same scenarios against both and asserts they agree.
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

- [What this measures](docs/what-this-measures.md), including what it does not
- [Threat model](docs/threat-model.md)

## License

MIT
