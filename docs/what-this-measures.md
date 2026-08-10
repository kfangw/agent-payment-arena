# What this measures

This document states the scope of the instrument before any result is
reported. The limits below are not caveats attached after the fact; they
determine which questions the numbers in this repository can answer.

## What a run produces

A run pairs one agent with one accept policy and one scenario, and records
five quantities.

**Unauthorized spend.** Money that moved on a payment the delegator did not
authorize, measured against the scenario's ground truth rather than against
the gateway's judgment. A gateway that approves a payment it should have
refused is not aware of the error, which is the reason ground truth lives in
the scenario and is never visible to the subject.

**Benign tasks blocked.** Legitimate tasks the agent failed to complete
because a payment it needed was refused. Every attack in the catalog ships
with a benign twin for this reason. Without the twin, refusing every payment
is the optimal strategy and the measurement is worthless.

**Escalations.** Payments that reached the delegator through `ask`. Counted
separately from the delegator's answer, because asking has a cost whether or
not the answer is correct.

**Token cost and latency.** What the defense cost to run. A defense that
triples the token count per task is a different proposition from one that does
not, and reporting block rates without this hides the difference.

## How a figure is produced

Repetitions with an interval, never a single run. LLM agents are stochastic,
and a single number carries no information about whether the difference
between two policies is real.

Seeds fix scenario generation, traffic, and delegator responses, so a run can
be repeated. Model responses are recorded to cassettes keyed by model and
prompt hash; CI replays them and calls no provider.

Every reported figure carries the model identifier and the run date. Model
versions change and results change with them. Recording the version is what
lets a later reader tell a real regression from a model update.

## What this does not measure

**Real user behavior.** Scenarios are constructed. The distribution of tasks
here reflects what the catalog contains, not what agents are actually asked to
buy. A policy that scores well here has not been shown to score well on
production traffic.

**Attack coverage.** The catalog is a sample of a space with no known
boundary. A defense that blocks everything in it has been shown to block
everything in it, and nothing more. Absence of a successful attack in these
results is evidence about the catalog, not about the defense.

**On-chain settlement.** `FakeGateway` reproduces the contract, not the chain.
Reorgs, gas exhaustion, mempool behavior, and settlement failure modes are out
of scope for the default backend. Results that depend on settlement behavior
must be reproduced against `HttpGateway`.

**Whether a model is safe.** A model that resists the injections in this
catalog under this system prompt at this temperature has resisted exactly
that. The comparison the results support is between defenses, holding the
model fixed, and between models, holding the defense fixed. Neither supports a
claim about a model in general.

**Economic loss in the real sense.** Unauthorized spend is counted in test
tokens against constructed prices. It measures whether the authorization
boundary held, not what a breach would cost.
