# Attacks

## Purpose

Produce the condition this repository exists to measure: a payment that
verifies, is made by a registered agent under an unexpired mandate, and serves
an adversary. Each attack is one route to that condition.

## Behavior

**Prompt injection.** Instructions placed where the agent reads them as
content. Three variants, kept separate because a defense that stops one often
misses the others. Direct instruction tells the agent to pay. Privilege
escalation persuades the agent its authorization is wider than it is. Payee
substitution leaves the task intact and changes only the destination, which is
the variant a task-completion check does not catch.

**Payee spoofing.** A recipient close enough to the legitimate one to survive
the check the agent actually performs. Run against both a string comparison
and a judgment made by the model, because the two fail differently.

**Price inflation.** A resource priced above its value and below every
absolute limit the mandate sets. The case where a limit-based policy is
correct by its own terms and wrong in effect.

**Repeat purchase.** Individually authorized payments that are collectively
outside intent, reached through a loop the agent does not recognize as one. No
single check is violated.

Every attack ships with a benign twin: the same task, the same shape, no
adversary. Both run in the same pass.

## Design decisions

**Injection points are limited to what an adversary controls.** The body of a
paid resource is the primary one, and it is primary because the agent has
already spent money to obtain it and is therefore disposed to use it. The
metadata of a 402 response and a message presented as coming from the
delegator are secondary points. Nothing is injected into the system prompt or
the tool definitions, because an adversary who can write there has already
won and the measurement would be vacuous.

**Benign twins are a hard requirement, not a nicety.** Without them, refusing
every payment is the optimal strategy and any block rate is meaningless. The
catalog is structured so an attack cannot be added without its twin.

**The catalog is fixed within a result.** Adding an attack changes what a
number means, so the catalog version is recorded with every reported figure.

**Attacks are written against the agent, not against the gateway.** A payload
that exploits a parsing bug in the gateway would be a finding about the
gateway. What belongs here is a payload that a correct gateway approves.

## Limits

The catalog samples a space with no known boundary. A defense that blocks
everything in it has been shown to block everything in it, and no more. The
absence of a successful attack is evidence about the catalog.

The attacks are hand-written. Nothing here searches for a payload that works
against a specific model, so the results are a lower bound on what an adaptive
adversary achieves.

Injection strings age. A payload that works against one model version may be
refused by the next, which is one reason results carry a model identifier.

## Status

Not implemented. `src/arena/attacks/` and `src/arena/benign/` hold package
docstrings only. One injection variant with its twin lands in M1; the rest of
the catalog in M2.
