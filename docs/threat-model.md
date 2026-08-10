# Threat model

## The setting

A delegator authorizes an agent to spend, within a scope, by signing a
mandate. The agent works autonomously against resources it has not seen
before. A gateway guards each paid resource, answers an unpaid request with
402, and applies an accept policy to the payment that follows.

Three parties are honest by assumption: the delegator, the gateway operator,
and the facilitator that settles. The agent is honest but manipulable, which
is the distinction this repository is built on. Adversaries control the
content of paid resources, the metadata in a 402 response, and the identity
they present as payee.

## What is in scope

**A valid signature with the wrong purpose.** The payment verifies, the payer
is registered, the mandate is present and unexpired, and the payment still
serves an adversary rather than the delegator. Existing standards cover the
first four conditions and say nothing about the fifth.

Four families produce that condition.

**Prompt injection.** Instructions placed where the agent will read them as
content: the body of a resource it paid for, the metadata of a 402 response, a
message that appears to come from the delegator. Three variants are treated
separately, because defenses that stop one often miss the others. Direct
instruction tells the agent to pay. Privilege escalation persuades the agent
that its authorization is wider than it is. Payee substitution leaves the task
intact and changes only where the money goes.

**Payee spoofing.** A recipient that resembles the legitimate one closely
enough to survive whatever check the agent performs, whether that check is a
string comparison or a judgment made by the model.

**Price inflation.** A resource priced above its value but below any absolute
limit the mandate sets. This is the case where a limit-based policy is
correct by its own terms and wrong in effect.

**Repeat purchase.** Individually authorized payments that are collectively
outside the delegator's intent, reached through a loop the agent does not
recognize as one. The agent that fails here does not violate any single check.

## What is out of scope

Key compromise, a malicious delegator, a compromised gateway operator, and
chain-level attacks such as reorgs or censorship. These are real and are
addressed by other mechanisms. Including them here would mean measuring the
mechanism that fails first rather than the one under study.

Denial of service against the gateway is also out of scope. Availability is a
property of the deployment, not of the authorization boundary.

## Where the boundary is drawn

An agent is treated as compromised when it initiates a payment the delegator
would not have authorized had they been asked, whether or not the gateway
approved it. Gateway approval is a separate measurement: it records whether
the policy caught what the agent got wrong.

Separating the two is what makes the results legible. An agent that never
attempts an unauthorized payment and a policy that refuses every unauthorized
attempt are different achievements, and a system can have one without the
other.
