# Delegator

## Purpose

Stop `ask` from looking free. A policy that escalates whenever it is unsure
scores well on unauthorized spend and well on over-blocking, and it does so by
moving the cost onto a person the measurement does not model. Modeling that
person is what puts the cost back.

## Behavior

The delegator answers confirmation requests. It signs a confirmation bound to
a single payment, refuses, or does neither.

Four behaviors are configurable.

**Delay.** An answer arrives after a distribution of wall-clock time, which
matters because a task blocked on an answer is a task not completed.

**Error.** The answer is wrong: an out-of-scope payment confirmed, or a
legitimate one refused. The delegator is not an oracle, and treating them as
one would make every escalation a guaranteed correct decision.

**Non-response.** No answer arrives. The policy has to decide what a timeout
means, and that decision is itself a policy under evaluation.

**Fatigue.** Answer quality degrades with the number of questions already
asked. A delegator asked twenty times approves faster and reads less closely
than one asked twice, which turns escalation frequency into a cause of error
rather than a defense against it.

## Design decisions

**The delegator is stated as an empirical response model, and its parameters
are declared rather than fitted.** Nothing here claims these rates match a
real person. They are settings whose effect on the comparison is measured by
varying them, and results report the setting used.

**Escalation is counted separately from the answer.** Asking has a cost
whether or not the answer is correct, and combining the two would let a
policy that asks constantly and happens to receive correct answers look
cheap.

**Fatigue is a function of count, not of time.** The count is what a policy
can observe and act on, and the reference gateway already keeps per-delegator
confirmation history, so the two sides model the same quantity.

**The delegator signs.** Confirmations are real signatures bound to a single
payment, matching the gateway's ask flow, so a confirmation cannot be replayed
onto another payment inside the arena any more than it could outside it.

## Limits

This is a model of a person, not a person. It supports comparisons between
policies under a stated delegator, not predictions of what a human would do.

The parameters are not calibrated against observed behavior. A result that is
sensitive to them says more about the setting than about the policy, which is
why sensitivity to the delegator setting is reported rather than hidden.

One delegator, one mandate, one agent. Shared authority across several agents
or several delegators is not modeled.

## Status

Not implemented. `src/arena/delegator/` holds the package docstring only. A
delegator that always answers correctly lands in M1 so the ask flow can run at
all; error, delay, non-response, and fatigue land in M3.
