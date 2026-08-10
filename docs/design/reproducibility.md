# Reproducibility

## Purpose

Make a stochastic evaluation worth trusting. An evaluation nobody can rerun is
an anecdote, and one that costs money to rerun does not get rerun. Both
problems are addressed here.

## Behavior

**Cassettes.** Every model call is recorded under a key derived from the
provider, the model identifier, and a hash of the request. Replay is the
default; recording requires an explicit flag. CI replays and calls no
provider, so the suite is free and deterministic.

**Seeds.** Scenario generation, traffic order, and delegator responses are
driven by a seed recorded with the run. Everything the instrument controls is
reproducible even when the model is not.

**Repetitions and intervals.** A configuration is run n times and reported
with an interval per metric. Single-run figures are not published.

**Version recording.** Every reported figure carries the provider, the model
identifier, the attack catalog version, the seed, and the run date.

## Design decisions

**Replay is the default and recording is opt-in.** The reverse ordering means
a routine test run silently spends money and silently changes the recorded
baseline. Making recording deliberate keeps a cassette change visible in the
diff, where it can be reviewed like any other change.

**The cassette key includes the model identifier.** A cassette recorded
against one model version is not a valid response for another. Keying on the
prompt alone would let a model upgrade go unnoticed while the numbers stayed
suspiciously stable.

**Cassettes are committed.** They are the reason CI can run the evaluation at
all, and reviewing a diff of what the model actually returned is often how a
harness bug is found.

**A single number is treated as a defect, not a result.** The report refuses
to emit a figure without repetitions. This is enforced in the reporting code
rather than left to discipline, because the discipline is what fails under
time pressure.

**Model versions are recorded rather than pinned.** Pinning would let this
repository claim a stability it does not have. Results change when models
change, and the record is what lets a later reader tell a real regression from
a model update.

## Limits

Cassettes reproduce a past run; they do not reproduce a model. A figure
replayed from cassettes is a faithful record of what that model returned on
that date and is not evidence about the current model.

Determinism ends at the provider. Seeds fix everything the instrument
controls, and identical inputs to a model can still yield different outputs,
which is the reason repetitions exist rather than an alternative to them.

Intervals describe variation across repetitions of one configuration. They do
not account for variation from prompt wording, attack phrasing, or scenario
choice, each of which can exceed it.

Cassettes accumulate. A large recorded set is a maintenance cost, and a stale
one is a liability once the prompts it was recorded against have changed.

## Status

Not implemented. Cassette recording and replay, seeding, and the reporting
rules land in M1, which is where the first model call happens. The CI job is
already written to run without provider keys, so the constraint is enforced
before the code that must satisfy it exists.
