# Agents

## Purpose

Hold the defense as the variable while everything else stays fixed. Four
agents run the same tasks against the same gateway under the same policy, and
the difference between their results is what a defense is worth.

## Behavior

**Scripted.** Deterministic and rule-based. Reads the 402, checks the amount
against the mandate, pays or refuses. It has no language model and therefore
cannot be injected, which makes it the control: the difference between it and
`llm_naive` isolates what the model contributes, in both directions.

**LLM, undefended.** A language model with the payment tools and a task. The
system prompt states the task and nothing about defending against instructions
found in content. This is the floor.

**LLM, prompt-defended.** The same model with a system prompt that states the
mandate, warns that resource content is data rather than instruction, and
requires the payee and amount to be checked against the mandate before paying.
The most common defense in practice, and the one most often assumed to work.

**LLM, schema-constrained.** The same model, with the mandate moved out of the
prompt and into the tool schema. The pay tool accepts only a payee from the
mandate's allowlist and an amount within its limit, so an out-of-scope payment
is not a call the model can make. The defense is in the surface, not in the
model's compliance.

## Design decisions

**A deterministic control is included.** Without it, every number is about
language models in general and nothing is attributable. The scripted agent
also produces the over-blocking baseline: it refuses exactly what its rule
says to refuse, which is the reference for whether a model's caution is
calibrated.

**One framework.** LangGraph, and no second one. A second framework would
double the maintenance and measure nothing about payment authority.

**Defenses are stacked in order of how much they constrain, not by strength.**
Prompt defense asks the model to comply; schema constraint removes the option.
Reporting them on the same axis would suggest they are the same kind of thing.

**Providers are behind a thin abstraction.** Enough to run the same agent on at
least two providers and show that a result is not an artifact of one, and no
more than that. A provider abstraction that grows features becomes a subject
of its own.

## Limits

Four agents are four points, not a space. Nothing here supports a claim about
defenses that were not run.

The prompt-defended agent is one prompt. A different wording gives a different
number, and the difference between two system prompts can exceed the
difference between two models. Reported results name the prompt.

Agents do not learn across runs, by design. An agent that adapts to the attack
catalog would be measuring the catalog.

## Status

Implemented. The deterministic catalog run includes the scripted control, a
content-following subject, and its schema-constrained wrapper. The naive and
prompt-defended LLM subjects use the same provider abstraction and cassette
path so a recorded model response can be compared without live calls in CI.
