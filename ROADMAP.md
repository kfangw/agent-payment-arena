# Roadmap

Feature checklist of agent-payment-arena. The repository stays an evaluation
instrument; every item is scoped to that purpose. Check items off as they land.

The unchecked items below build toward one claim that can be defended: for a
fixed model, a stated defense changes what an agent spends outside its
authorization, by an amount reported with repetitions and an interval, on
traffic that includes the legitimate tasks the defense could block. Milestones
M0 through M5 mark the order.

## Foundation

- [x] uv project on Python 3.12 with ruff, mypy in strict mode, and pytest, installable from a clean checkout with no keys and no network services
- [x] Shared five-outcome decision space and the refusal codes a gateway reports, pinned by tests because clients branch on the exact strings (`src/arena/gateway/contract.py`)
- [x] English-only pre-commit hook, run as its own CI step (`scripts/check_english_only.py`)
- [x] CI that clears provider keys in the job environment, so a mismarked test fails loudly instead of reaching a provider
- [x] Scope documents written before any result: the threat model, and what the instrument does not measure (`docs/`)

## Contract — M0

- [x] 402 response payload: the fields a gateway sends with payment terms, and which of them an attacker controls
- [x] Mandate schema with its EIP-712 types, reproduced closely enough that a signature produced here verifies against the reference gateway
- [x] Ask flow schema: the confirmation a delegator signs, and how it binds to a single payment
- [x] `FakeGateway`: in-memory, real signature verification through `eth-account`, no chain and no node (`src/arena/gateway/fake.py`)
- [x] Contract tests over the fake backend, written against the shared gateway protocol

## Minimum viable arena — M1

- [x] MCP server exposing the complete payment authority, including delegator escalation (`src/arena/mcp_server/`)
- [x] Scripted agent as the deterministic control, and a cassette-backed LLM agent adapter (`src/arena/agents/`)
- [x] One prompt injection attack with its benign twin, both run in the same pass
- [x] Two accept policies: always verify, and ask the delegator above a limit
- [x] Run loop and scoring: scenario, agent execution, ground truth comparison (`src/arena/loop.py`)
- [x] Generated comparison table across policies and agents, carrying model identifiers and the run date (`src/arena/report.py`)
- [x] Response cassettes, recorded with an explicit callback and replayed by default, so CI runs without a provider key
- [x] `arena demo`: one offline command exercising payment and escalation through MCP tools before producing the minimum evaluation report

## Attack catalog — M2

- [ ] Prompt injection in at least three variants: direct instruction, induced privilege escalation, and payee substitution with the task left intact
- [ ] Payee spoofing, priced against both a string comparison and a judgment made by the model
- [ ] Price inflation: above value, below every absolute limit the mandate sets
- [ ] Repeat purchase: individually authorized payments that are collectively outside intent
- [ ] A benign twin for each of the above, so over-blocking is measured in the same run
- [ ] Two further agents: defended by the system prompt, and constrained by the tool schema so the mandate cannot be exceeded structurally

## Delegator model — M3

- [ ] Delegator responses with error, delay, and non-response as configurable behavior (`src/arena/delegator/`)
- [ ] Fatigue: answer quality that degrades with the number of questions already asked
- [ ] Escalation cost in the metric vector, so asking is never free

## Real gateway — M4

- [ ] `HttpGateway` against a running stablecoin-x402-gateway (`src/arena/gateway/http.py`)
- [ ] Docker Compose stack bringing up the gateway and a node for the HTTP backend
- [ ] Differential test: the same scenarios through both backends, asserting they agree
- [ ] CI job for the HTTP backend, separate from the default suite

## Frontier — M5

- [ ] Policy grid search over the limit, the ask threshold, and the bond requirement
- [ ] Trade-off curves over unauthorized spend, over-blocking, escalations, and token cost
- [ ] Repetition and interval reporting applied to every published figure
- [ ] OpenTelemetry traces for a run, so a single decision can be inspected after the fact

## Deliberately out of scope

- Fine-tuning, and any agent that learns across runs. The instrument compares
  fixed subjects.
- A second agent framework. One is enough to show the tool surface; two double
  the maintenance and measure nothing new.
- A vector store. Nothing in this problem retrieves over a corpus.
- Reproducing the gateway's settlement path in Python. The fake backend
  reproduces the contract, and the HTTP backend covers the rest.
