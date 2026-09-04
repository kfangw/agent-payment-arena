"""Deterministic execution loop for one scenario and agent."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

from arena.agents.protocol import EvaluationAgent
from arena.payment import PaymentAuthority
from arena.scenarios import Scenario
from arena.traces import PaymentEvent, RunTrace


def run_scenario(
    scenario: Scenario,
    agent: EvaluationAgent,
    authority: PaymentAuthority,
    *,
    now: datetime,
    repetition: int = 0,
) -> RunTrace:
    """Execute one scenario and retain every payment outcome."""
    started = perf_counter()
    decision = agent.decide(scenario.task, scenario.resource)
    events: tuple[PaymentEvent, ...] = ()
    completed = False
    if decision.pay and decision.payee is not None and decision.amount is not None:
        results = authority.pay_with_escalation(
            scenario.resource.url,
            decision.payee,
            decision.amount,
            now=now,
            nonce_key=f"{scenario.scenario_id}:{agent.agent_id}:{repetition}",
        )
        events = tuple(PaymentEvent(decision.payee, decision.amount, result) for result in results)
        completed = results[-1].settled
    return RunTrace(
        scenario_id=scenario.scenario_id,
        agent_id=agent.agent_id,
        events=events,
        task_completed=completed,
        prompt_tokens=decision.prompt_tokens,
        completion_tokens=decision.completion_tokens,
        latency_ms=(perf_counter() - started) * 1000,
    )
