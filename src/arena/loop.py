"""Deterministic execution loop for one scenario and agent."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

from arena.agents.protocol import EvaluationAgent
from arena.payment import PaymentAuthority
from arena.scenarios import Scenario
from arena.telemetry import TraceSink
from arena.traces import PaymentEvent, RunTrace


def run_scenario(
    scenario: Scenario,
    agent: EvaluationAgent,
    authority: PaymentAuthority,
    *,
    now: datetime,
    repetition: int = 0,
    trace_sink: TraceSink | None = None,
) -> RunTrace:
    """Execute one scenario and retain every payment outcome."""
    started = perf_counter()
    decision = agent.decide(scenario.task, scenario.resource)
    events: tuple[PaymentEvent, ...] = ()
    completed = False
    escalation_latency_ms = 0.0
    if decision.pay and decision.payee is not None and decision.amount is not None:
        collected: list[PaymentEvent] = []
        for payment_index in range(decision.payment_count):
            results = authority.pay_with_escalation(
                scenario.resource.url,
                decision.payee,
                decision.amount,
                now=now,
                nonce_key=(f"{scenario.scenario_id}:{agent.agent_id}:{repetition}:{payment_index}"),
            )
            collected.extend(
                PaymentEvent(decision.payee, decision.amount, result) for result in results
            )
            escalation_latency_ms += authority.last_escalation_latency_ms
        events = tuple(collected)
        completed = any(event.result.settled for event in events)
    trace = RunTrace(
        scenario_id=scenario.scenario_id,
        agent_id=agent.agent_id,
        events=events,
        task_completed=completed,
        prompt_tokens=decision.prompt_tokens,
        completion_tokens=decision.completion_tokens,
        latency_ms=(perf_counter() - started) * 1000,
        escalation_latency_ms=escalation_latency_ms,
    )
    if trace_sink is not None:
        for index, event in enumerate(events):
            trace_sink.record(
                "arena.payment",
                {
                    "scenario.id": scenario.scenario_id,
                    "agent.id": agent.agent_id,
                    "payment.index": index,
                    "payment.payee": event.payee,
                    "payment.amount": event.amount,
                    "gateway.action": event.result.action.value,
                    "payment.settled": event.result.settled,
                },
            )
        trace_sink.record(
            "arena.run",
            {
                "scenario.id": scenario.scenario_id,
                "agent.id": agent.agent_id,
                "task.completed": completed,
                "escalation.latency_ms": escalation_latency_ms,
            },
        )
    return trace
