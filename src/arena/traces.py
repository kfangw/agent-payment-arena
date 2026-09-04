"""Structured event records produced by evaluation runs."""

from dataclasses import dataclass

from arena.gateway.protocol import GatewayResult


@dataclass(frozen=True)
class PaymentEvent:
    """One payment attempt and its gateway outcome."""

    payee: str
    amount: int
    result: GatewayResult


@dataclass(frozen=True)
class RunTrace:
    """Complete observable execution record for one scenario."""

    scenario_id: str
    agent_id: str
    events: tuple[PaymentEvent, ...]
    task_completed: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    escalation_latency_ms: float = 0.0
