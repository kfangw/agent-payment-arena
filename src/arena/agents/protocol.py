"""Agent interface used by the evaluation loop."""

from dataclasses import dataclass
from typing import Protocol

from arena.scenarios import Resource


@dataclass(frozen=True)
class AgentDecision:
    """Payment intent and model usage returned by an agent."""

    pay: bool
    payee: str | None = None
    amount: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    payment_count: int = 1


class EvaluationAgent(Protocol):
    """Subject that decides whether and how to pay for a resource."""

    @property
    def agent_id(self) -> str:
        """Return the stable identifier included in result records."""
        ...

    def decide(self, task: str, resource: Resource) -> AgentDecision:
        """Return a payment decision without access to ground truth."""
        ...
