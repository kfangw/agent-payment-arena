"""Deterministic control agents for harness validation."""

from dataclasses import dataclass

from arena.agents.protocol import AgentDecision
from arena.scenarios import Resource


@dataclass(frozen=True)
class ScriptedAgent:
    """Pay visible terms only when they satisfy configured authorization."""

    allowed_payees: frozenset[str]
    maximum_amount: int
    agent_id: str = "scripted"

    def decide(self, task: str, resource: Resource) -> AgentDecision:
        """Apply deterministic allowlist and amount checks."""
        allowed = {payee.lower() for payee in self.allowed_payees}
        if resource.payee.lower() not in allowed or resource.amount > self.maximum_amount:
            return AgentDecision(pay=False)
        return AgentDecision(
            pay=True,
            payee=resource.payee,
            amount=resource.amount,
            payment_count=min(resource.payment_count, 1),
        )


@dataclass(frozen=True)
class ContentFollowingAgent:
    """Deterministic vulnerable subject that trusts visible payment terms."""

    agent_id: str = "content-following"

    def decide(self, task: str, resource: Resource) -> AgentDecision:
        """Pay the recipient and amount supplied by the resource."""
        return AgentDecision(
            pay=True,
            payee=resource.payee,
            amount=resource.amount,
            payment_count=resource.payment_count,
        )
