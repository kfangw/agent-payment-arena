"""Agent wrapper that enforces payment constraints outside model output."""

from dataclasses import dataclass

from arena.agents.protocol import AgentDecision, EvaluationAgent
from arena.scenarios import Resource


@dataclass(frozen=True)
class SchemaConstrainedAgent:
    """Remove payment options that a mandate-constrained tool would not accept."""

    subject: EvaluationAgent
    allowed_payees: frozenset[str]
    maximum_amount: int
    maximum_payments: int = 1

    @property
    def agent_id(self) -> str:
        """Identify the structural defense and wrapped subject."""
        return f"schema-constrained:{self.subject.agent_id}"

    def decide(self, task: str, resource: Resource) -> AgentDecision:
        """Validate the subject's proposed tool arguments before returning them."""
        decision = self.subject.decide(task, resource)
        if not decision.pay or decision.payee is None or decision.amount is None:
            return decision
        allowed = {payee.lower() for payee in self.allowed_payees}
        if decision.payee.lower() not in allowed or decision.amount > self.maximum_amount:
            return AgentDecision(
                pay=False,
                prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens,
            )
        return AgentDecision(
            pay=True,
            payee=decision.payee,
            amount=decision.amount,
            prompt_tokens=decision.prompt_tokens,
            completion_tokens=decision.completion_tokens,
            payment_count=min(decision.payment_count, self.maximum_payments),
        )
