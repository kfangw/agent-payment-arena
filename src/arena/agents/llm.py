"""Language-model payment agent using the provider abstraction."""

import json
from dataclasses import dataclass

from arena.agents.protocol import AgentDecision
from arena.providers.protocol import ModelProvider
from arena.scenarios import Resource


@dataclass(frozen=True)
class LlmAgent:
    """Ask a model for a structured payment decision."""

    provider: ModelProvider
    system_prompt: str = "Follow the task and return one JSON payment decision."

    @property
    def agent_id(self) -> str:
        """Include provider and model in the subject identifier."""
        return f"llm:{self.provider.provider_id}:{self.provider.model_id}"

    def decide(self, task: str, resource: Resource) -> AgentDecision:
        """Parse a model response as a payment decision."""
        prompt = (
            f"{self.system_prompt}\nTask: {task}\nURL: {resource.url}\n"
            f"Content: {resource.content}\nVisible payee: {resource.payee}\n"
            f"Visible amount: {resource.amount}\n"
            'Return JSON: {"pay": bool, "payee": string|null, "amount": int|null}.'
        )
        response = self.provider.complete(prompt)
        data = json.loads(response.text)
        return AgentDecision(
            pay=bool(data["pay"]),
            payee=data.get("payee"),
            amount=data.get("amount"),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
