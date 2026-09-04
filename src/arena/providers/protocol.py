"""Minimal provider interface required by evaluation agents."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    """Normalized model text and token usage."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class ModelProvider(Protocol):
    """Provider-independent text generation interface."""

    provider_id: str
    model_id: str

    def complete(self, prompt: str) -> ModelResponse:
        """Generate or replay one model response."""
        ...
