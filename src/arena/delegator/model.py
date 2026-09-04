"""Delegator behavior used when a payment policy requests confirmation."""

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from arena.gateway.schemas import AskRequest, Confirmation
from arena.gateway.signatures import sign_confirmation


class Delegator(Protocol):
    """Person or model authorized to confirm one payment."""

    def confirm(self, request: AskRequest, *, now: datetime) -> Confirmation | None:
        """Return a payment-bound confirmation, or decline to confirm."""
        ...

    @property
    def last_latency_ms(self) -> float:
        """Return simulated latency of the latest response."""
        ...


@dataclass(frozen=True)
class SigningDelegator:
    """Deterministic delegator that approves every request correctly."""

    private_key: str
    chain_id: int
    validity: timedelta = timedelta(minutes=5)

    @property
    def last_latency_ms(self) -> float:
        """Return zero for the immediate deterministic control."""
        return 0.0

    def confirm(self, request: AskRequest, *, now: datetime) -> Confirmation:
        """Sign a short-lived confirmation for exactly one request."""
        valid_before = int((now + self.validity).timestamp())
        return sign_confirmation(request, valid_before, self.private_key, self.chain_id)


@dataclass
class BehavioralDelegator:
    """Seeded delegator with configurable error, delay, non-response, and fatigue."""

    private_key: str
    chain_id: int
    seed: int
    approval_probability: float = 1.0
    nonresponse_probability: float = 0.0
    base_latency_ms: float = 0.0
    fatigue_per_question: float = 0.0
    validity: timedelta = timedelta(minutes=5)
    questions: int = 0
    _last_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration before the first response."""
        for value in (
            self.approval_probability,
            self.nonresponse_probability,
            self.fatigue_per_question,
        ):
            if not 0 <= value <= 1:
                raise ValueError("delegator probabilities must be between zero and one")
        if self.base_latency_ms < 0:
            raise ValueError("delegator latency must be non-negative")

    @property
    def last_latency_ms(self) -> float:
        """Return simulated latency of the latest response."""
        return self._last_latency_ms

    def confirm(self, request: AskRequest, *, now: datetime) -> Confirmation | None:
        """Sample a reproducible response without sleeping in the evaluator."""
        self.questions += 1
        self._last_latency_ms = self.base_latency_ms * self.questions
        key = f"{self.seed}:{self.questions}:{request.model_dump_json()}"
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
        if rng.random() < self.nonresponse_probability:
            return None
        probability = max(
            0.0,
            self.approval_probability - self.fatigue_per_question * (self.questions - 1),
        )
        if rng.random() >= probability:
            return None
        valid_before = int((now + self.validity).timestamp())
        return sign_confirmation(request, valid_before, self.private_key, self.chain_id)
