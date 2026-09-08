"""Validated finite settings and public decision state."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    """One public exposure and stage chain; empty hazards means settled."""

    amount: float
    hazards: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Reject amounts and hazards outside the model's domain."""
        if not math.isfinite(self.amount) or self.amount <= 0:
            raise ValueError("amount must be finite and positive")
        if any(not math.isfinite(f) or not 0 <= f <= 1 for f in self.hazards):
            raise ValueError("hazards must lie in [0, 1]")


@dataclass(frozen=True)
class Setting:
    """Independent finite request distributions in chronological order."""

    schedule: tuple[tuple[tuple[float, Request], ...], ...]
    prior: float = 0.5
    low: float = 0.01
    high: float = 0.99
    margin: float = 1.0
    harm: float = 1.0
    query_cost: float = 0.1
    wait_cost: float = 0.0
    response_rate: float = 1.0
    deadline: int = 2

    def __post_init__(self) -> None:
        """Reject settings whose numbers fall outside the model's domain."""
        numbers = (
            self.prior,
            self.low,
            self.high,
            self.margin,
            self.harm,
            self.query_cost,
            self.wait_cost,
            self.response_rate,
        )
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("parameters must be finite")
        if not 0 <= self.prior <= 1 or not 0 < self.low < self.high < 1:
            raise ValueError("invalid type probabilities")
        if min(self.margin, self.harm, self.query_cost) <= 0 or self.wait_cost < 0:
            raise ValueError("invalid costs")
        if not 0 <= self.response_rate <= 1:
            raise ValueError("response_rate must lie in [0, 1]")
        if type(self.deadline) is not int or self.deadline < 1:
            raise ValueError("deadline must be a positive integer")
        if not self.schedule:
            raise ValueError("schedule cannot be empty")
        for distribution in self.schedule:
            if not distribution or any(not math.isfinite(w) or w <= 0 for w, _ in distribution):
                raise ValueError("weights must be finite and positive")
            if not math.isclose(sum(w for w, _ in distribution), 1.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError("weights must sum to one")

    @property
    def response_probability(self) -> float:
        """Probability of an answer within the full window without failure."""
        return 1 - (1 - self.response_rate) ** self.deadline

    def belief(self, misuse: int, normal: int) -> float:
        """Stable posterior from observed labels only, without discretization."""
        if min(misuse, normal) < 0:
            raise ValueError("negative label counts")
        if self.prior in (0, 1):
            return self.prior
        odds = (
            math.log(self.prior / (1 - self.prior))
            + misuse * math.log(self.high / self.low)
            + normal * math.log((1 - self.high) / (1 - self.low))
        )
        if odds >= 0:
            return 1 / (1 + math.exp(-odds))
        exp_odds = math.exp(odds)
        return exp_odds / (1 + exp_odds)

    def risk(self, misuse: int, normal: int) -> float:
        """Predict the current request's intent, not its settlement outcome."""
        return self.low + (self.high - self.low) * self.belief(misuse, normal)


@dataclass(frozen=True)
class PublicState:
    """Policy input, excluding latent intent and realized harm."""

    request_index: int
    request: Request
    stage: int
    misuse_labels: int = 0
    normal_labels: int = 0
