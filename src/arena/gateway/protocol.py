"""Backend and policy interfaces shared by the evaluation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from arena.gateway.contract import Action, ErrorCode
from arena.gateway.schemas import PaymentPayload, PaymentRequirements


@dataclass(frozen=True)
class GatewayResult:
    """Normalized outcome returned by every gateway backend."""

    action: Action
    error_code: ErrorCode | None = None
    reason: str = ""
    settled: bool = False


@dataclass(frozen=True)
class PaymentContext:
    """Inputs visible to an accept policy for one payment."""

    payload: PaymentPayload
    requirements: PaymentRequirements
    verified: bool
    now: datetime


class AcceptPolicy(Protocol):
    """Side-effect-free decision policy."""

    def decide(self, context: PaymentContext) -> GatewayResult:
        """Return a decision for one verified or rejected payment."""
        ...


class Gateway(Protocol):
    """Operations the evaluation loop requires from a gateway backend."""

    def requirements(self, resource: str) -> PaymentRequirements:
        """Return the payment terms for a protected resource."""
        ...

    def submit(self, payload: PaymentPayload, *, now: datetime) -> GatewayResult:
        """Evaluate and possibly settle one payment."""
        ...
