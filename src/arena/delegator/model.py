"""Delegator behavior used when a payment policy requests confirmation."""

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


@dataclass(frozen=True)
class SigningDelegator:
    """Deterministic delegator that approves every request correctly."""

    private_key: str
    chain_id: int
    validity: timedelta = timedelta(minutes=5)

    def confirm(self, request: AskRequest, *, now: datetime) -> Confirmation:
        """Sign a short-lived confirmation for exactly one request."""
        valid_before = int((now + self.validity).timestamp())
        return sign_confirmation(request, valid_before, self.private_key, self.chain_id)
