"""Baseline accept policies for payment evaluations."""

from dataclasses import dataclass

from arena.gateway.contract import Action, ErrorCode
from arena.gateway.protocol import GatewayResult, PaymentContext


@dataclass(frozen=True)
class AlwaysVerifyPolicy:
    """Approve exactly when cryptographic verification succeeded."""

    def decide(self, context: PaymentContext) -> GatewayResult:
        """Approve verified payments and reject all others."""
        if context.verified:
            return GatewayResult(Action.APPROVE)
        return GatewayResult(Action.REJECT, ErrorCode.VERIFICATION_FAILED)


@dataclass(frozen=True)
class AskAbovePolicy:
    """Ask the delegator before approving payments above a threshold."""

    threshold: int

    def decide(self, context: PaymentContext) -> GatewayResult:
        """Return an ask decision when the signed amount is above the limit."""
        amount = int(context.payload.payload.authorization.value)
        if amount > self.threshold and context.payload.confirmation is None:
            return GatewayResult(
                Action.ASK,
                ErrorCode.CONFIRMATION_REQUIRED,
                "payment exceeds the confirmation threshold",
            )
        return GatewayResult(Action.APPROVE)
