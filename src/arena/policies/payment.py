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


@dataclass(frozen=True)
class ParameterizedPolicy:
    """Apply configurable reject, bond, and ask thresholds in strict order."""

    spending_limit: int
    ask_threshold: int | None = None
    bond_threshold: int | None = None

    def decide(self, context: PaymentContext) -> GatewayResult:
        """Choose the first configured guard crossed by a verified payment."""
        if not context.verified:
            return GatewayResult(Action.REJECT, ErrorCode.VERIFICATION_FAILED)
        amount = int(context.payload.payload.authorization.value)
        if amount > self.spending_limit:
            return GatewayResult(
                Action.REJECT, ErrorCode.POLICY_REJECTED, "payment exceeds spending limit"
            )
        if self.bond_threshold is not None and amount > self.bond_threshold:
            return GatewayResult(
                Action.REQUIRE_BOND, ErrorCode.BOND_REQUIRED, "payment requires a bond"
            )
        if (
            self.ask_threshold is not None
            and amount > self.ask_threshold
            and context.payload.confirmation is None
        ):
            return GatewayResult(
                Action.ASK, ErrorCode.CONFIRMATION_REQUIRED, "payment requires confirmation"
            )
        return GatewayResult(Action.APPROVE)
