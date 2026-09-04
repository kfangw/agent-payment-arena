"""In-memory gateway implementing the evaluation-facing x402 contract."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

from arena.gateway.contract import Action, ErrorCode
from arena.gateway.protocol import AcceptPolicy, GatewayResult, PaymentContext
from arena.gateway.schemas import Mandate, PaymentPayload, PaymentRequirements
from arena.gateway.signatures import verify_authorization, verify_confirmation, verify_mandate


class FakeGateway:
    """Verify signed payments and enforce mandate scope without a chain."""

    def __init__(
        self,
        *,
        network: str,
        chain_id: int,
        pay_to: str,
        asset: str,
        price: int,
        token_name: str = "KRW Test Stablecoin",
        token_version: str = "1",
        policy: AcceptPolicy | None = None,
    ) -> None:
        self.network = network
        self.chain_id = chain_id
        self.pay_to = pay_to
        self.asset = asset
        self.price = price
        self.token_name = token_name
        self.token_version = token_version
        self.policy = policy
        self._used_nonces: set[str] = set()
        self._spend: dict[str, deque[tuple[datetime, int]]] = defaultdict(deque)

    def requirements(self, resource: str) -> PaymentRequirements:
        """Return exact-scheme terms for one resource."""
        return PaymentRequirements(
            network=self.network,
            maxAmountRequired=str(self.price),
            resource=resource,
            description="x402 protected resource",
            mimeType="application/json",
            payTo=self.pay_to,
            maxTimeoutSeconds=60,
            asset=self.asset,
            extra={"name": self.token_name, "version": self.token_version},
        )

    def submit(self, payload: PaymentPayload, *, resource: str, now: datetime) -> GatewayResult:
        """Verify, authorize, account for, and settle one payment in memory."""
        authorization = payload.payload.authorization
        terms = self.requirements(resource)
        if (
            payload.x402_version != 1
            or payload.scheme != "exact"
            or payload.network != self.network
        ):
            return self._reject(ErrorCode.VERIFICATION_FAILED, "payment protocol mismatch")
        if (
            authorization.to.lower() != self.pay_to.lower()
            or int(authorization.value) != self.price
        ):
            return self._reject(ErrorCode.VERIFICATION_FAILED, "payment does not match terms")
        timestamp = int(now.timestamp())
        if timestamp < int(authorization.valid_after) or timestamp >= int(
            authorization.valid_before
        ):
            return self._reject(ErrorCode.VERIFICATION_FAILED, "authorization is outside validity")
        if authorization.nonce in self._used_nonces:
            return self._reject(
                ErrorCode.VERIFICATION_FAILED, "authorization nonce was already used"
            )
        if not verify_authorization(
            authorization,
            payload.payload.signature,
            token_name=self.token_name,
            token_version=self.token_version,
            chain_id=self.chain_id,
            token_address=self.asset,
        ):
            return self._reject(ErrorCode.VERIFICATION_FAILED, "authorization signature is invalid")
        mandate_result = self._check_mandate(payload, terms.resource, now)
        if mandate_result is not None:
            return mandate_result
        if payload.confirmation is not None and not self._confirmation_approves(
            payload, terms.resource, now
        ):
            return self._reject(ErrorCode.MANDATE_INVALID, "confirmation is invalid")
        context = PaymentContext(payload=payload, requirements=terms, verified=True, now=now)
        decision = self.policy.decide(context) if self.policy else GatewayResult(Action.APPROVE)
        if decision.action is not Action.APPROVE:
            return decision
        self._used_nonces.add(authorization.nonce)
        self._record_spend(payload, now)
        return GatewayResult(Action.APPROVE, settled=True)

    def _check_mandate(
        self, payload: PaymentPayload, resource: str, now: datetime
    ) -> GatewayResult | None:
        signed = payload.mandate
        if signed is None:
            return self._reject(ErrorCode.MANDATE_MISSING, "a signed mandate is required")
        mandate = signed.mandate
        if not verify_mandate(signed, self.chain_id):
            return self._reject(ErrorCode.MANDATE_INVALID, "mandate signature is invalid")
        authorization = payload.payload.authorization
        if mandate.agent.lower() != authorization.from_address.lower():
            return self._reject(ErrorCode.MANDATE_INVALID, "payer is not the mandated agent")
        timestamp = int(now.timestamp())
        if timestamp < int(mandate.valid_after) or timestamp >= int(mandate.valid_before):
            return self._reject(ErrorCode.MANDATE_EXPIRED, "mandate is outside validity")
        if mandate.allowed_payees and self.pay_to.lower() not in {
            payee.lower() for payee in mandate.allowed_payees
        }:
            return self._reject(ErrorCode.MANDATE_EXCEEDED, "payee is outside mandate scope")
        if mandate.allowed_resources and not any(
            resource.startswith(prefix) for prefix in mandate.allowed_resources
        ):
            return self._reject(ErrorCode.MANDATE_EXCEEDED, "resource is outside mandate scope")
        amount = int(authorization.value)
        if (
            int(mandate.max_amount_per_payment)
            and amount > int(mandate.max_amount_per_payment)
            and not self._confirmation_approves(payload, resource, now)
        ):
            return self._reject(ErrorCode.MANDATE_EXCEEDED, "amount exceeds mandate limit")
        return self._check_windows(mandate, amount, now)

    def _check_windows(self, mandate: Mandate, amount: int, now: datetime) -> GatewayResult | None:
        events = self._spend[mandate.mandate_id]
        budget_window = int(mandate.budget_window_seconds)
        if budget_window:
            while events and (now - events[0][0]).total_seconds() >= budget_window:
                events.popleft()
        if int(mandate.budget_amount) and sum(value for _, value in events) + amount > int(
            mandate.budget_amount
        ):
            return self._reject(ErrorCode.MANDATE_BUDGET_EXCEEDED, "mandate budget exceeded")
        rate_window = int(mandate.rate_window_seconds)
        if int(mandate.max_payments_per_window) and rate_window:
            recent = sum(1 for at, _ in events if (now - at).total_seconds() < rate_window)
            if recent >= int(mandate.max_payments_per_window):
                return self._reject(ErrorCode.MANDATE_RATE_EXCEEDED, "mandate rate exceeded")
        return None

    def _confirmation_approves(self, payload: PaymentPayload, resource: str, now: datetime) -> bool:
        confirmation = payload.confirmation
        signed = payload.mandate
        if confirmation is None or signed is None:
            return False
        authorization = payload.payload.authorization
        return (
            verify_confirmation(confirmation, signed.mandate.delegator, self.chain_id)
            and confirmation.mandate_id == signed.mandate.mandate_id
            and confirmation.authorization_nonce == authorization.nonce
            and confirmation.amount == authorization.value
            and confirmation.resource == resource
            and int(now.timestamp()) < int(confirmation.valid_before)
        )

    def _record_spend(self, payload: PaymentPayload, now: datetime) -> None:
        if payload.mandate is not None:
            self._spend[payload.mandate.mandate.mandate_id].append(
                (now, int(payload.payload.authorization.value))
            )

    @staticmethod
    def _reject(code: ErrorCode, reason: str) -> GatewayResult:
        return GatewayResult(Action.REJECT, error_code=code, reason=reason)
