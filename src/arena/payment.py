"""Payment authority shared by scripted and tool-driven agents."""

from __future__ import annotations

import hashlib
from datetime import datetime

from arena.delegator.model import Delegator
from arena.gateway.contract import Action
from arena.gateway.protocol import Gateway, GatewayResult
from arena.gateway.schemas import (
    AskRequest,
    Authorization,
    Confirmation,
    ExactPayload,
    PaymentPayload,
    SignedMandate,
)
from arena.gateway.signatures import sign_authorization


class PaymentAuthority:
    """Create signed payment attempts for one mandated agent."""

    def __init__(
        self,
        gateway: Gateway,
        signed_mandate: SignedMandate,
        agent_private_key: str,
        *,
        chain_id: int,
        token_name: str,
        token_version: str,
        delegator: Delegator | None = None,
    ) -> None:
        self.gateway = gateway
        self.signed_mandate = signed_mandate
        self.agent_private_key = agent_private_key
        self.chain_id = chain_id
        self.token_name = token_name
        self.token_version = token_version
        self.delegator = delegator
        self.last_escalation_latency_ms = 0.0

    def ask_request(self, resource: str, amount: int, nonce_key: str) -> AskRequest:
        """Describe the exact prospective payment a delegator must confirm."""
        return AskRequest(
            mandateId=self.signed_mandate.mandate.mandate_id,
            authorizationNonce=self._nonce(nonce_key),
            amount=str(amount),
            resource=resource,
        )

    def pay(
        self,
        resource: str,
        payee: str,
        amount: int,
        *,
        now: datetime,
        nonce_key: str,
        confirmation: Confirmation | None = None,
    ) -> GatewayResult:
        """Sign and submit one exact-scheme payment."""
        terms = self.gateway.requirements(resource)
        nonce = self._nonce(nonce_key)
        authorization = Authorization(
            **{
                "from": self.signed_mandate.mandate.agent,
                "to": payee,
                "value": str(amount),
                "validAfter": "0",
                "validBefore": str(int(now.timestamp()) + terms.max_timeout_seconds),
                "nonce": nonce,
            }
        )
        signature = sign_authorization(
            authorization,
            self.agent_private_key,
            token_name=self.token_name,
            token_version=self.token_version,
            chain_id=self.chain_id,
            token_address=terms.asset,
        )
        payload = PaymentPayload(
            x402Version=1,
            scheme=terms.scheme,
            network=terms.network,
            payload=ExactPayload(signature=signature, authorization=authorization),
            mandate=self.signed_mandate,
            confirmation=confirmation,
        )
        return self.gateway.submit(payload, resource=resource, now=now)

    def pay_with_escalation(
        self, resource: str, payee: str, amount: int, *, now: datetime, nonce_key: str
    ) -> tuple[GatewayResult, ...]:
        """Attempt payment and retry once with a delegator confirmation after ASK."""
        first = self.pay(resource, payee, amount, now=now, nonce_key=nonce_key)
        self.last_escalation_latency_ms = 0.0
        if first.action is not Action.ASK or self.delegator is None:
            return (first,)
        request = self.ask_request(resource, amount, nonce_key)
        confirmation = self.delegator.confirm(request, now=now)
        self.last_escalation_latency_ms = self.delegator.last_latency_ms
        if confirmation is None:
            return (first,)
        second = self.pay(
            resource,
            payee,
            amount,
            now=now,
            nonce_key=nonce_key,
            confirmation=confirmation,
        )
        return first, second

    @staticmethod
    def _nonce(nonce_key: str) -> str:
        return "0x" + hashlib.sha256(nonce_key.encode()).hexdigest()
