"""Payment authority shared by scripted and tool-driven agents."""

from __future__ import annotations

import hashlib
from datetime import datetime

from arena.gateway.protocol import Gateway, GatewayResult
from arena.gateway.schemas import Authorization, ExactPayload, PaymentPayload, SignedMandate
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
    ) -> None:
        self.gateway = gateway
        self.signed_mandate = signed_mandate
        self.agent_private_key = agent_private_key
        self.chain_id = chain_id
        self.token_name = token_name
        self.token_version = token_version

    def pay(
        self, resource: str, payee: str, amount: int, *, now: datetime, nonce_key: str
    ) -> GatewayResult:
        """Sign and submit one exact-scheme payment."""
        terms = self.gateway.requirements(resource)
        nonce = "0x" + hashlib.sha256(nonce_key.encode()).hexdigest()
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
        )
        return self.gateway.submit(payload, resource=resource, now=now)
