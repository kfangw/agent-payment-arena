"""HTTP adapter for a running x402 gateway."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from urllib.parse import urlsplit

import httpx

from arena.gateway.contract import Action, ErrorCode
from arena.gateway.protocol import GatewayResult
from arena.gateway.schemas import PaymentPayload, PaymentRequirements, RequirementsResponse

ACTION_BY_CODE = {
    ErrorCode.CONFIRMATION_REQUIRED: Action.ASK,
    ErrorCode.PAYMENT_DEFERRED: Action.DEFER,
    ErrorCode.BOND_REQUIRED: Action.REQUIRE_BOND,
}


class HttpGateway:
    """Carry the shared gateway protocol over x402 HTTP headers."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def requirements(self, resource: str) -> PaymentRequirements:
        """Fetch and validate the first payment option from a 402 response."""
        response = self.client.get(self._url(resource))
        if response.status_code != httpx.codes.PAYMENT_REQUIRED:
            raise RuntimeError(f"gateway returned HTTP {response.status_code}, expected 402")
        required = RequirementsResponse.model_validate(response.json())
        if not required.accepts:
            raise RuntimeError("gateway returned no payment options")
        return required.accepts[0]

    def submit(self, payload: PaymentPayload, *, resource: str, now: datetime) -> GatewayResult:
        """Send a base64-encoded X-PAYMENT header and normalize the outcome."""
        del now  # The live gateway uses its own clock.
        encoded = base64.b64encode(
            json.dumps(
                payload.model_dump(by_alias=True, mode="json"), separators=(",", ":")
            ).encode()
        ).decode()
        response = self.client.get(self._url(resource), headers={"X-PAYMENT": encoded})
        if response.is_success:
            return GatewayResult(Action.APPROVE, settled=True)
        if response.status_code != httpx.codes.PAYMENT_REQUIRED:
            return GatewayResult(
                Action.REJECT,
                ErrorCode.SETTLEMENT_ERROR,
                f"unexpected HTTP status {response.status_code}",
            )
        required = RequirementsResponse.model_validate(response.json())
        try:
            code = ErrorCode(required.error_code or ErrorCode.POLICY_REJECTED)
        except ValueError:
            code = ErrorCode.VERIFICATION_ERROR
        return GatewayResult(ACTION_BY_CODE.get(code, Action.REJECT), code, required.error)

    def close(self) -> None:
        """Close the internally-created HTTP client."""
        if self._owns_client:
            self.client.close()

    def _url(self, resource: str) -> str:
        path = urlsplit(resource).path or "/"
        return f"{self.base_url}{path}"
