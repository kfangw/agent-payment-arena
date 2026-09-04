"""Unit tests for the HTTP gateway wire adapter."""

import base64
import json
from datetime import UTC, datetime

import httpx

from arena.gateway.contract import Action, ErrorCode
from arena.gateway.http import HttpGateway
from tests.test_fake_gateway import payment

RESOURCE = "https://merchant.test/item"
TERMS = {
    "x402Version": 1,
    "error": "payment required",
    "errorCode": "payment_required",
    "accepts": [
        {
            "scheme": "exact",
            "network": "eip155:1337",
            "maxAmountRequired": "25",
            "resource": "http://gateway/item",
            "description": "x402 protected resource",
            "mimeType": "application/json",
            "payTo": "0x" + "22" * 20,
            "maxTimeoutSeconds": 60,
            "asset": "0x" + "33" * 20,
            "extra": {},
        }
    ],
}


def test_requirements_rewrites_only_the_origin() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://gateway/item"
        return httpx.Response(402, json=TERMS)

    gateway = HttpGateway(
        "http://gateway", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert gateway.requirements(RESOURCE).max_amount_required == "25"


def test_submit_encodes_payload_and_normalizes_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raw = base64.b64decode(request.headers["X-PAYMENT"])
        assert json.loads(raw)["mandate"]["mandate"]["mandateId"]
        return httpx.Response(200, json={"report": "ready"})

    gateway = HttpGateway(
        "http://gateway", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert gateway.submit(payment(), resource=RESOURCE, now=datetime.now(UTC)).settled


def test_submit_preserves_machine_readable_policy_outcome() -> None:
    body = {**TERMS, "error": "confirmation needed", "errorCode": "confirmation_required"}
    transport = httpx.MockTransport(lambda request: httpx.Response(402, json=body))
    gateway = HttpGateway("http://gateway", client=httpx.Client(transport=transport))
    result = gateway.submit(payment(), resource=RESOURCE, now=datetime.now(UTC))
    assert result.action is Action.ASK
    assert result.error_code is ErrorCode.CONFIRMATION_REQUIRED
