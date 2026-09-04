"""Tests for baseline payment policies."""

from datetime import UTC, datetime

from arena.gateway.contract import Action
from arena.gateway.protocol import PaymentContext
from arena.gateway.schemas import Authorization, ExactPayload, PaymentPayload, PaymentRequirements
from arena.policies.payment import AlwaysVerifyPolicy, AskAbovePolicy

ADDRESS = "0x" + "11" * 20


def context(amount: int, *, verified: bool = True) -> PaymentContext:
    authorization = Authorization(
        **{
            "from": ADDRESS,
            "to": ADDRESS,
            "value": str(amount),
            "validAfter": "0",
            "validBefore": "2",
            "nonce": "0x" + "22" * 32,
        }
    )
    payload = PaymentPayload(
        x402Version=1,
        network="eip155:1",
        payload=ExactPayload(signature="0x" + "33" * 65, authorization=authorization),
    )
    terms = PaymentRequirements(
        network="eip155:1",
        maxAmountRequired=str(amount),
        resource="https://example.test/item",
        description="item",
        mimeType="application/json",
        payTo=ADDRESS,
        maxTimeoutSeconds=60,
        asset=ADDRESS,
    )
    return PaymentContext(payload, terms, verified, datetime(1970, 1, 1, tzinfo=UTC))


def test_always_verify_follows_verification() -> None:
    assert AlwaysVerifyPolicy().decide(context(10)).action is Action.APPROVE
    assert AlwaysVerifyPolicy().decide(context(10, verified=False)).action is Action.REJECT


def test_ask_above_threshold() -> None:
    policy = AskAbovePolicy(10)
    assert policy.decide(context(10)).action is Action.APPROVE
    assert policy.decide(context(11)).action is Action.ASK
