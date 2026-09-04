"""Contract tests for x402 wire models."""

import pytest
from pydantic import ValidationError

from arena.gateway.schemas import Mandate, PaymentRequirements

ALICE = "0x" + "11" * 20
BOB = "0x" + "22" * 20
TOKEN = "0x" + "33" * 20
MANDATE_ID = "0x" + "44" * 32


def test_payment_requirements_emit_reference_wire_names() -> None:
    terms = PaymentRequirements(
        network="eip155:1337",
        maxAmountRequired="25",
        resource="https://merchant.test/item",
        description="item",
        mimeType="application/json",
        payTo=BOB,
        maxTimeoutSeconds=60,
        asset=TOKEN,
    )

    assert terms.model_dump(by_alias=True)["maxAmountRequired"] == "25"
    assert terms.model_dump(by_alias=True)["payTo"] == BOB


def test_mandate_round_trips_reference_wire_shape() -> None:
    raw = {
        "delegator": ALICE,
        "agent": BOB,
        "maxAmountPerPayment": "100",
        "allowedPayees": [TOKEN],
        "allowedResources": ["https://merchant.test/"],
        "validAfter": "10",
        "validBefore": "20",
        "budgetAmount": "500",
        "budgetWindowSeconds": "3600",
        "maxPaymentsPerWindow": "5",
        "rateWindowSeconds": "60",
        "mandateId": MANDATE_ID,
    }

    assert Mandate.model_validate(raw).model_dump(by_alias=True, mode="json") == raw


@pytest.mark.parametrize("field", ["delegator", "agent"])
def test_mandate_rejects_invalid_addresses(field: str) -> None:
    raw = {
        "delegator": ALICE,
        "agent": BOB,
        "maxAmountPerPayment": "100",
        "validAfter": "0",
        "validBefore": "20",
        "budgetAmount": "0",
        "budgetWindowSeconds": "0",
        "maxPaymentsPerWindow": "0",
        "rateWindowSeconds": "0",
        "mandateId": MANDATE_ID,
    }
    raw[field] = "invalid"

    with pytest.raises(ValidationError):
        Mandate.model_validate(raw)
