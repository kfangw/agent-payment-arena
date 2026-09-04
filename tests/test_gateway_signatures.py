"""Tests for EIP-712 mandate and confirmation binding."""

from eth_account import Account

from arena.gateway.schemas import AskRequest, Mandate
from arena.gateway.signatures import (
    sign_confirmation,
    sign_mandate,
    verify_confirmation,
    verify_mandate,
)

AGENT = "0x" + "22" * 20
PAYEE = "0x" + "33" * 20


def mandate(delegator: str) -> Mandate:
    return Mandate(
        delegator=delegator,
        agent=AGENT,
        maxAmountPerPayment="100",
        allowedPayees=(PAYEE,),
        allowedResources=("https://merchant.test/",),
        validAfter="0",
        validBefore="2000000000",
        budgetAmount="500",
        budgetWindowSeconds="3600",
        maxPaymentsPerWindow="5",
        rateWindowSeconds="60",
        mandateId="0x" + "44" * 32,
    )


def test_mandate_signature_rejects_tampering() -> None:
    account = Account.create("delegator")
    original = sign_mandate(mandate(account.address), account.key.hex(), 1337)

    assert verify_mandate(original, 1337)
    tampered = original.model_copy(
        update={"mandate": original.mandate.model_copy(update={"max_amount_per_payment": "101"})}
    )
    assert not verify_mandate(tampered, 1337)


def test_confirmation_is_bound_to_payment_nonce_and_amount() -> None:
    account = Account.create("delegator")
    request = AskRequest(
        mandateId="0x" + "44" * 32,
        authorizationNonce="0x" + "55" * 32,
        amount="150",
        resource="https://merchant.test/item",
    )
    confirmation = sign_confirmation(request, 2_000_000_000, account.key.hex(), 1337)

    assert verify_confirmation(confirmation, account.address, 1337)
    assert not verify_confirmation(
        confirmation.model_copy(update={"amount": "151"}), account.address, 1337
    )
