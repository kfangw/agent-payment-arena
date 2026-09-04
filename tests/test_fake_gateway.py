"""Contract tests for the in-memory payment gateway."""

from datetime import UTC, datetime

from eth_account import Account

from arena.gateway.contract import Action, ErrorCode
from arena.gateway.fake import FakeGateway
from arena.gateway.schemas import Authorization, ExactPayload, Mandate, PaymentPayload
from arena.gateway.signatures import sign_authorization, sign_mandate

PAYEE = "0x" + "22" * 20
TOKEN = "0x" + "33" * 20
RESOURCE = "https://merchant.test/item"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def gateway() -> FakeGateway:
    return FakeGateway(network="eip155:1337", chain_id=1337, pay_to=PAYEE, asset=TOKEN, price=25)


def payment(*, amount: int = 25, nonce_byte: str = "55") -> PaymentPayload:
    delegator = Account.from_key("0x" + "01" * 32)
    agent = Account.from_key("0x" + "02" * 32)
    mandate = Mandate(
        delegator=delegator.address,
        agent=agent.address,
        maxAmountPerPayment="100",
        allowedPayees=(PAYEE,),
        allowedResources=("https://merchant.test/",),
        validAfter="0",
        validBefore="2000000000",
        budgetAmount="100",
        budgetWindowSeconds="3600",
        maxPaymentsPerWindow="3",
        rateWindowSeconds="60",
        mandateId="0x" + "44" * 32,
    )
    authorization = Authorization(
        **{
            "from": agent.address,
            "to": PAYEE,
            "value": str(amount),
            "validAfter": "0",
            "validBefore": "2000000000",
            "nonce": "0x" + nonce_byte * 32,
        }
    )
    signature = sign_authorization(
        authorization,
        agent.key.hex(),
        token_name="KRW Test Stablecoin",
        token_version="1",
        chain_id=1337,
        token_address=TOKEN,
    )
    return PaymentPayload(
        x402Version=1,
        network="eip155:1337",
        payload=ExactPayload(signature=signature, authorization=authorization),
        mandate=sign_mandate(mandate, delegator.key.hex(), 1337),
    )


def test_valid_payment_settles_once() -> None:
    backend = gateway()
    payload = payment()

    assert backend.submit(payload, resource=RESOURCE, now=NOW).settled
    replay = backend.submit(payload, resource=RESOURCE, now=NOW)
    assert replay.action is Action.REJECT
    assert replay.error_code is ErrorCode.VERIFICATION_FAILED


def test_tampered_authorization_is_rejected() -> None:
    backend = gateway()
    payload = payment().model_copy(
        update={
            "payload": payment().payload.model_copy(
                update={
                    "authorization": payment().payload.authorization.model_copy(
                        update={"value": "26"}
                    )
                }
            )
        }
    )

    result = backend.submit(payload, resource=RESOURCE, now=NOW)
    assert result.error_code is ErrorCode.VERIFICATION_FAILED


def test_missing_mandate_is_rejected() -> None:
    backend = gateway()
    payload = payment().model_copy(update={"mandate": None})

    result = backend.submit(payload, resource=RESOURCE, now=NOW)
    assert result.error_code is ErrorCode.MANDATE_MISSING


def test_resource_outside_scope_is_rejected() -> None:
    result = gateway().submit(payment(), resource="https://attacker.test/item", now=NOW)
    assert result.error_code is ErrorCode.MANDATE_EXCEEDED
