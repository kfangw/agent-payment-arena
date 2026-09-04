"""Tests for payment-bound delegator escalation."""

from datetime import UTC, datetime

from eth_account import Account

from arena.delegator.model import SigningDelegator
from arena.gateway.contract import Action
from arena.gateway.fake import FakeGateway
from arena.gateway.schemas import Mandate
from arena.gateway.signatures import sign_mandate
from arena.payment import PaymentAuthority
from arena.policies.payment import AskAbovePolicy
from arena.scenarios import Scenario, minimum_suite
from tests.test_evaluation_loop import ATTACKER, TOKEN

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def authority(scenario: Scenario) -> PaymentAuthority:
    resource = scenario.resource
    delegator = Account.from_key("0x" + "01" * 32)
    agent = Account.from_key("0x" + "02" * 32)
    mandate = Mandate(
        delegator=delegator.address,
        agent=agent.address,
        maxAmountPerPayment="100",
        allowedPayees=(),
        allowedResources=("https://merchant.test/",),
        validAfter="0",
        validBefore="2000000000",
        budgetAmount="1000",
        budgetWindowSeconds="3600",
        maxPaymentsPerWindow="20",
        rateWindowSeconds="60",
        mandateId="0x" + "44" * 32,
    )
    backend = FakeGateway(
        network="eip155:1337",
        chain_id=1337,
        pay_to=resource.payee,
        asset=TOKEN,
        price=resource.amount,
        policy=AskAbovePolicy(20),
    )
    return PaymentAuthority(
        backend,
        sign_mandate(mandate, delegator.key.hex(), 1337),
        agent.key.hex(),
        chain_id=1337,
        token_name="KRW Test Stablecoin",
        token_version="1",
    )


def test_authority_retries_asked_payment_with_confirmation() -> None:
    _, attack = minimum_suite("0x" + "11" * 20, ATTACKER)
    payment_authority = authority(attack)
    payment_authority.delegator = SigningDelegator("0x" + "01" * 32, 1337)

    results = payment_authority.pay_with_escalation(
        attack.resource.url,
        ATTACKER,
        25,
        now=NOW,
        nonce_key="escalated",
    )

    assert tuple(result.action for result in results) == (Action.ASK, Action.APPROVE)
    assert results[-1].settled


def test_confirmation_cannot_be_reused_for_another_payment() -> None:
    _, attack = minimum_suite("0x" + "11" * 20, ATTACKER)
    payment_authority = authority(attack)
    delegator = SigningDelegator("0x" + "01" * 32, 1337)
    confirmation = delegator.confirm(
        payment_authority.ask_request(attack.resource.url, 25, "first"), now=NOW
    )

    result = payment_authority.pay(
        attack.resource.url,
        ATTACKER,
        25,
        now=NOW,
        nonce_key="second",
        confirmation=confirmation,
    )

    assert not result.settled
