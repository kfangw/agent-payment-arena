"""End-to-end tests for the deterministic evaluation runtime."""

from datetime import UTC, datetime

from eth_account import Account

from arena.agents.scripted import ContentFollowingAgent, ScriptedAgent
from arena.gateway.fake import FakeGateway
from arena.gateway.schemas import Mandate
from arena.gateway.signatures import sign_mandate
from arena.loop import run_scenario
from arena.payment import PaymentAuthority
from arena.scenarios import Scenario, minimum_suite

MERCHANT = "0x" + "11" * 20
ATTACKER = "0x" + "22" * 20
TOKEN = "0x" + "33" * 20
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def authority(scenario: Scenario) -> PaymentAuthority:
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
        pay_to=scenario.resource.payee,
        asset=TOKEN,
        price=scenario.resource.amount,
    )
    return PaymentAuthority(
        backend,
        sign_mandate(mandate, delegator.key.hex(), 1337),
        agent.key.hex(),
        chain_id=1337,
        token_name="KRW Test Stablecoin",
        token_version="1",
    )


def test_scripted_agent_completes_benign_task_and_blocks_attack() -> None:
    benign, attack = minimum_suite(MERCHANT, ATTACKER)
    agent = ScriptedAgent(frozenset({MERCHANT}), 25)

    assert run_scenario(benign, agent, authority(benign), now=NOW).task_completed
    blocked = run_scenario(attack, agent, authority(attack), now=NOW)
    assert not blocked.task_completed
    assert blocked.events == ()


def test_content_following_agent_executes_injected_payment() -> None:
    _, attack = minimum_suite(MERCHANT, ATTACKER)
    trace = run_scenario(attack, ContentFollowingAgent(), authority(attack), now=NOW)

    assert trace.task_completed
    assert trace.events[0].payee == ATTACKER
    assert trace.events[0].result.settled
