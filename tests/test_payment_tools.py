"""Tests for the transport-independent MCP payment tools."""

from datetime import UTC, datetime

from arena.delegator.model import SigningDelegator
from arena.mcp_server.tools import PaymentTools
from arena.scenarios import minimum_suite
from tests.test_evaluation_loop import ATTACKER, authority


def test_tools_expose_terms_and_mandate_without_ground_truth() -> None:
    _, attack = minimum_suite("0x" + "11" * 20, ATTACKER)
    tools = PaymentTools(attack.resource, authority(attack), datetime(2026, 1, 1, tzinfo=UTC))

    fetched = tools.fetch_resource()
    assert fetched["content"] == attack.resource.content
    assert "ground_truth" not in fetched
    mandate = tools.inspect_mandate()["mandate"]
    assert isinstance(mandate, dict)
    assert mandate["agent"]


def test_pay_tool_uses_shared_payment_authority() -> None:
    _, attack = minimum_suite("0x" + "11" * 20, ATTACKER)
    tools = PaymentTools(attack.resource, authority(attack), datetime(2026, 1, 1, tzinfo=UTC))

    assert tools.pay(ATTACKER, 25, "tool-call").settled


def test_tools_complete_delegator_confirmation_flow() -> None:
    _, attack = minimum_suite("0x" + "11" * 20, ATTACKER)
    payment_authority = authority(attack)
    payment_authority.delegator = SigningDelegator("0x" + "01" * 32, 1337)
    tools = PaymentTools(attack.resource, payment_authority, datetime(2026, 1, 1, tzinfo=UTC))

    confirmation = tools.ask_delegator(25, "confirmed-tool-call")
    assert confirmation is not None
    assert tools.pay(ATTACKER, 25, "confirmed-tool-call", confirmation).settled
