"""Tests for seeded delegator error, delay, non-response, and fatigue."""

from datetime import UTC, datetime

from arena.delegator.model import BehavioralDelegator
from arena.evaluation import DELEGATOR_KEY, run_attack_suite
from arena.gateway.schemas import AskRequest

NOW = datetime(2026, 1, 1, tzinfo=UTC)
REQUEST = AskRequest(
    mandateId="0x" + "11" * 32,
    authorizationNonce="0x" + "22" * 32,
    amount="25",
    resource="https://merchant.test/report",
)


def test_nonresponse_and_delay_are_explicit_and_reproducible() -> None:
    first = BehavioralDelegator(
        DELEGATOR_KEY, 1337, seed=4, nonresponse_probability=1, base_latency_ms=250
    )
    second = BehavioralDelegator(
        DELEGATOR_KEY, 1337, seed=4, nonresponse_probability=1, base_latency_ms=250
    )
    assert first.confirm(REQUEST, now=NOW) is None
    assert second.confirm(REQUEST, now=NOW) is None
    assert first.last_latency_ms == second.last_latency_ms == 250


def test_fatigue_reduces_approval_after_repeated_questions() -> None:
    delegator = BehavioralDelegator(
        DELEGATOR_KEY, 1337, seed=1, approval_probability=1, fatigue_per_question=1
    )
    assert delegator.confirm(REQUEST, now=NOW) is not None
    assert delegator.confirm(REQUEST, now=NOW) is None


def test_evaluation_prices_simulated_escalation_latency() -> None:
    result = run_attack_suite(
        1,
        delegator_factory=lambda seed: BehavioralDelegator(
            DELEGATOR_KEY, 1337, seed, base_latency_ms=100
        ),
    )
    asked = [record for record in result.records if record.policy_id == "ask-above-20"]
    assert any(record.metrics.escalation_latency_ms >= 100 for record in asked)
