"""Tests for evaluation trace emission without requiring an exporter."""

from collections.abc import Mapping
from datetime import UTC, datetime

from arena.agents.scripted import ContentFollowingAgent
from arena.loop import run_scenario
from arena.scenarios import minimum_suite
from tests.test_evaluation_loop import ATTACKER, MERCHANT, authority


class RecordingSink:
    """In-memory trace sink used to inspect emitted attributes."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, str | int | float | bool]]] = []

    def record(self, name: str, attributes: Mapping[str, str | int | float | bool]) -> None:
        """Retain a copy of one trace operation."""
        self.records.append((name, dict(attributes)))


def test_run_emits_payment_and_summary_operations() -> None:
    _, attack = minimum_suite(MERCHANT, ATTACKER)
    sink = RecordingSink()
    run_scenario(
        attack,
        ContentFollowingAgent(),
        authority(attack),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        trace_sink=sink,
    )
    assert [name for name, _ in sink.records] == ["arena.payment", "arena.run"]
    assert sink.records[0][1]["payment.settled"] is True
