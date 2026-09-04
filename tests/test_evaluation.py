"""Tests for repeated evaluation execution."""

import pytest

from arena.evaluation import run_minimum_suite


def test_minimum_suite_runs_complete_matrix() -> None:
    result = run_minimum_suite(repetitions=2, seed=10)

    assert len(result.records) == 16
    assert {record.seed for record in result.records} == {10, 11}
    assert {record.policy_id for record in result.records} == {
        "always-verify",
        "ask-above-20",
    }
    vulnerable = [
        record
        for record in result.records
        if record.agent_id == "content-following"
        and record.scenario_id == "purchase-direct-injection"
        and record.policy_id == "always-verify"
    ]
    assert all(record.metrics.unauthorized_spend == 25 for record in vulnerable)


def test_repetitions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="repetitions must be positive"):
        run_minimum_suite(0)
