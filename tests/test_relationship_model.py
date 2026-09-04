"""Tests for stable repeated-relationship identifiers."""

import pytest

from arena.experiments.relationship.model import (
    POLICY_STREAM_IDS,
    RelationshipAction,
    Verdict,
    policy_stream_id,
)


def test_serialized_action_and_verdict_values_are_stable() -> None:
    assert [action.value for action in RelationshipAction] == [
        "pay",
        "verify",
        "defer",
        "exit",
    ]
    assert [verdict.value for verdict in Verdict] == [
        "pass",
        "fail",
        "not_checked",
        "pending",
    ]


def test_policy_stream_ids_are_unique_and_permanent() -> None:
    assert POLICY_STREAM_IDS["compiled_deposit_exit"] == 1
    assert POLICY_STREAM_IDS["no_deposit_optimum"] == 9
    assert len(POLICY_STREAM_IDS.values()) == len(set(POLICY_STREAM_IDS.values()))
    assert policy_stream_id("wald_sprt") == 5


def test_unknown_policy_has_no_implicit_random_stream() -> None:
    with pytest.raises(ValueError, match="no stable stream id"):
        policy_stream_id("unregistered")
