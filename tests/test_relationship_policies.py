"""Tests for compiled repeated-relationship policies."""

import json

import numpy as np
import pytest

from arena.experiments.artifacts import parameter_hash
from arena.experiments.relationship.model import (
    PolicyContext,
    RelationshipAction,
    RelationshipObservation,
    RelationshipParameters,
    Verdict,
)
from arena.experiments.relationship.policies import compile_deposit_exit_policy
from arena.experiments.relationship.value import ValueSolverConfig


def _parameters() -> RelationshipParameters:
    return RelationshipParameters(
        v=1.0,
        u=0.2,
        alpha=0.02,
        beta=0.10,
        bad_cheat_rate=1.0,
        verification_cost=3.0,
        escrow_cost_rate=0.05,
        prior_bad=0.05,
        continuation_probability=0.98,
        public_sampling_rate=0.05,
        deposit=10.0,
        max_transactions=100,
    )


def _observation(belief: float) -> RelationshipObservation:
    return RelationshipObservation(
        transaction_index=0,
        belief_bad=belief,
        deposit=10.0,
        pending_verdicts=0,
        last_verdict=Verdict.NOT_CHECKED,
        cumulative_public_checks=0,
        cumulative_discretionary_checks=0,
    )


def test_compiled_policy_uses_public_belief_and_exit_boundary() -> None:
    parameters = _parameters()
    policy = compile_deposit_exit_policy(
        parameters,
        ValueSolverConfig(grid_spacing=0.001, tolerance=1e-11),
    )
    policy.reset(PolicyContext(parameters), np.random.default_rng(1))

    assert policy.act(_observation(0.05)) is RelationshipAction.PAY
    assert policy.act(_observation(policy.solution.exit_boundary)) is RelationshipAction.EXIT
    assert policy.act(_observation(0.9)) is RelationshipAction.EXIT


def test_policy_requires_reset_and_rejects_different_public_terms() -> None:
    policy = compile_deposit_exit_policy(
        _parameters(),
        ValueSolverConfig(grid_spacing=0.002, tolerance=1e-10),
    )

    with pytest.raises(RuntimeError, match="reset"):
        policy.act(_observation(0.05))
    changed = RelationshipParameters(**{**policy.parameters.__dict__, "deposit": 9.0})
    with pytest.raises(ValueError, match="do not match"):
        policy.reset(PolicyContext(changed), np.random.default_rng(2))


def test_snapshot_is_deterministic_complete_and_json_serializable() -> None:
    config = ValueSolverConfig(grid_spacing=0.002, tolerance=1e-10)
    first = compile_deposit_exit_policy(_parameters(), config).snapshot()
    second = compile_deposit_exit_policy(_parameters(), config).snapshot()

    assert first == second
    assert parameter_hash(first) == parameter_hash(second)
    assert first["policy_id"] == "compiled_deposit_exit"
    solution = first["solution"]
    parameters = first["parameters"]
    assert isinstance(solution, dict)
    assert isinstance(parameters, dict)
    assert solution["tie_order"] == ["exit", "defer", "verify", "pay"]
    assert parameters["public_sampling_rate"] == 0.05
    assert isinstance(solution["belief_grid"], list)
    assert isinstance(solution["value_grid"], list)
    assert isinstance(solution["action_grid"], list)
    assert len(solution["belief_grid"]) == len(solution["value_grid"])
    assert len(solution["belief_grid"]) == len(solution["action_grid"])
    json.dumps(first, sort_keys=True)


def test_off_grid_belief_is_evaluated_without_rounding() -> None:
    parameters = _parameters()
    policy = compile_deposit_exit_policy(
        parameters,
        ValueSolverConfig(grid_spacing=0.01, tolerance=1e-10),
    )
    policy.reset(PolicyContext(parameters), np.random.default_rng(3))

    below = policy.solution.exit_boundary - 0.0049
    above = policy.solution.exit_boundary + 0.0049
    assert policy.act(_observation(below)) is RelationshipAction.PAY
    assert policy.act(_observation(above)) is RelationshipAction.EXIT
