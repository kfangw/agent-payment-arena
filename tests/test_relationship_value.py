"""Tests for relationship value iteration and likelihood lattices."""

import math

import numpy as np
import pytest

from arena.experiments.relationship.model import RelationshipAction, RelationshipParameters
from arena.experiments.relationship.value import (
    ValueSolverConfig,
    analytic_pay_exit_boundary,
    check_pay_exit_e0,
    finite_lattice_exit_probability,
    immediate_action_payoffs,
    require_valid_payoff_convention,
    solve_value,
)


def _parameters(**updates: float | int) -> RelationshipParameters:
    values: dict[str, float | int] = {
        "v": 1.0,
        "u": 0.2,
        "alpha": 0.02,
        "beta": 0.10,
        "bad_cheat_rate": 1.0,
        "verification_cost": 3.0,
        "escrow_cost_rate": 0.05,
        "prior_bad": 0.05,
        "continuation_probability": 0.80,
        "public_sampling_rate": 0.05,
        "deposit": 10.0,
        "max_transactions": 100,
    }
    values.update(updates)
    return RelationshipParameters(**values)  # type: ignore[arg-type]


def test_immediate_payoffs_match_type_conditional_endpoints() -> None:
    pay, verify, defer = immediate_action_payoffs(
        _parameters(),
        np.asarray([0.0, 1.0]),
    )

    assert pay == pytest.approx([0.051, -1.105])
    assert verify == pytest.approx([-2.78, -3.1])
    assert defer == pytest.approx([-2.854, -3.15])


def test_value_iteration_converges_and_respects_endpoints() -> None:
    solution = solve_value(
        _parameters(),
        ValueSolverConfig(grid_spacing=0.002, tolerance=1e-11),
    )

    assert solution.residual <= 1e-11
    assert solution.values[0] == pytest.approx(0.051 / 0.2, abs=1e-9)
    assert solution.values[-1] == 0
    assert solution.actions[0] is RelationshipAction.PAY
    assert solution.actions[-1] is RelationshipAction.EXIT
    assert solution.exit_boundary > 0.05
    assert solution.beliefs.flags.writeable is False


def test_e0_boundary_matches_analytic_pay_exit_solution() -> None:
    check = check_pay_exit_e0(
        _parameters(continuation_probability=0.98),
        ValueSolverConfig(grid_spacing=0.001, tolerance=1e-11),
    )

    assert check.passed
    assert check.solution.exit_boundary == pytest.approx(
        check.analytic_exit_boundary,
        abs=0.002,
    )
    assert check.analytic_exit_boundary == pytest.approx(0.12574, abs=0.002)


def test_lattice_bounds_contain_small_enumerated_tree() -> None:
    relative = -0.8
    failure_probability = 0.3
    survival = 0.7
    alpha = 0.1
    beta = 0.2
    cheat_rate = 1.0
    max_passes = 0
    bounds = finite_lattice_exit_probability(
        relative_logit=relative,
        failure_probability=failure_probability,
        alpha=alpha,
        beta=beta,
        cheat_rate=cheat_rate,
        survival_to_next_check=survival,
        max_passes=max_passes,
    )

    delta_failure = math.log((1 - beta) / alpha)
    failures_to_exit = math.ceil(-relative / delta_failure)
    enumerated_lower = (survival * failure_probability) ** failures_to_exit

    assert bounds.lower == pytest.approx(enumerated_lower)
    assert bounds.lower <= bounds.upper <= 1


def test_zero_false_positive_lattice_reduces_to_first_failure() -> None:
    bounds = finite_lattice_exit_probability(
        relative_logit=-1.0,
        failure_probability=0.4,
        alpha=0.0,
        beta=0.2,
        cheat_rate=1.0,
        survival_to_next_check=0.5,
        max_passes=3,
    )

    assert bounds.lower == bounds.upper == pytest.approx(2 / 7)


def test_negative_payoff_conventions_are_rejected() -> None:
    parameters = _parameters()

    require_valid_payoff_convention(
        parameters,
        deferred_failure_compensation=0.0,
        prepaid_failure_compensation=1.0,
    )
    with pytest.raises(ValueError, match="deferred"):
        require_valid_payoff_convention(
            parameters,
            deferred_failure_compensation=1.0,
            prepaid_failure_compensation=1.0,
        )
    with pytest.raises(ValueError, match="exceed"):
        require_valid_payoff_convention(
            parameters,
            deferred_failure_compensation=0.0,
            prepaid_failure_compensation=10.0,
        )


def test_closed_form_reduces_to_myopic_boundary_without_continuation() -> None:
    boundary = analytic_pay_exit_boundary(_parameters(continuation_probability=0.0))

    assert boundary == pytest.approx(0.051 / (0.051 + 1.105))
