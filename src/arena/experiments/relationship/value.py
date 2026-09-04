"""Numerical value and likelihood-lattice kernels for relationship policies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arena.experiments.relationship.belief import bad_failure_probability, update_belief
from arena.experiments.relationship.model import (
    RelationshipAction,
    RelationshipParameters,
    Verdict,
)


@dataclass(frozen=True)
class ValueSolverConfig:
    """Numerical controls fixed by the evaluation protocol."""

    grid_spacing: float = 0.001
    tolerance: float = 1e-10
    max_iterations: int = 100_000

    def __post_init__(self) -> None:
        """Reject controls that cannot define a finite solver."""
        if not 0 < self.grid_spacing <= 1:
            raise ValueError("grid_spacing must be in (0, 1]")
        reciprocal = 1 / self.grid_spacing
        if not math.isclose(reciprocal, round(reciprocal), abs_tol=1e-10):
            raise ValueError("grid_spacing must divide the unit interval")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class ActionRegion:
    """One maximal interval carrying the same grid action."""

    action: RelationshipAction
    lower: float
    upper: float


@dataclass(frozen=True)
class ValueSolution:
    """A converged Bellman solution and its deterministic action map."""

    beliefs: NDArray[np.float64]
    values: NDArray[np.float64]
    actions: tuple[RelationshipAction, ...]
    residual: float
    iterations: int
    regions: tuple[ActionRegion, ...]

    def __post_init__(self) -> None:
        """Keep returned numerical arrays immutable."""
        self.beliefs.flags.writeable = False
        self.values.flags.writeable = False

    @property
    def exit_boundary(self) -> float:
        """Return the first belief at which exit is selected."""
        for belief, action in zip(self.beliefs, self.actions, strict=True):
            if action is RelationshipAction.EXIT:
                return float(belief)
        raise ValueError("solution has no exit region")


@dataclass(frozen=True)
class LatticeBounds:
    """Lower and upper exit-probability bounds from a finite lattice."""

    lower: float
    upper: float
    max_passes: int

    @property
    def width(self) -> float:
        """Return the unresolved probability interval."""
        return self.upper - self.lower


@dataclass(frozen=True)
class E0ValueCheck:
    """The numerical quantities used by the E0 value acceptance gate."""

    solution: ValueSolution
    analytic_exit_boundary: float
    boundary_error: float
    boundary_tolerance: float
    residual_tolerance: float

    @property
    def passed(self) -> bool:
        """Return whether convergence and boundary agreement both pass."""
        return (
            self.solution.residual <= self.residual_tolerance
            and self.boundary_error <= self.boundary_tolerance
        )


DEFAULT_VALUE_SOLVER_CONFIG = ValueSolverConfig()


def solve_value(
    parameters: RelationshipParameters,
    config: ValueSolverConfig = DEFAULT_VALUE_SOLVER_CONFIG,
) -> ValueSolution:
    """Solve the infinite-horizon buyer Bellman equation on a uniform grid."""
    size = round(1 / config.grid_spacing) + 1
    beliefs = np.linspace(0.0, 1.0, size, dtype=np.float64)
    values = np.zeros_like(beliefs)

    for iteration in range(1, config.max_iterations + 1):
        action_values = bellman_action_values(parameters, beliefs, values)
        updated, actions = _select_actions(action_values)
        residual = float(np.max(np.abs(updated - values)))
        if not np.isfinite(residual) or not np.all(np.isfinite(updated)):
            raise ArithmeticError("value iteration produced a nonfinite value")
        values = updated
        if residual <= config.tolerance:
            final_action_values = bellman_action_values(parameters, beliefs, values)
            final_values, final_actions = _select_actions(final_action_values)
            final_residual = float(np.max(np.abs(final_values - values)))
            return ValueSolution(
                beliefs=beliefs,
                values=values,
                actions=final_actions,
                residual=final_residual,
                iterations=iteration,
                regions=_regions(beliefs, final_actions),
            )
    raise RuntimeError("value iteration reached max_iterations without convergence")


def bellman_action_values(
    parameters: RelationshipParameters,
    beliefs: NDArray[np.float64],
    values: NDArray[np.float64],
) -> dict[RelationshipAction, NDArray[np.float64]]:
    """Return all four Bellman action values on one belief grid."""
    if beliefs.ndim != 1 or values.shape != beliefs.shape or len(beliefs) < 2:
        raise ValueError("beliefs and values must be aligned one-dimensional grids")
    if not np.all(np.diff(beliefs) > 0) or beliefs[0] != 0 or beliefs[-1] != 1:
        raise ValueError("belief grid must increase from zero to one")

    bad_failure = bad_failure_probability(
        parameters.alpha,
        parameters.beta,
        parameters.bad_cheat_rate,
    )
    failure = beliefs * bad_failure + (1 - beliefs) * parameters.alpha
    failed_belief = _posterior_grid(beliefs, bad_failure, parameters.alpha, failure)
    passed_belief = _posterior_grid(
        beliefs,
        1 - bad_failure,
        1 - parameters.alpha,
        1 - failure,
    )
    observed_value = failure * np.interp(failed_belief, beliefs, values) + (
        1 - failure
    ) * np.interp(passed_belief, beliefs, values)

    pay, verify, defer = immediate_action_payoffs(parameters, beliefs)
    continuation = parameters.continuation_probability
    sampling = parameters.public_sampling_rate
    return {
        RelationshipAction.PAY: pay
        + continuation * ((1 - sampling) * values + sampling * observed_value),
        RelationshipAction.VERIFY: verify + continuation * observed_value,
        RelationshipAction.DEFER: defer + continuation * observed_value,
        RelationshipAction.EXIT: np.zeros_like(beliefs),
    }


def immediate_action_payoffs(
    parameters: RelationshipParameters,
    beliefs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return pay, prepaid-verify, and deferred-payment one-step payoffs."""
    cheat_rate = parameters.bad_cheat_rate
    cheated_probability = beliefs * cheat_rate
    bad_failure = bad_failure_probability(
        parameters.alpha,
        parameters.beta,
        cheat_rate,
    )
    failure = beliefs * bad_failure + (1 - beliefs) * parameters.alpha
    compensation = min(parameters.deposit, parameters.v)

    base_delivery = parameters.u * parameters.v * (1 - cheated_probability)
    base_delivery -= parameters.v * cheated_probability
    verify = base_delivery - parameters.verification_cost + compensation * failure
    pay = base_delivery - parameters.public_sampling_rate * parameters.verification_cost
    pay += parameters.public_sampling_rate * compensation * failure
    defer = (
        -parameters.escrow_cost_rate * parameters.v
        - parameters.verification_cost
        + (1 - cheated_probability) * (1 - parameters.alpha) * parameters.u * parameters.v
        - cheated_probability * parameters.beta * parameters.v
    )
    return pay, verify, defer


def action_at_belief(
    parameters: RelationshipParameters,
    solution: ValueSolution,
    belief: float,
) -> RelationshipAction:
    """Choose an action at an off-grid belief using linear value interpolation."""
    if not 0 <= belief <= 1:
        raise ValueError("belief must be between zero and one")
    bad_failure = bad_failure_probability(
        parameters.alpha,
        parameters.beta,
        parameters.bad_cheat_rate,
    )
    failure = belief * bad_failure + (1 - belief) * parameters.alpha
    failed_belief = update_belief(
        belief,
        Verdict.FAIL,
        alpha=parameters.alpha,
        beta=parameters.beta,
        cheat_rate=parameters.bad_cheat_rate,
    )
    passed_belief = update_belief(
        belief,
        Verdict.PASS,
        alpha=parameters.alpha,
        beta=parameters.beta,
        cheat_rate=parameters.bad_cheat_rate,
    )
    current_value = float(np.interp(belief, solution.beliefs, solution.values))
    observed_value = failure * float(
        np.interp(failed_belief, solution.beliefs, solution.values)
    ) + (1 - failure) * float(np.interp(passed_belief, solution.beliefs, solution.values))
    pay, verify, defer = immediate_action_payoffs(parameters, np.asarray([belief]))
    continuation = parameters.continuation_probability
    sampling = parameters.public_sampling_rate
    scalar_values = {
        RelationshipAction.PAY: float(pay[0])
        + continuation * ((1 - sampling) * current_value + sampling * observed_value),
        RelationshipAction.VERIFY: float(verify[0]) + continuation * observed_value,
        RelationshipAction.DEFER: float(defer[0]) + continuation * observed_value,
        RelationshipAction.EXIT: 0.0,
    }
    return max(
        _ACTION_TIE_ORDER,
        key=lambda action: (scalar_values[action], -_ACTION_TIE_ORDER.index(action)),
    )


def finite_lattice_exit_probability(
    *,
    relative_logit: float,
    failure_probability: float,
    alpha: float,
    beta: float,
    cheat_rate: float,
    survival_to_next_check: float,
    max_passes: int,
) -> LatticeBounds:
    """Bound exit probability using the exact finite likelihood lattice."""
    if relative_logit >= 0:
        return LatticeBounds(1.0, 1.0, max_passes)
    for name, value in (
        ("failure_probability", failure_probability),
        ("survival_to_next_check", survival_to_next_check),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if max_passes < 0:
        raise ValueError("max_passes must be nonnegative")
    bad_failure = bad_failure_probability(alpha, beta, cheat_rate)
    if alpha == 0:
        numerator = survival_to_next_check * failure_probability
        denominator = 1 - survival_to_next_check * (1 - failure_probability)
        exact = numerator / denominator if denominator > 0 else 0.0
        return LatticeBounds(exact, exact, max_passes)
    delta_failure = math.log(bad_failure / alpha)
    if bad_failure == 1:
        failures_to_exit = math.ceil(-relative_logit / delta_failure)
        exact = (survival_to_next_check * failure_probability) ** failures_to_exit
        return LatticeBounds(exact, exact, max_passes)
    delta_pass = math.log((1 - bad_failure) / (1 - alpha))

    def evaluate(tail_value: float) -> float:
        cache: dict[tuple[int, int], float] = {}

        def visit(failures: int, passes: int) -> float:
            position = relative_logit + failures * delta_failure + passes * delta_pass
            if position >= 0:
                return 1.0
            if passes > max_passes:
                return tail_value
            key = (failures, passes)
            if key not in cache:
                cache[key] = survival_to_next_check * (
                    failure_probability * visit(failures + 1, passes)
                    + (1 - failure_probability) * visit(failures, passes + 1)
                )
            return cache[key]

        return visit(0, 0)

    return LatticeBounds(evaluate(0.0), evaluate(1.0), max_passes)


def analytic_pay_exit_boundary(
    parameters: RelationshipParameters,
    *,
    max_passes: int = 200,
) -> float:
    """Return the closed-form boundary for a pay-or-exit policy."""
    sampling = parameters.public_sampling_rate
    continuation = parameters.continuation_probability
    bad_failure = bad_failure_probability(
        parameters.alpha,
        parameters.beta,
        parameters.bad_cheat_rate,
    )
    compensation = min(parameters.deposit, parameters.v)
    honest_payoff = (
        parameters.u * parameters.v
        - sampling * parameters.verification_cost
        + sampling * compensation * parameters.alpha
    )
    bad_payoff = (
        parameters.u * parameters.v * (1 - parameters.bad_cheat_rate)
        - parameters.v * parameters.bad_cheat_rate
        - sampling * parameters.verification_cost
        + sampling * compensation * bad_failure
    )
    if not honest_payoff > 0 >= bad_payoff:
        raise ValueError("pay-or-exit boundary requires positive honest and nonpositive bad payoff")
    if sampling == 0 or continuation == 0:
        return honest_payoff / (honest_payoff - bad_payoff)
    survival = sampling * continuation / (1 - continuation * (1 - sampling))
    delta_pass = math.log((1 - bad_failure) / (1 - parameters.alpha))
    honest_exit = finite_lattice_exit_probability(
        relative_logit=delta_pass,
        failure_probability=parameters.alpha,
        alpha=parameters.alpha,
        beta=parameters.beta,
        cheat_rate=parameters.bad_cheat_rate,
        survival_to_next_check=survival,
        max_passes=max_passes,
    )
    bad_exit = finite_lattice_exit_probability(
        relative_logit=delta_pass,
        failure_probability=bad_failure,
        alpha=parameters.alpha,
        beta=parameters.beta,
        cheat_rate=parameters.bad_cheat_rate,
        survival_to_next_check=survival,
        max_passes=max_passes,
    )
    honest_factor = 1 + continuation * sampling * (1 - parameters.alpha) * (
        1 - honest_exit.lower
    ) / (1 - continuation)
    bad_factor = 1 + continuation * sampling * (1 - bad_failure) * (1 - bad_exit.lower) / (
        1 - continuation
    )
    return honest_payoff * honest_factor / (honest_payoff * honest_factor - bad_payoff * bad_factor)


def check_pay_exit_e0(
    parameters: RelationshipParameters,
    config: ValueSolverConfig = DEFAULT_VALUE_SOLVER_CONFIG,
) -> E0ValueCheck:
    """Compare value iteration with the analytic pay-or-exit boundary."""
    solution = solve_value(parameters, config)
    analytic = analytic_pay_exit_boundary(parameters)
    boundary_tolerance = 2 * config.grid_spacing
    return E0ValueCheck(
        solution=solution,
        analytic_exit_boundary=analytic,
        boundary_error=abs(solution.exit_boundary - analytic),
        boundary_tolerance=boundary_tolerance,
        residual_tolerance=config.tolerance,
    )


def require_valid_payoff_convention(
    parameters: RelationshipParameters,
    *,
    deferred_failure_compensation: float,
    prepaid_failure_compensation: float,
) -> None:
    """Reject the two double-credit payoff conventions used as E0 controls."""
    if deferred_failure_compensation != 0:
        raise ValueError("deferred failure must not receive loss compensation")
    if prepaid_failure_compensation > min(parameters.deposit, parameters.v):
        raise ValueError("prepaid failure compensation cannot exceed exposure")


def _posterior_grid(
    beliefs: NDArray[np.float64],
    bad_likelihood: float,
    honest_likelihood: float,
    predictive_likelihood: NDArray[np.float64],
) -> NDArray[np.float64]:
    numerator = beliefs * bad_likelihood
    return np.divide(
        numerator,
        predictive_likelihood,
        out=beliefs.copy(),
        where=predictive_likelihood > 0,
    )


def _select_actions(
    action_values: dict[RelationshipAction, NDArray[np.float64]],
) -> tuple[NDArray[np.float64], tuple[RelationshipAction, ...]]:
    stacked = np.stack([action_values[action] for action in _ACTION_TIE_ORDER])
    indices = np.argmax(stacked, axis=0)
    selected = stacked[indices, np.arange(stacked.shape[1])]
    return selected, tuple(_ACTION_TIE_ORDER[index] for index in indices)


def _regions(
    beliefs: NDArray[np.float64],
    actions: tuple[RelationshipAction, ...],
) -> tuple[ActionRegion, ...]:
    regions: list[ActionRegion] = []
    start = 0
    for index in range(1, len(actions) + 1):
        if index == len(actions) or actions[index] is not actions[start]:
            regions.append(
                ActionRegion(actions[start], float(beliefs[start]), float(beliefs[index - 1]))
            )
            start = index
    return tuple(regions)


_ACTION_TIE_ORDER = (
    RelationshipAction.EXIT,
    RelationshipAction.DEFER,
    RelationshipAction.VERIFY,
    RelationshipAction.PAY,
)
