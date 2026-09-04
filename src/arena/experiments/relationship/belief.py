"""Bayesian belief updates for a persistent binary seller type."""

from __future__ import annotations

import math

from arena.experiments.relationship.model import Verdict


def bad_failure_probability(alpha: float, beta: float, cheat_rate: float) -> float:
    """Return the failure probability conditional on a bad seller."""
    _validate_rates(alpha, beta, cheat_rate)
    return cheat_rate * (1 - beta) + (1 - cheat_rate) * alpha


def predictive_failure_probability(
    prior_bad: float,
    alpha: float,
    beta: float,
    cheat_rate: float,
) -> float:
    """Return the failure probability before observing a verdict."""
    _validate_probability("prior_bad", prior_bad)
    bad_failure = bad_failure_probability(alpha, beta, cheat_rate)
    return prior_bad * bad_failure + (1 - prior_bad) * alpha


def update_belief(
    prior_bad: float,
    verdict: Verdict,
    *,
    alpha: float,
    beta: float,
    cheat_rate: float,
) -> float:
    """Update the probability of a bad seller after one observable verdict."""
    _validate_probability("prior_bad", prior_bad)
    bad_failure = bad_failure_probability(alpha, beta, cheat_rate)
    if verdict in {Verdict.NOT_CHECKED, Verdict.PENDING}:
        return prior_bad
    if prior_bad in {0.0, 1.0}:
        return prior_bad

    if verdict is Verdict.FAIL:
        bad_likelihood = bad_failure
        honest_likelihood = alpha
    elif verdict is Verdict.PASS:
        bad_likelihood = 1 - bad_failure
        honest_likelihood = 1 - alpha
    else:
        raise AssertionError(f"unhandled verdict: {verdict!r}")

    if bad_likelihood == honest_likelihood == 0:
        raise ValueError(f"{verdict.value} has zero probability under both seller types")
    if bad_likelihood == 0:
        return 0.0
    if honest_likelihood == 0:
        return 1.0

    log_odds = (
        math.log(prior_bad)
        - math.log1p(-prior_bad)
        + math.log(bad_likelihood)
        - math.log(honest_likelihood)
    )
    if log_odds >= 0:
        inverse = math.exp(-log_odds)
        return 1 / (1 + inverse)
    odds = math.exp(log_odds)
    return odds / (1 + odds)


def _validate_rates(alpha: float, beta: float, cheat_rate: float) -> None:
    _validate_probability("alpha", alpha)
    _validate_probability("beta", beta)
    _validate_probability("cheat_rate", cheat_rate)
    if 1 - beta <= alpha:
        raise ValueError("predicate must satisfy 1 - beta > alpha")


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
