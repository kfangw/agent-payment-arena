"""Tests for persistent-seller belief updates."""

import pytest

from arena.experiments.relationship.belief import (
    bad_failure_probability,
    predictive_failure_probability,
    update_belief,
)
from arena.experiments.relationship.model import Verdict


def test_failure_probability_and_bayes_updates_match_hand_calculation() -> None:
    bad_failure = bad_failure_probability(alpha=0.02, beta=0.10, cheat_rate=1.0)

    assert bad_failure == pytest.approx(0.9)
    assert predictive_failure_probability(0.05, 0.02, 0.10, 1.0) == pytest.approx(0.064)
    assert update_belief(
        0.05,
        Verdict.FAIL,
        alpha=0.02,
        beta=0.10,
        cheat_rate=1.0,
    ) == pytest.approx(0.703125)
    assert update_belief(
        0.05,
        Verdict.PASS,
        alpha=0.02,
        beta=0.10,
        cheat_rate=1.0,
    ) == pytest.approx(0.005341880341880341)


@pytest.mark.parametrize("verdict", [Verdict.NOT_CHECKED, Verdict.PENDING])
def test_no_observation_leaves_belief_unchanged(verdict: Verdict) -> None:
    assert update_belief(0.37, verdict, alpha=0.02, beta=0.10, cheat_rate=1.0) == 0.37


def test_absorbing_prior_endpoints_remain_stable() -> None:
    for verdict in (Verdict.PASS, Verdict.FAIL):
        assert update_belief(0.0, verdict, alpha=0.02, beta=0.10, cheat_rate=1.0) == 0.0
        assert update_belief(1.0, verdict, alpha=0.02, beta=0.10, cheat_rate=1.0) == 1.0


def test_failed_evidence_raises_and_passed_evidence_lowers_bad_belief() -> None:
    prior = 0.4
    failed = update_belief(prior, Verdict.FAIL, alpha=0.02, beta=0.10, cheat_rate=1.0)
    passed = update_belief(prior, Verdict.PASS, alpha=0.02, beta=0.10, cheat_rate=1.0)

    assert passed < prior < failed


@pytest.mark.parametrize(
    ("alpha", "beta", "cheat_rate"),
    [(-0.1, 0.1, 1.0), (0.1, 1.1, 1.0), (0.1, 0.1, 1.1), (0.6, 0.4, 1.0)],
)
def test_invalid_or_uninformative_rates_are_rejected(
    alpha: float,
    beta: float,
    cheat_rate: float,
) -> None:
    with pytest.raises(ValueError):
        bad_failure_probability(alpha, beta, cheat_rate)
