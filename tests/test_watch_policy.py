"""Regression tests for settlement observation policies."""

import numpy as np
from duel.core import GRANT, REJECT, VERIFY, WAIT
from duel.watch import (
    FixedActionWatchPolicy,
    WatchBandPolicy,
    block_sums,
    horizon_grid,
    tune_watch_policy,
)
from numpy.typing import NDArray


def test_horizon_grid_includes_zero_and_boundary() -> None:
    assert horizon_grid(5) == [0, 1, 2, 4, 5]


def test_watch_band_policy_waits_then_uses_thresholds() -> None:
    policy = WatchBandPolicy(horizon=2, lower=0.2, upper=0.8)

    assert policy(1, 1.0, 0.1) == WAIT
    assert policy(2, 1.0, 0.1) == GRANT
    assert policy(2, 1.0, 0.5) == VERIFY
    assert policy(2, 1.0, 0.9) == REJECT


def test_tuning_selects_best_horizon_and_thresholds() -> None:
    suspicion = np.array([0.1, 0.9])

    def payoffs(horizon: int, action: int) -> NDArray[np.float64]:
        bonus = float(horizon)
        if action == GRANT:
            return np.array([3.0 + bonus, 0.0])
        if action == REJECT:
            return np.array([0.0, 3.0 + bonus])
        return np.ones(2)

    best, value, rows = tune_watch_policy(
        payoffs,
        [(0.2, 0.8)],
        [0, 1],
        suspicion,
    )

    assert best == (1, 0.2, 0.8)
    assert value == 4.0
    assert len(rows) == 2


def test_block_sums_preserves_episode_boundaries() -> None:
    assert block_sums(np.array([1.0, 2.0, 4.0]), np.array([0, 0, 1])).tolist() == [3.0, 4.0]


def test_fixed_action_policy_waits_for_configured_horizon() -> None:
    policy = FixedActionWatchPolicy(horizon=1, action=REJECT)

    assert policy(0, 1.0, 0.5) == WAIT
    assert policy(1, 1.0, 0.5) == REJECT
