"""Policies that observe settlement for a bounded horizon before deciding."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import GRANT, REJECT, VERIFY, WAIT

DEFAULT_HORIZONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
PAYMENTS_PER_EPISODE = 50


def horizon_grid(horizon: int) -> list[int]:
    """Return the default observation horizons clipped to a cell boundary."""
    return sorted({value for value in DEFAULT_HORIZONS if value < horizon} | {0, horizon})


@dataclass
class WatchBandPolicy:
    """Wait for a fixed number of ticks, then apply a two-threshold rule."""

    horizon: int
    lower: float
    upper: float

    def __call__(self, stage: int, value: float, suspicion: float) -> int:
        if stage < self.horizon:
            return WAIT
        if suspicion < self.lower:
            return GRANT
        if suspicion > self.upper:
            return REJECT
        return VERIFY


@dataclass
class FixedActionWatchPolicy:
    """Wait for a fixed number of ticks, then take one action."""

    horizon: int
    action: int

    def __call__(self, stage: int, value: float, suspicion: float) -> int:
        return WAIT if stage < self.horizon else self.action


class OutageWatchBandPolicy:
    """Two-threshold observation policy for bounded outage episodes."""

    def __init__(
        self,
        horizon: int,
        lower: float,
        upper: float,
        episode_horizon: int,
        final_stage: int,
    ) -> None:
        self.horizon = horizon
        self.lower = lower
        self.upper = upper
        self.episode_horizon = episode_horizon
        self.final_stage = final_stage + 1

    def __call__(
        self, stage: int, remaining: int, response: bool, value: float, suspicion: float
    ) -> int:
        elapsed = self.episode_horizon - remaining
        if elapsed < self.horizon and stage < self.final_stage and remaining > 0:
            return WAIT
        if suspicion < self.lower:
            return GRANT
        if suspicion > self.upper:
            return REJECT
        return VERIFY


class FixedActionOutageWatchPolicy:
    """Fixed-action observation policy for bounded outage episodes."""

    def __init__(self, horizon: int, action: int, episode_horizon: int, final_stage: int) -> None:
        self.horizon = horizon
        self.action = action
        self.episode_horizon = episode_horizon
        self.final_stage = final_stage + 1

    def __call__(
        self, stage: int, remaining: int, response: bool, value: float, suspicion: float
    ) -> int:
        elapsed = self.episode_horizon - remaining
        if elapsed < self.horizon and stage < self.final_stage and remaining > 0:
            return WAIT
        return self.action


def tune_watch_policy(
    force_payoffs: Callable[[int, int], NDArray[np.float64]],
    threshold_grid: Sequence[tuple[float, float]],
    horizons: Sequence[int],
    suspicion: ArrayLike,
) -> tuple[tuple[int, float, float], float, list[dict[str, int | float]]]:
    """Select the best observation horizon and threshold pair."""
    if not threshold_grid or not horizons:
        raise ValueError("threshold grid and horizons must be nonempty")
    suspicion_values = np.asarray(suspicion, dtype=float)
    best: tuple[int, float, float] | None = None
    best_value = -np.inf
    rows: list[dict[str, int | float]] = []
    for horizon in horizons:
        grant = force_payoffs(horizon, GRANT)
        reject = force_payoffs(horizon, REJECT)
        verify = force_payoffs(horizon, VERIFY)
        horizon_best, horizon_value = None, -np.inf
        for lower, upper in threshold_grid:
            selected = np.where(
                suspicion_values < lower,
                grant,
                np.where(suspicion_values > upper, reject, verify),
            )
            value = float(selected.mean())
            if value > horizon_value:
                horizon_best, horizon_value = (lower, upper), value
        rows.append(
            {
                "k": int(horizon),
                "a": float(horizon_best[0]),
                "b": float(horizon_best[1]),
                "tune_mean": horizon_value,
            }
        )
        if horizon_value > best_value:
            best = (int(horizon), float(horizon_best[0]), float(horizon_best[1]))
            best_value = horizon_value
    if best is None:
        raise RuntimeError("tuning did not produce a candidate")
    return best, float(best_value), rows


def block_sums(values: ArrayLike, episodes: ArrayLike) -> NDArray[np.float64]:
    """Return one weighted sum for each contiguous episode identifier."""
    episode_ids = np.asarray(episodes, dtype=np.int64)
    weights = np.asarray(values, dtype=float)
    return np.bincount(episode_ids, weights=weights)
