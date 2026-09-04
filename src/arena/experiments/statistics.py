"""Statistical utilities for paired, block-structured experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _blocks(
    block_sums: ArrayLike, block_counts: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and normalize paired block summaries."""
    sums = np.asarray(block_sums, dtype=float)
    counts = np.asarray(block_counts, dtype=float)
    if sums.ndim != 1 or counts.ndim != 1:
        raise ValueError("block summaries must be one-dimensional")
    if len(sums) == 0 or len(sums) != len(counts):
        raise ValueError("block sums and counts must have the same nonzero length")
    if not np.all(np.isfinite(sums)) or not np.all(np.isfinite(counts)):
        raise ValueError("block summaries must be finite")
    if np.any(counts <= 0):
        raise ValueError("block counts must be positive")
    return sums, counts


def ratio_mean(block_sums: ArrayLike, block_counts: ArrayLike) -> float:
    """Return an observation-level mean from block sums and counts."""
    sums, counts = _blocks(block_sums, block_counts)
    return float(np.sum(sums) / np.sum(counts))


def bootstrap_interval(
    block_sums: ArrayLike,
    block_counts: ArrayLike,
    n_resamples: int = 10_000,
    seed: int = 7,
    level: float = 0.95,
) -> tuple[float, float]:
    """Estimate a percentile interval by resampling complete blocks."""
    sums, counts = _blocks(block_sums, block_counts)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one")
    rng = np.random.default_rng(seed)
    block_count = len(sums)
    means = np.empty(n_resamples, dtype=float)
    chunk_size = max(1, int(2_000_000 // max(block_count, 1)))
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        selected = rng.integers(0, block_count, size=(stop - start, block_count))
        means[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lower_quantile = (1 - level) / 2
    upper_quantile = 1 - lower_quantile
    return (
        float(np.quantile(means, lower_quantile)),
        float(np.quantile(means, upper_quantile)),
    )


def sign_flip_p_value(
    block_sums: ArrayLike,
    block_counts: ArrayLike,
    n_resamples: int = 10_000,
    seed: int = 11,
) -> float:
    """Return a two-sided block sign-flip permutation p-value."""
    sums, counts = _blocks(block_sums, block_counts)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    total_count = float(np.sum(counts))
    observed = abs(float(np.sum(sums)) / total_count)
    rng = np.random.default_rng(seed)
    block_count = len(sums)
    hits = 0
    chunk_size = max(1, int(2_000_000 // max(block_count, 1)))
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        signs = rng.integers(0, 2, size=(stop - start, block_count)) * 2 - 1
        null_values = np.abs((signs * sums).sum(axis=1) / total_count)
        hits += int(np.sum(null_values >= observed))
    return float((1 + hits) / (1 + n_resamples))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply the Holm step-down adjustment in the original order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, p_values[index] * (count - rank))
        adjusted[index] = min(running, 1.0)
    return adjusted


def classify_interval(
    lower: float,
    upper: float,
    adjusted_p_value: float,
    equivalence_margin: float,
    alpha: float = 0.05,
) -> str:
    """Classify an interval as superior, inferior, equivalent, or undetermined."""
    if lower > 0 and adjusted_p_value < alpha:
        return "superior"
    if upper < 0 and adjusted_p_value < alpha:
        return "inferior"
    if -equivalence_margin <= lower and upper <= equivalence_margin:
        return "equivalent"
    return "undetermined"


def scaled_effect(mean_difference: float, mean_exposure: float) -> dict[str, float | None]:
    """Express an effect per thousand observations and in basis points."""
    basis_points = mean_difference / mean_exposure * 1e4 if mean_exposure else None
    return {"per_1000": mean_difference * 1000.0, "bp": basis_points}
