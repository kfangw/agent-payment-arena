"""The judgement layer: interval, significance, multiplicity, verdict.

Everything works on episode block sums and the shared block counts, the
interchange format the runs emit.  A paired difference between two
policies is the elementwise difference of their block sums; the counts
are shared because both policies replay the same evaluation draws.  The
bootstrap gives the interval (size), the sign-flip permutation gives the
p value (significance), and Holm controls the family of confirmatory
comparisons.
"""
from __future__ import annotations

import numpy as np


def ratio_mean(block_sums: np.ndarray, block_counts: np.ndarray) -> float:
    """Payment-level mean as a ratio of sums, so unequal episode sizes
    weight correctly."""
    return float(np.sum(block_sums) / np.sum(block_counts))


def boot_ci(block_sums, block_counts, n_boot: int = 10_000, seed: int = 7,
            level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the paired mean, resampling by episode
    block.  block_sums holds per-episode diff sums."""
    sums = np.asarray(block_sums, dtype=float)
    counts = np.asarray(block_counts, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(sums)
    # Draw in chunks: the index matrix is n_boot by n, which for a cell
    # with a hundred thousand episodes would not fit in memory at once.
    # The draws and the statistic are unchanged; only the working set is.
    means = np.empty(n_boot, dtype=float)
    step = max(1, int(2_000_000 // max(n, 1)))
    for a in range(0, n_boot, step):
        b = min(a + step, n_boot)
        pick = rng.integers(0, n, size=(b - a, n))
        means[a:b] = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def perm_p(block_sums, block_counts, n_perm: int = 10_000, seed: int = 11) -> float:
    """Two-sided p value from a block sign-flip permutation of the paired
    difference.  Each episode's diff sum flips sign independently under
    the null of no difference."""
    sums = np.asarray(block_sums, dtype=float)
    total = float(np.sum(block_counts))
    obs = abs(float(np.sum(sums)) / total)
    rng = np.random.default_rng(seed)
    n = len(sums)
    # Same chunking as the bootstrap, for the same reason.
    hits = 0
    step = max(1, int(2_000_000 // max(n, 1)))
    for a in range(0, n_perm, step):
        b = min(a + step, n_perm)
        signs = rng.integers(0, 2, size=(b - a, n)) * 2 - 1
        null = np.abs((signs * sums).sum(axis=1) / total)
        hits += int(np.sum(null >= obs))
    return float((1 + hits) / (1 + n_perm))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjustment, returned in the original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (m - rank))
        adj[idx] = min(running, 1.0)
    return adj


def verdict(low: float, high: float, p_holm: float, eps: float,
            alpha: float = 0.05) -> str:
    """One of superior / inferior / equivalent / undetermined (spec 3.5),
    tested in that priority order."""
    if low > 0 and p_holm < alpha:
        return "superior"
    if high < 0 and p_holm < alpha:
        return "inferior"
    if -eps <= low and high <= eps:
        return "equivalent"
    return "undetermined"


def units(mean_diff: float, mean_exposure: float) -> dict:
    """Advantage in the two reported units: dollars per 1000 payments and
    basis points of exposure."""
    return dict(per_1000=mean_diff * 1000.0,
                bp=(mean_diff / mean_exposure * 1e4) if mean_exposure else None)
