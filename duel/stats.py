"""Compatibility names for the shared experiment statistics API."""

from __future__ import annotations

from arena.experiments.statistics import (
    bootstrap_interval,
    classify_interval,
    holm_adjust,
    ratio_mean,
    scaled_effect,
    sign_flip_p_value,
)


def boot_ci(block_sums, block_counts, n_boot=10_000, seed=7, level=0.95):
    """Call the shared block bootstrap with the legacy argument names."""
    return bootstrap_interval(block_sums, block_counts, n_boot, seed, level)


def perm_p(block_sums, block_counts, n_perm=10_000, seed=11):
    """Call the shared sign-flip test with the legacy argument names."""
    return sign_flip_p_value(block_sums, block_counts, n_perm, seed)


holm = holm_adjust
verdict = classify_interval
units = scaled_effect

__all__ = ["boot_ci", "holm", "perm_p", "ratio_mean", "units", "verdict"]
