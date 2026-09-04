"""Tests for reusable block-level statistical utilities."""

import pytest

from arena.experiments.statistics import (
    bootstrap_interval,
    classify_interval,
    holm_adjust,
    ratio_mean,
    scaled_effect,
    sign_flip_p_value,
)


def test_ratio_mean_weights_blocks_by_observation_count() -> None:
    assert ratio_mean([2.0, 8.0], [2, 8]) == pytest.approx(1.0)


def test_resampling_is_seeded_and_bounded() -> None:
    sums = [-3.0, -1.0, 2.0, 4.0]
    counts = [3, 2, 2, 3]

    first = bootstrap_interval(sums, counts, n_resamples=500, seed=9)
    second = bootstrap_interval(sums, counts, n_resamples=500, seed=9)
    p_value = sign_flip_p_value(sums, counts, n_resamples=500, seed=10)

    assert first == second
    assert first[0] <= ratio_mean(sums, counts) <= first[1]
    assert 0 < p_value <= 1


def test_multiple_comparison_and_reporting_helpers() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert classify_interval(0.1, 0.3, 0.01, 0.05) == "superior"
    assert classify_interval(-0.02, 0.02, 0.9, 0.05) == "equivalent"
    assert scaled_effect(0.002, 0.5) == {"per_1000": 2.0, "bp": 40.0}
    assert scaled_effect(0.002, 0.0)["bp"] is None


@pytest.mark.parametrize(
    ("sums", "counts"),
    [([], []), ([1], []), ([[1]], [[1]]), ([1], [0]), ([float("nan")], [1])],
)
def test_invalid_blocks_are_rejected(sums: object, counts: object) -> None:
    with pytest.raises(ValueError):
        ratio_mean(sums, counts)  # type: ignore[arg-type]
