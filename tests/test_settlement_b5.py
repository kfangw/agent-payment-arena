"""Regression tests for the self-contained B5 runner."""

from __future__ import annotations

import numpy as np

from arena.experiments.settlement.b5 import StateMatchedOutagePolicy, run_cell
from arena.experiments.settlement.core import GRANT, REJECT, VERIFY


def test_b5_uses_public_regime_and_thresholds() -> None:
    policy = StateMatchedOutagePolicy(0.25, 0.75)
    assert policy(0, 10, 1, 50.0, 0.10) == REJECT
    assert policy(0, 10, 0, 50.0, 0.10) == GRANT
    assert policy(0, 10, 0, 50.0, 0.50) == VERIFY
    assert policy(0, 10, 0, 50.0, 0.90) == REJECT


def test_b5_small_run_is_deterministic_and_self_contained() -> None:
    kwargs = {"seed": 7, "n_tune": 120, "n_eval": 200, "b3_n": 9, "n_boot": 40, "n_perm": 40}
    first, parameters = run_cell("F1", **kwargs)
    second, _ = run_cell("F1", **kwargs)
    assert parameters["seed"] == 7
    assert first["policy"]["b3_grid_points"] == 9
    np.testing.assert_array_equal(first["b5_block_sums"], second["b5_block_sums"])
    np.testing.assert_array_equal(first["a_block_sums"], second["a_block_sums"])
    assert "sources" not in first
