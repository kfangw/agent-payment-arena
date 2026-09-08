"""Rescoring a saved run must reproduce it exactly at the run's own accounting."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.misspec import identity_check, rescore


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("recovery") / "run"
    cmd = [
        sys.executable,
        "-m",
        "arena.experiments.settlement.recovery",
        "--flow",
        "F2",
        "--seed",
        "11",
        "--n-tune",
        "100",
        "--n-eval",
        "100",
        "--repeats",
        "1",
        "--block-size",
        "50",
        "--n-boot",
        "20",
        "--n-v",
        "5",
        "--grid-points",
        "5",
        "--max-watch",
        "1",
        "--conditions",
        "outage",
        "--recovery",
        "0",
        "1",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_identity_at_own_accounting(tiny_run: Path) -> None:
    _, base, _ = envs_for("mid")["E-outage"]
    for recovery in (0.0, 1.0):
        gaps = identity_check(tiny_run, 0, "outage", recovery, 5, base)
        assert max(gaps.values()) == 0.0


def test_misspecified_scores_differ_only_through_settlement(tiny_run: Path) -> None:
    _, base, _ = envs_for("mid")["E-outage"]
    _draws, believed = rescore(tiny_run, 0, "outage", 0.0, 0.0, "policy", 5, base)
    _, actual = rescore(tiny_run, 0, "outage", 0.0, 1.0, "policy", 5, base)
    for name in believed:
        # Same policy, same paths: release decisions and queries cannot change.
        np.testing.assert_array_equal(believed[name][:, 1], actual[name][:, 1])
        np.testing.assert_array_equal(believed[name][:, 6], actual[name][:, 6])
        # Continuation can only add payoff and only to payments that settle late.
        gain = actual[name][:, 0] - believed[name][:, 0]
        assert np.all(gain >= -1e-12)
        assert np.all((gain > 1e-12) <= (actual[name][:, 8] > 0))


def test_cli_writes_summary(tiny_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "rescored"
    cmd = [
        sys.executable,
        "-m",
        "arena.experiments.settlement.misspec",
        "--run",
        str(tiny_run),
        "--believed",
        "0",
        "--actual",
        "1",
        "--n-v",
        "5",
        "--n-boot",
        "20",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["identity_check"]
    assert {r["exercise_at"] for r in summary["rows"]} == {"policy", "world"}
    assert (out / "0-outage-believed-0-actual-1-policy-outcomes.npz").exists()
