"""Tests for deterministic experiment artifact handling."""

import json
from pathlib import Path

import numpy as np
import pytest

from arena.experiments.artifacts import (
    parameter_hash,
    result_envelope,
    to_json_value,
    write_json_once,
)


def test_json_conversion_handles_nested_numpy_values() -> None:
    value = {"x": np.arange(3), "y": (np.float32(2), np.int64(4), True)}

    assert to_json_value(value) == {"x": [0, 1, 2], "y": [2.0, 4, True]}


def test_parameter_hash_is_canonical_and_value_sensitive() -> None:
    first = {"b": np.int64(2), "a": 1}
    reordered = {"a": 1, "b": 2}

    assert parameter_hash(first) == parameter_hash(reordered)
    assert parameter_hash(first) != parameter_hash({"a": 1, "b": 3})


def test_result_envelope_preserves_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("arena.experiments.artifacts.git_revision", lambda: "abc123")

    envelope = result_envelope(
        kind="comparison",
        cell="environment x flow",
        seed=3,
        evaluation_size=100,
        tuning_size=20,
        parameters={"threshold": 0.5},
        payload={"mean": np.float64(0.25)},
    )

    assert envelope["code"] == "abc123"
    assert envelope["n_eval"] == 100
    assert envelope["payload"] == {"mean": 0.25}
    assert isinstance(envelope["created_utc"], str)


def test_write_json_once_refuses_to_replace_results(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    write_json_once(path, {"value": np.int64(1)})

    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError, match="result already exists"):
        write_json_once(path, {"value": 2})


def test_unsupported_json_values_fail_early() -> None:
    with pytest.raises(TypeError, match="unsupported JSON value"):
        to_json_value({1, 2})
