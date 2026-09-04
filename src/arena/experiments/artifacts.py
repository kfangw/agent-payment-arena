"""Deterministic metadata and JSON output for experiment results."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def to_json_value(value: object) -> JsonValue:
    """Convert nested Python and NumPy values to deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_value(cast(object, value.tolist()))
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def parameter_hash(parameters: Mapping[str, object]) -> str:
    """Return a stable SHA-256 digest of fully resolved parameters."""
    encoded = json.dumps(to_json_value(parameters), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_revision() -> str | None:
    """Return the current Git revision, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def result_envelope(
    *,
    kind: str,
    cell: str,
    seed: int,
    evaluation_size: int,
    tuning_size: int,
    parameters: Mapping[str, object],
    payload: object,
) -> dict[str, JsonValue]:
    """Wrap a result payload with reproducibility metadata."""
    return {
        "kind": kind,
        "cell": cell,
        "seed": seed,
        "n_eval": evaluation_size,
        "n_tune": tuning_size,
        "params_hash": parameter_hash(parameters),
        "code": git_revision(),
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "payload": to_json_value(payload),
    }


def write_json_once(path: str | Path, value: object) -> None:
    """Write a JSON result without replacing an existing artifact."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"result already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        json.dump(to_json_value(value), stream, indent=1)
