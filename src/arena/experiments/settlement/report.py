"""Compatibility names for the shared experiment artifact API."""

from __future__ import annotations

from arena.experiments.artifacts import (
    git_revision as git_rev,
    parameter_hash as params_hash,
    result_envelope,
    to_json_value as jsonable,
    write_json_once as write_once,
)


def envelope(
    kind: str,
    cell: str,
    seed: int,
    n_eval: int,
    n_tune: int,
    resolved: dict[str, object],
    payload: object,
) -> dict[str, object]:
    """Build an envelope using the legacy positional signature."""
    return result_envelope(
        kind=kind,
        cell=cell,
        seed=seed,
        evaluation_size=n_eval,
        tuning_size=n_tune,
        parameters=resolved,
        payload=payload,
    )


__all__ = ["envelope", "git_rev", "jsonable", "params_hash", "write_once"]
