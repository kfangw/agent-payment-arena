"""Acceptance checks for the result envelope (spec 1.4).

Run:  python -m arena.experiments.settlement.validate_report
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .report import envelope, jsonable, params_hash, write_once


def hash_is_order_independent() -> None:
    """params_hash ignores key order and coerces numpy types."""
    a = dict(b=np.float64(1.5), a=np.int64(3), grid=np.array([1, 2, 3]))
    b = dict(grid=[1, 2, 3], a=3, b=1.5)
    assert params_hash(a) == params_hash(b), "hash depends on key order or dtype"
    c = dict(grid=[1, 2, 3], a=3, b=1.6)
    assert params_hash(a) != params_hash(c), "hash blind to a value change"
    print(f"hash ok: order/dtype-invariant, sensitive to values ({params_hash(a)[:12]})")


def jsonable_is_total() -> None:
    """jsonable leaves no numpy types behind."""
    out = jsonable(dict(x=np.arange(3), y=(np.float32(2.0), True, "s")))
    import json

    json.dumps(out)  # would raise on a stray numpy type
    assert out["x"] == [0, 1, 2]
    assert out["y"][1] is True
    print("jsonable ok: nested numpy fully coerced")


def envelope_has_fields() -> None:
    """envelope carries every common field."""
    env = envelope(
        "settlement",
        "E-fast x F1",
        1,
        200000,
        50000,
        dict(cw=0.01),
        dict(means={"A2": 0.3}),
    )
    for key in (
        "kind",
        "cell",
        "seed",
        "n_eval",
        "n_tune",
        "params_hash",
        "code",
        "created_utc",
        "payload",
    ):
        assert key in env, f"missing field {key}"
    assert env["kind"] == "settlement"
    print("envelope ok: all common fields present")


def write_once_refuses_overwrite() -> None:
    """The second write to the same path raises."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "r.json")
        write_once(path, dict(a=1))
        assert Path(path).exists()
        try:
            write_once(path, dict(a=2))
        except FileExistsError:
            print("write_once ok: refuses to overwrite")
            return
        raise AssertionError("overwrite was not refused")


def main() -> None:
    """Run every check; raise on the first failure."""
    hash_is_order_independent()
    jsonable_is_total()
    envelope_has_fields()
    write_once_refuses_overwrite()
    print("\nreport envelope: all checks passed")


if __name__ == "__main__":
    main()
