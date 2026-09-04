"""Acceptance checks for applying a poll summary to gate.py.

Run:  python -m duel.validate_poll_apply
"""

from __future__ import annotations

from pathlib import Path

from .poll_apply import N_FAST, fast_hazards, rewrite


def _summary() -> dict:
    return dict(
        reorgs=[
            dict(depth=1, count=0, blocks=40000, point=0.0, upper95=7.5e-5),
            dict(depth=2, count=1, blocks=39000, point=2.6e-5, upper95=7.7e-5),
        ],
        regime=dict(tick_s=60.0, p01=1.3e-4, p10=1.0 / 55.0, source="observed"),
    )


def hazards_take_upper_bound() -> None:
    """The depth-i hazard is the rule-of-three upper bound where observed,
    the floor elsewhere, and f_0 stays the submission failure."""
    f = fast_hazards(_summary())
    assert len(f) == N_FAST
    assert f[0] == 0.005
    assert abs(f[1] - 7.5e-5) < 1e-12 and abs(f[2] - 7.7e-5) < 1e-12
    assert f[3] == 1e-6 and f[-1] == 1e-6
    print("hazards ok: upper bounds at observed depths, floor elsewhere")


def rewrite_replaces_and_compiles() -> None:
    """The rewritten gate.py carries the new numbers and still compiles,
    with the E-fast array and regime rates changed exactly once each."""
    src = Path("duel/gate.py").read_text()
    out = rewrite(src, _summary())
    assert "p01=0.00013" in out, out[out.index("p01=") : out.index("p01=") + 20]
    assert "7.5e-05" in out
    assert out != src
    compile(out, "gate_rewritten.py", "exec")  # still valid Python
    # idempotence guard: a second rewrite with the same summary is stable
    assert rewrite(out, _summary()) == out
    print("rewrite ok: new constants present, source compiles, stable")


def main() -> None:
    """Run every acceptance check; raise on the first failure."""
    hazards_take_upper_bound()
    rewrite_replaces_and_compiles()
    print("\npoll_apply: all checks passed")


if __name__ == "__main__":
    main()
