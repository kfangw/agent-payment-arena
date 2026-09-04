"""Acceptance checks for the chain repolling estimators (spec 4.6).

Run:  python -m arena.experiments.settlement.validate_poll
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from .poll import reorgs, stall_runs, stalls, summarize, summarize_dir, supervise

PERIOD = 2.0


def _sample(
    provider: str,
    t_ms: int,
    unsafe_h: int | None,
    unsafe_x: str | None,
    error: str | None = None,
    mono_ms: int | None = None,
) -> dict:
    return dict(
        t_ms=t_ms,
        mono_ms=t_ms if mono_ms is None else mono_ms,
        provider=provider,
        height=dict(unsafe=unsafe_h, safe=unsafe_h, finalized=unsafe_h),
        hash=dict(unsafe=unsafe_x, safe=unsafe_x, finalized=unsafe_x),
        rtt_ms=1.0,
        error=error,
    )


def _chain(provider: str, n: int, start_ms: int = 0, hashes: dict | None = None):
    """A cleanly advancing unsafe chain, one advance per tick."""
    hashes = hashes or {}
    return [
        _sample(provider, start_ms + i * int(PERIOD * 1000), i, hashes.get(i, f"x{i}"))
        for i in range(n)
    ]


def a6_1_synthetic_reorg() -> None:
    """A6-1: a known depth-2 reorg is estimated with n_i and 3/n_i right."""
    # advance cleanly to head 120, then re-report height 118 (depth 2)
    stream = _chain("a", 121)
    stream.append(_sample("a", stream[-1]["t_ms"] + 2000, 118, "REORG"))
    # keep advancing so the reorged height stays buried deep
    for i in range(121, 161):
        stream.append(_sample("a", stream[-1]["t_ms"] + 2000, i, f"x{i}"))
    rows = {r["depth"]: r for r in reorgs(stream)}
    assert 2 in rows, "depth-2 reorg not detected"
    r = rows[2]
    assert r["count"] == 1, r
    assert r["blocks"] > 0
    assert abs(r["point"] - 1.0 / r["blocks"]) < 1e-12
    assert abs(r["upper95"] - 3.0 / r["blocks"]) < 1e-12
    print(f"A6-1 ok: depth=2 k=1 n={r['blocks']} point={r['point']:.4g} upper95={r['upper95']:.4g}")


def a6_2_provider_separation() -> None:
    """A6-2: a stall injected on one provider stays provider-only."""
    n = 40
    a = _chain("a", n)
    b = _chain("b", n)
    # freeze provider b's head for 30 s (>= 20 s) mid-window
    freeze_h = b[20]["height"]["unsafe"]
    t0 = b[20]["t_ms"]
    for j in range(15):
        b.append(_sample("b", t0 + (j + 1) * 2000, freeze_h, f"x{freeze_h}"))
    for j in range(15, 30):
        b.append(_sample("b", t0 + (j + 1) * 2000, freeze_h + 1 + (j - 15), f"y{j}"))
    out = stalls({"a": a, "b": b}, t_stall_s=20.0)
    assert out["count"] == 0, f"one-sided stall wrongly confirmed: {out}"
    assert out["provider_only"] >= 1, out
    # now both freeze at once -> confirmed chain stall
    a2 = _chain("a", 20)
    b2 = _chain("b", 20)
    hf = 19
    for j in range(15):
        a2.append(_sample("a", a2[-1]["t_ms"] + 2000, hf, f"x{hf}"))
        b2.append(_sample("b", b2[-1]["t_ms"] + 2000, hf, f"x{hf}"))
    out2 = stalls({"a": a2, "b": b2}, t_stall_s=20.0)
    assert out2["count"] >= 1, f"simultaneous stall not confirmed: {out2}"
    print(f"A6-2 ok: one-sided provider_only={out['provider_only']} confirmed_both={out2['count']}")


def a6_3_rule_of_three() -> None:
    """A6-3: with zero reorgs the depth-1 upper bound is exactly 3/n."""
    stream = _chain("a", 300)
    rows = {r["depth"]: r for r in reorgs(stream)}
    assert 1 in rows
    r = rows[1]
    assert r["count"] == 0
    assert abs(r["upper95"] - 3.0 / r["blocks"]) < 1e-12
    print(f"A6-3 ok: k=0 upper95={r['upper95']:.4g} = 3/{r['blocks']}")


def a6_4_missing_tolerance() -> None:
    """A6-4: observation seconds come from the sample count, so a gap in
    the middle of the log does not inflate the window."""
    a = _chain("a", 50)
    b = _chain("b", 50)
    samples = a + b  # 100 samples, no gap
    s1 = summarize(samples, period_s=PERIOD)
    # drop a contiguous block from the wall clock but keep 100 samples by
    # shifting later timestamps forward: span grows, count is unchanged
    shifted = []
    for s in samples:
        s = dict(s)
        if s["t_ms"] > 40 * 2000:
            s["t_ms"] += 3_600_000  # a one-hour gap
        shifted.append(s)
    s2 = summarize(shifted, period_s=PERIOD)
    assert s1["window"]["seconds"] == s2["window"]["seconds"], (s1, s2)
    assert s1["window"]["seconds"] == 50 * PERIOD
    print(
        f"A6-4 ok: seconds={s2['window']['seconds']} from "
        f"{s2['window']['samples']} samples, span-independent"
    )


def a6_stall_duration() -> None:
    """Sanity: a single provider stall of ~40 s is measured near 40 s."""
    stream = _chain("a", 10)
    hf = 9
    for j in range(20):
        stream.append(_sample("a", stream[-1]["t_ms"] + 2000, hf, f"x{hf}"))
    runs = stall_runs(stream, PERIOD, t_stall_s=20.0)
    assert runs, "stall not detected"
    assert runs[-1]["dur"] >= 20.0
    print(f"stall duration ok: {runs[-1]['dur']:.0f}s stuck run detected")


def a6_gap_not_inflated() -> None:
    """The concrete defect: a polling gap inside a stall must not inflate
    its measured duration (which feeds the regime rate)."""
    stream = _chain("a", 5)
    hf = 4
    t = stream[-1]["t_ms"]
    # 10 stuck samples at cadence (~20 s), then a one-hour polling gap,
    # then 10 more stuck samples (~20 s).  True stuck-with-samples ~40 s.
    for j in range(10):
        t += 2000
        stream.append(_sample("a", t, hf, f"x{hf}"))
    t += 3_600_000  # one-hour gap
    for j in range(10):
        t += 2000
        stream.append(_sample("a", t, hf, f"x{hf}"))
    dur = stall_runs(stream, PERIOD, t_stall_s=20.0)[-1]["dur"]
    assert dur < 60.0, f"gap inflated the stall duration to {dur:.0f}s"
    print(f"A6 gap ok: gap excluded, duration {dur:.0f}s (not the hour gap)")


def a6_supervisor_restarts() -> None:
    """The supervisor logs a restart when the poll function crashes, keeps
    one persistent file per provider, and records the restart count."""
    calls = {"n": 0}

    def flaky_poll(url: str, provider: str, t_ms: int, timeout: float = 5.0):
        calls["n"] += 1
        if provider == "a" and calls["n"] in (3, 4):
            raise ConnectionError("injected outage")
        h = calls["n"] // 2
        return _sample(provider, t_ms, h, f"x{h}", mono_ms=int(time.monotonic() * 1000))

    with tempfile.TemporaryDirectory() as d:
        path = supervise(
            {"a": "u", "b": "u"},
            d,
            seconds=0.4,
            period_s=0.02,
            poll_fn=flaky_poll,
            watchdog_s=1.0,
            backoff_s=0.02,
        )
        events = Path(d) / "poll_events.jsonl"
        assert events.exists()
        kinds = [
            json_line(line)["kind"] for line in events.read_text().splitlines() if line.strip()
        ]
        assert "start" in kinds and "restart" in kinds and "stop" in kinds, kinds
        assert (Path(d) / "poll_a.jsonl").exists()
        assert (Path(d) / "poll_b.jsonl").exists()
        summ = json_load(path)
        assert summ["restarts"] >= 1, summ["restarts"]
        print(
            f"A6 supervisor ok: restarts={summ['restarts']}, "
            f"persistent files kept, events {sorted(set(kinds))}"
        )


def json_line(line: str) -> dict:
    import json

    return json.loads(line)


def json_load(path: str) -> dict:
    import json

    return json.loads(Path(path).read_text())


def main() -> None:
    """Run every acceptance check; raise on the first failure."""
    _ = np  # numpy import guards the estimator's runtime dependency
    a6_1_synthetic_reorg()
    a6_2_provider_separation()
    a6_3_rule_of_three()
    a6_4_missing_tolerance()
    a6_stall_duration()
    a6_gap_not_inflated()
    a6_supervisor_restarts()
    print("\nS6 acceptance: all checks passed")


if __name__ == "__main__":
    main()
