"""Dual-provider chain repolling: measure reorg depth, stalls, and the
regime-switch rates that E-fast and E-outage stand on.

Two RPC endpoints are polled in lock step so a fault on one can be told
apart from a fault on the chain: only a stall both providers report at
once counts as a chain stall.  The pure estimators (reorgs, stalls,
regime) run on a list of samples and carry no IO, so a synthetic log
exercises them without a network.

A sample is one poll of one provider:

    {"t_ms": int, "provider": str,
     "height": {"unsafe": h, "safe": h, "finalized": h},
     "hash":   {"unsafe": x, "safe": x, "finalized": x},
     "rtt_ms": float, "error": str | None}

Run:  python -m duel.poll --rpc-a URL --rpc-b URL --seconds 86400
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

TAGS = ("unsafe", "safe", "finalized")
# The public JSON-RPC tag for the unsafe head is "latest".
TAG_RPC = {"unsafe": "latest", "safe": "safe", "finalized": "finalized"}


# ------------------------------------------------------------ IO layer
def _rpc(url: str, method: str, params: list, timeout: float) -> dict:
    """One JSON-RPC call; returns the result object or raises."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body,  # noqa: S310  (declared RPC URL)
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode())
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload["result"]


def poll_once(url: str, provider: str, t_ms: int, timeout: float = 5.0) -> dict:
    """Poll the three heads of one provider.  Never raises; a failed call
    lands in the sample's error field so the loop keeps its cadence.

    The sample carries both a wall clock (t_ms, comparable across process
    restarts) and a monotonic clock (mono_ms, immune to wall-clock jumps);
    durations are measured on the monotonic clock, and a gap between
    monotonic stamps marks a polling gap that must not be counted."""
    height: dict = {}
    hsh: dict = {}
    err = None
    t0 = time.monotonic()
    try:
        for tag in TAGS:
            blk = _rpc(url, "eth_getBlockByNumber", [TAG_RPC[tag], False], timeout)
            height[tag] = int(blk["number"], 16) if blk and blk.get("number") else None
            hsh[tag] = blk["hash"] if blk else None
    except Exception as exc:  # noqa: BLE001  (any failure is a datum, not a crash)
        err = type(exc).__name__
    t1 = time.monotonic()
    return dict(t_ms=t_ms, mono_ms=int(t1 * 1000), provider=provider,
                height=height, hash=hsh, rtt_ms=(t1 - t0) * 1000.0, error=err)


# ------------------------------------------------------------ pure estimators
def by_provider(samples: list[dict]) -> dict[str, list[dict]]:
    """Split a mixed log into per-provider streams, time order preserved."""
    out: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        out[s["provider"]].append(s)
    return dict(out)


def reorgs(stream: list[dict]) -> list[dict]:
    """Per-depth reorg counts on one provider's unsafe chain.

    A reorg is a height whose hash changes after we first saw it; its
    depth is the head height at that moment minus the changed height.
    n_i, the blocks that were buried at least i deep and so could have
    reorged at depth i, is the denominator of the rule-of-three bound.
    """
    seen: dict = {}                 # height -> latest hash
    head = None
    kd: dict = defaultdict(int)     # depth -> reorg count, depth at its moment
    for s in stream:
        if s.get("error"):
            continue
        h = s["height"].get("unsafe")
        x = s["hash"].get("unsafe")
        if h is None or x is None:
            continue
        if head is None or h > head:
            head = h
        if h in seen and seen[h] != x:
            kd[head - h] += 1     # reorg depth uses the head at this moment
        seen[h] = x
    if head is None:
        return []
    # The unsafe head only advances, so a height hh ends buried head - hh
    # deep; n_i, the blocks buried at least i deep, is decided by the final
    # head alone (no per-tick scan).
    burials = sorted(head - hh for hh in seen)
    out = []
    max_depth = max([*kd, burials[-1] if burials else 0])
    for i in range(1, max_depth + 1):
        n_i = sum(1 for b in burials if b >= i)
        if n_i == 0:
            continue
        k_i = kd.get(i, 0)
        out.append(dict(depth=i, count=k_i, blocks=n_i,
                        point=k_i / n_i, upper95=3.0 / n_i))
    return out


def _mono(s: dict) -> int | None:
    """Monotonic stamp, falling back to the wall clock for older logs."""
    return s.get("mono_ms", s.get("t_ms"))


def stall_runs(stream: list[dict], period_s: float, t_stall_s: float = 20.0,
               gap_factor: float = 3.0) -> list[dict]:
    """Runs where the unsafe head stayed put for at least t_stall_s.

    Duration is accumulated on the monotonic clock, but a step longer than
    gap_factor * period is a polling gap and contributes no time: a missing
    stretch must not inflate the stall, because that duration feeds the
    regime rate directly.  Each run carries its monotonic start and end so
    two providers can be checked for a simultaneous stall.
    """
    runs: list[dict] = []
    last_h = None
    prev_mono = None
    run: dict | None = None
    for s in stream:
        if s.get("error"):
            continue                       # a failed poll is a gap, not a datum
        h = s["height"].get("unsafe")
        mono = _mono(s)
        if h is None or mono is None:
            continue
        if last_h is not None and h == last_h:
            step = (mono - prev_mono) / 1000.0 if prev_mono is not None else period_s
            inc = step if step <= gap_factor * period_s else 0.0   # drop gaps
            if run is None:
                run = dict(start_mono=prev_mono, end_mono=mono, dur=0.0, samples=1)
            run["dur"] += inc
            run["end_mono"] = mono
            run["samples"] += 1
        else:
            if run is not None and run["dur"] >= t_stall_s:
                runs.append(run)
            run = None
        last_h, prev_mono = h, mono
    if run is not None and run["dur"] >= t_stall_s:
        runs.append(run)
    return runs


def _overlap(a: dict, b: dict) -> bool:
    return a["start_mono"] <= b["end_mono"] and b["start_mono"] <= a["end_mono"]


def stalls(streams: dict[str, list[dict]], period_s: float = 2.0,
           t_stall_s: float = 20.0) -> dict:
    """Chain stalls need both providers stuck at once; a one-sided stall
    is a provider fault.  Confirmed durations are the gap-robust overlap
    length (the smaller of the two runs' measured durations)."""
    per = {p: stall_runs(st, period_s, t_stall_s) for p, st in streams.items()}
    provs = list(per)
    confirmed: list[float] = []
    provider_only = 0
    if len(provs) >= 2:
        a_runs, b_runs = per[provs[0]], per[provs[1]]
        for a in a_runs:
            hit = next((b for b in b_runs if _overlap(a, b)), None)
            if hit is not None:
                confirmed.append(min(a["dur"], hit["dur"]))
            else:
                provider_only += 1
        provider_only += sum(1 for b in b_runs
                             if not any(_overlap(a, b) for a in a_runs))
    else:
        provider_only = sum(len(v) for v in per.values())
    return dict(count=len(confirmed), durations_s=confirmed,
                provider_only=provider_only)


def regime(n_stalls: int, seconds: float, mean_duration_s: float,
           tick_s: float) -> dict:
    """Coarse-tick regime rates from the stall record (spec 4.3)."""
    ticks = seconds / tick_s if tick_s > 0 else 0.0
    p01 = (n_stalls / ticks) if ticks > 0 else 0.0
    p10 = (tick_s / mean_duration_s) if mean_duration_s > 0 else 0.0
    return dict(tick_s=tick_s, p01=p01, p10=p10)


def block_intervals(stream: list[dict]) -> dict:
    """Seconds between unsafe-head advances: mean and two quantiles."""
    import numpy as np
    times = []
    last_h = None
    last_ms = None
    for s in stream:
        if s.get("error"):
            continue
        h = s["height"].get("unsafe")
        if h is None:
            continue
        if last_h is not None and h > last_h and last_ms is not None:
            times.append((s["t_ms"] - last_ms) / 1000.0 / (h - last_h))
        if last_h is None or h > last_h:
            last_h, last_ms = h, s["t_ms"]
    if not times:
        return dict(mean=None, p50=None, p95=None)
    arr = np.asarray(times)
    return dict(mean=float(arr.mean()), p50=float(np.quantile(arr, 0.5)),
                p95=float(np.quantile(arr, 0.95)))


def summarize(samples: list[dict], period_s: float = 2.0, tick_s: float = 60.0,
              t_stall_s: float = 20.0) -> dict:
    """Assemble the spec 4.5 summary from a full log.  Observation time is
    counted from the sample count, not the wall-clock span, so polling
    gaps do not inflate it."""
    streams = by_provider(samples)
    provs = sorted(streams)
    n_per = {p: len(streams[p]) for p in provs}
    seconds = (min(n_per.values()) if n_per else 0) * period_s
    reorg_rows = reorgs(streams[provs[0]]) if provs else []
    stall = stalls(streams, period_s, t_stall_s)
    durs = stall["durations_s"]
    mean_d = (sum(durs) / len(durs)) if durs else 0.0
    reg = regime(stall["count"], seconds, mean_d, tick_s)
    reg["source"] = "observed" if stall["count"] > 0 else "public_record"
    t0 = min((s["t_ms"] for s in samples), default=0)
    t1 = max((s["t_ms"] for s in samples), default=0)
    return dict(
        window=dict(start_ms=t0, end_ms=t1, seconds=seconds,
                    samples=len(samples)),
        providers=provs,
        block_interval_s=block_intervals(streams[provs[0]]) if provs else {},
        reorgs=reorg_rows,
        stalls=dict(count=stall["count"], durations_s=durs,
                    confirmed_both=stall["count"]),
        provider_only_incidents=stall["provider_only"],
        regime=reg,
    )


# ------------------------------------------------------------ collection loop
def _log_event(events_path: str, kind: str, reason: str) -> None:
    """Append one supervisor event (start / restart / stop) so a
    multi-day collection records its own interruptions instead of hiding
    them by making a fresh file each run."""
    ev = dict(kind=kind, reason=reason, t_ms=int(time.time() * 1000),
              mono_ms=int(time.monotonic() * 1000))
    with open(events_path, "a") as fh:
        fh.write(json.dumps(ev) + "\n")


def collect(urls: dict[str, str], out_dir: str, seconds: float,
            period_s: float = 2.0, poll_fn=poll_once,
            watchdog_s: float = 120.0) -> None:
    """Poll every provider every period_s for `seconds`, appending raw
    jsonl to one stable file per provider.  Raises if no provider has
    answered within watchdog_s, so the supervisor can restart it.

    Files are opened in append mode and never rotated per run, so a
    restart continues the same log rather than orphaning the last one.
    """
    os.makedirs(out_dir, exist_ok=True)
    handles = {p: open(os.path.join(  # noqa: SIM115  (long-lived append handles)
        out_dir, f"poll_{p}.jsonl"), "a") for p in urls}
    deadline = time.monotonic() + seconds
    last_ok = time.monotonic()
    try:
        while time.monotonic() < deadline:
            t_ms = int(time.time() * 1000)
            any_ok = False
            for p, url in urls.items():
                s = poll_fn(url, p, t_ms)
                handles[p].write(json.dumps(s) + "\n")
                handles[p].flush()
                any_ok = any_ok or not s.get("error")
            if any_ok:
                last_ok = time.monotonic()
            elif time.monotonic() - last_ok > watchdog_s:
                raise RuntimeError("watchdog: no provider answered")
            time.sleep(period_s)
    finally:
        for h in handles.values():
            h.close()


def supervise(urls: dict[str, str], out_dir: str, seconds: float,
              period_s: float = 2.0, poll_fn=poll_once,
              watchdog_s: float = 120.0, backoff_s: float = 5.0) -> str:
    """Run collect under a watcher: on any crash or watchdog trip, log a
    restart event and resume until the overall deadline, then summarise
    the whole persistent log.  Returns the summary path."""
    os.makedirs(out_dir, exist_ok=True)
    events = os.path.join(out_dir, "poll_events.jsonl")
    _log_event(events, "start", "supervisor start")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            collect(urls, out_dir, end - time.monotonic(), period_s,
                    poll_fn=poll_fn, watchdog_s=watchdog_s)
        except Exception as exc:  # noqa: BLE001  (restart on any failure)
            _log_event(events, "restart", f"{type(exc).__name__}: {exc}")
            if time.monotonic() < end:
                time.sleep(min(backoff_s, max(0.0, end - time.monotonic())))
    _log_event(events, "stop", "deadline reached")
    return summarize_dir(out_dir, period_s)


def summarize_dir(out_dir: str, period_s: float = 2.0) -> str:
    """Summarise the persistent per-provider logs into one summary file."""
    samples: list[dict] = []
    for path in sorted(Path(out_dir).glob("poll_*.jsonl")):
        if path.name == "poll_events.jsonl":
            continue                       # the supervisor log, not samples
        samples.extend(load_jsonl(str(path)))
    summary = summarize(samples, period_s=period_s)
    summary["restarts"] = _count_restarts(os.path.join(out_dir,
                                                        "poll_events.jsonl"))
    out = os.path.join(out_dir, "poll_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    return out


def _count_restarts(events_path: str) -> int:
    if not Path(events_path).exists():
        return 0
    return sum(1 for ev in load_jsonl(events_path) if ev.get("kind") == "restart")


def load_jsonl(path: str) -> list[dict]:
    """Read a raw poll log back into a sample list."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> None:
    """CLI entry: run a supervised collection against two endpoints."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc-a", default=os.environ.get("RPC_A"))
    ap.add_argument("--rpc-b", default=os.environ.get("RPC_B"))
    ap.add_argument("--seconds", type=float, default=86400.0)
    ap.add_argument("--period", type=float, default=2.0)
    ap.add_argument("--watchdog", type=float, default=120.0)
    ap.add_argument("--out", default="poll")
    args = ap.parse_args(argv)
    if not args.rpc_a or not args.rpc_b:
        ap.error("two endpoints required (--rpc-a/--rpc-b or RPC_A/RPC_B)")
    urls = {"a": args.rpc_a, "b": args.rpc_b}
    path = supervise(urls, args.out, args.seconds, args.period,
                     watchdog_s=args.watchdog)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
