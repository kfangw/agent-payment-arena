#!/usr/bin/env python3
"""Base settlement-halt measurement: collect L1 batch-posting gaps (v2).

Why L1. L2 block timestamps are forced to parent+2 s by the OP Stack rules and
are unrelated to wall-clock time (a 28-hour stall left no trace in the
timestamps). An actual settlement halt is an interval in which no batch
reaches L1, so we measure the gaps between transactions sent to the Base batch
inbox.

Notes on the implementation
  - Etherscan returns at most 1,000 rows per request (asking for 10,000 still
    returns 1,000).
  - A batch is posted in roughly every L1 block (median gap 12 s). Twelve months
    is about 2.6 million transactions, so the script streams over them and
    keeps only the gap statistics.
  - Progress is saved so that a run can be interrupted and resumed.

    export ETHERSCAN_API_KEY=...        # read from the environment, never stored
    python3 collect_batches.py                 # 12 months, from scratch
    python3 collect_batches.py --resume        # continue an interrupted run
    python3 collect_batches.py --months 6

Outputs (same folder)
    gaps.csv             gaps above 60 s only (all halt candidates); input of the fit
    gap-histogram.json   distribution of all gaps (counts per log-spaced bin)
    collect-meta.json    range, counts, source (provenance)
    .progress.json       resume state (may be deleted after completion)
"""

import argparse
import bisect
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BATCH_INBOX = "0xff00000000000000000000000000000000008453"
API = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1
PAGE = 1000           # observed Etherscan upper limit
SLEEP = 0.35          # free tier: 3 requests per second
TIMEOUT = 60
RETRIES = 5
KEEP_GAP = 60         # keep only gaps above this many seconds (normal is 12 s)
HIST_EDGES = [0, 12, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600,
              7200, 21600, 86400, 10**9]

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)  # noqa: E731


def get(params, key):
    q = dict(params, chainid=CHAIN_ID, apikey=key)
    url = f"{API}?{urllib.parse.urlencode(q)}"
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                d = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
            continue
        msg = str(d.get("message", ""))
        if "rate limit" in msg.lower():
            last = msg
            time.sleep(2.0 * (attempt + 1))
            continue
        return d
    raise SystemExit(f"request failed {RETRIES} times ({last}). Continue with --resume.")


def block_at(ts, key, closest="after"):
    return int(get({"module": "block", "action": "getblocknobytime",
                    "timestamp": ts, "closest": closest}, key)["result"])


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600} h {sec % 3600 // 60} min" if sec >= 3600 else f"{sec // 60} min {sec % 60} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=float, default=12)
    ap.add_argument("--address", default=BATCH_INBOX)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    if not key:
        sys.exit("ETHERSCAN_API_KEY is empty.\n  export ETHERSCAN_API_KEY=...")

    st = {}
    if a.resume and os.path.exists(P(".progress.json")):
        st = json.load(open(P(".progress.json")))
        print(f"resuming: block {st['start']:,}, {st['n_tx']:,} transactions already processed")

    if st:
        b_from, b_to = st["b_from"], st["b_to"]
        start, prev_ts, n_tx, n_err = st["start"], st["prev_ts"], st["n_tx"], st["n_err"]
        hist = st["hist"]
        gaps = [tuple(g) for g in st["gaps"]]
        first_ts = st["first_ts"]
        last_hash = st.get("last_hash")
    else:
        now = int(time.time())
        t_from = now - int(a.months * 30 * 24 * 3600)
        print(f"target {a.address}")
        print(f"period      {datetime.fromtimestamp(t_from, timezone.utc):%Y-%m-%d} ~ "
              f"{datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d} ({a.months} months)")
        b_from = block_at(t_from, key, "after")
        time.sleep(SLEEP)
        b_to = block_at(now - 600, key, "before")
        print(f"L1 blocks  {b_from:,} ~ {b_to:,}  ({b_to - b_from:,} blocks)\n")
        start, prev_ts, n_tx, n_err = b_from, None, 0, 0
        hist = [0] * (len(HIST_EDGES) - 1)
        gaps, first_ts, last_hash = [], None, None

    t0, calls, stall = time.time(), 0, 0
    while start <= b_to:
        d = get({"module": "account", "action": "txlist", "address": a.address,
                 "startblock": start, "endblock": b_to,
                 "page": 1, "offset": PAGE, "sort": "asc"}, key)
        calls += 1
        res = d.get("result")
        if not isinstance(res, list):
            print(f"  unexpected response: {str(d)[:200]}")
            break
        if not res:
            break

        seen_last = False
        for t in res:
            if last_hash and not seen_last:      # skip duplicates at the boundary block
                if t["hash"] == last_hash:
                    seen_last = True
                continue
            if t.get("isError") == "1":
                n_err += 1
                continue
            ts = int(t["timeStamp"])
            if first_ts is None:
                first_ts = ts
            if prev_ts is not None:
                g = ts - prev_ts
                if g >= 0:
                    hist[bisect.bisect_right(HIST_EDGES, g) - 1] += 1
                    if g > KEEP_GAP:
                        gaps.append((prev_ts, ts, g))
            prev_ts = ts
            n_tx += 1

        new_last = int(res[-1]["blockNumber"])
        last_hash = res[-1]["hash"]
        stall = stall + 1 if new_last <= start else 0
        start = new_last + 1 if stall else new_last
        if stall > 2:
            print("  no progress; stopping.")
            break

        if calls % 25 == 0 or len(res) < PAGE:
            done = max(new_last - b_from, 1)
            frac = done / max(b_to - b_from, 1)
            eta = (time.time() - t0) / frac * (1 - frac)
            print(f"  request {calls:>5}  block {new_last:,}  {frac * 100:>5.1f}%  "
                  f"total {n_tx:,}  above 60 s: {len(gaps):,}  remaining about {hms(eta)}")
            json.dump({"b_from": b_from, "b_to": b_to, "start": start,
                       "prev_ts": prev_ts, "n_tx": n_tx, "n_err": n_err,
                       "hist": hist, "gaps": gaps, "first_ts": first_ts,
                       "last_hash": last_hash}, open(P(".progress.json"), "w"))
        if len(res) < PAGE:
            break
        time.sleep(SLEEP)

    # ---- outputs
    with open(P("gaps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_utc", "end_utc", "gap_seconds", "gap_minutes"])
        for s, e, g in gaps:
            w.writerow([datetime.fromtimestamp(s, timezone.utc).isoformat(),
                        datetime.fromtimestamp(e, timezone.utc).isoformat(),
                        g, round(g / 60, 2)])
    json.dump({"edges_seconds": HIST_EDGES, "counts": hist},
              open(P("gap-histogram.json"), "w"), indent=2)

    span = (prev_ts - first_ts) if (prev_ts and first_ts) else 0
    meta = {"collected_at": datetime.now(timezone.utc).isoformat(),
            "source": "Etherscan V2 account/txlist, chainid=1",
            "address": a.address, "months": a.months,
            "l1_block_range": [b_from, b_to],
            "time_range_utc": [datetime.fromtimestamp(first_ts, timezone.utc).isoformat()
                               if first_ts else None,
                               datetime.fromtimestamp(prev_ts, timezone.utc).isoformat()
                               if prev_ts else None],
            "span_days": round(span / 86400, 2),
            "n_tx_success": n_tx, "n_tx_failed": n_err,
            "n_gaps_over_60s": len(gaps), "api_calls": calls,
            "note": "L2 block timestamps are not wall-clock time under the OP Stack rules and are not used."}
    json.dump(meta, open(P("collect-meta.json"), "w"), indent=2, ensure_ascii=False)

    print(f"\nsuccessful {n_tx:,} (failed {n_err:,}), window {span / 86400:.1f} days, requests {calls:,}")
    if gaps:
        gs = sorted(g for _, _, g in gaps)
        print(f"gaps above 60 s: {len(gs):,}, maximum {gs[-1] / 3600:.2f} hours")
        print("\nHalts by threshold (this table is the input of the fit)")
        for m in (5, 10, 15, 30, 60, 120):
            sel = [g for g in gs if g > m * 60]
            if sel:
                tot = sum(sel)
                print(f"  > {m:>3} min: {len(sel):>4}  sum {tot / 3600:>7.1f} h  "
                      f"mean {tot / len(sel) / 60:>7.1f} min  share {tot / span * 100:>6.3f}%")
            else:
                print(f"  > {m:>3} min: 0")
    else:
        print("No gap above 60 s.")
    print(f"\nWritten: {P('gaps.csv')}\n      {P('gap-histogram.json')}\n      {P('collect-meta.json')}")


if __name__ == "__main__":
    main()
