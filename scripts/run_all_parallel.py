"""Run every remaining R1 (B4) and R2 (delta channels) cell in parallel.

Each cell is an independent process; a thread pool caps concurrency at the
core count.  Cells whose result file already exists are skipped, so the
launcher is resumable.  Per-cell stdout goes to reports/_logs/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"

B4_CELLS = [
    ("E-fast", "F1", 1),
    ("E-fast", "F2", 2),
    ("E-fast", "F3", 3),
    ("E-slow", "F1", 4),
    ("E-slow", "F2", 5),
    ("E-slow", "F3", 6),
    ("E-outage", "F1", 7),
    ("E-outage", "F2", 8),
    ("E-outage", "F3", 9),
]
RSHIFT_CELLS = [("E-slow x F2", 2004), ("E-outage x F2", 2006)]


def jobs():
    out = []
    for env, flow, seed in B4_CELLS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if (ROOT / "results" / f"b4_{tag}.json").exists():
            continue
        out.append(
            (
                f"b4_{tag}",
                [
                    sys.executable,
                    "-m",
                    "duel.b4",
                    "--env",
                    env,
                    "--flow",
                    flow,
                    "--seed",
                    str(seed),
                    "--out",
                    "results",
                ],
            )
        )
    for cell, seed in RSHIFT_CELLS:
        e, f = (s.strip() for s in cell.split("x"))
        tag = f"{e}_{f}_delta_s{seed}"
        if (ROOT / "results_inject" / f"rshift_{tag}.json").exists():
            continue
        out.append(
            (
                f"rshift_{tag}",
                [
                    sys.executable,
                    "-m",
                    "duel.rshift",
                    "--cell",
                    cell,
                    "--seed",
                    str(seed),
                    "--out",
                    "results_inject",
                ],
            )
        )
    return out


def run_one(name, cmd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return name, rc


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, os.cpu_count())
    todo = jobs()
    print(f"launching {len(todo)} cells on {workers} workers", flush=True)
    for name, _ in todo:
        print(f"  queued {name}", flush=True)
    ok, bad = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, n, c): n for n, c in todo}
        for fut in as_completed(futs):
            name, rc = fut.result()
            (ok if rc == 0 else bad).append(name)
            print(f"done {name} rc={rc}  ({len(ok) + len(bad)}/{len(todo)})", flush=True)
    print(f"\ncompleted ok={len(ok)} failed={len(bad)}", flush=True)
    if bad:
        print("FAILED:", bad, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
