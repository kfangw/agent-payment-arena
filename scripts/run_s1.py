"""Run S1 over the nine confirmatory cells in parallel.

Two independent jobs per cell: the B3 grid sweep (duel.grid -> duel_gridN)
and the B4 retune at the finest suspicion grid (duel.b4 --b3-n 161 ->
b4_gridN).  Both reuse the base cell's draws and read A2 from the base
result, so neither recomputes family A.  Resumable: a cell whose output
exists is skipped.  Per-cell stdout goes to reports/_logs/.
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"

CELLS = [
    ("E-fast", "F1", 1), ("E-fast", "F2", 2), ("E-fast", "F3", 3),
    ("E-slow", "F1", 4), ("E-slow", "F2", 5), ("E-slow", "F3", 6),
    ("E-outage", "F1", 7), ("E-outage", "F2", 8), ("E-outage", "F3", 9),
]
GRID_N = 161


def jobs():
    out = []
    for env, flow, seed in CELLS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if not (ROOT / "results" / f"duel_gridN_{tag}.json").exists():
            out.append((f"grid_{tag}",
                        [sys.executable, "-m", "duel.grid", "--env", env,
                         "--flow", flow, "--seed", str(seed), "--out", "results"]))
        if not (ROOT / "results" / f"b4_gridN_{tag}.json").exists():
            out.append((f"b4gridN_{tag}",
                        [sys.executable, "-m", "duel.b4", "--env", env,
                         "--flow", flow, "--seed", str(seed), "--b3-n",
                         str(GRID_N), "--name", "b4_gridN", "--out", "results"]))
    return out


def run_one(name, cmd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    return name, rc


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, os.cpu_count())
    todo = jobs()
    print(f"launching {len(todo)} jobs on {workers} workers", flush=True)
    ok, bad = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, n, c): n for n, c in todo}
        for fut in as_completed(futs):
            name, rc = fut.result()
            (ok if rc == 0 else bad).append(name)
            print(f"done {name} rc={rc}  ({len(ok)+len(bad)}/{len(todo)})",
                  flush=True)
    print(f"\ncompleted ok={len(ok)} failed={len(bad)}", flush=True)
    if bad:
        print("FAILED:", bad, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
