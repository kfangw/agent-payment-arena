"""Run S3: the four injection axes against B4 on the two main-narrative cells.

Each (cell, axis) is an independent job at the base cell's seed and full
evaluation size, so the no-injection level reproduces the Table 9 A2 - B4.
Eight jobs run in parallel; a job whose output exists is skipped.  Per-job
stdout goes to reports/_logs/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"

# main-narrative cells at their exp1 base seeds (Table 9 assignment)
CELL_SEED = [("E-slow", "F2", 5), ("E-outage", "F2", 8)]
AXES = ["delta", "kappa", "lambda", "noise"]


def jobs():
    out = []
    for env, flow, seed in CELL_SEED:
        for axis in AXES:
            tag = f"{env}_{flow}_{axis}_s{seed}"
            if (ROOT / "results_inject" / f"b4_{tag}.json").exists():
                continue
            out.append(
                (
                    f"injb4_{tag}",
                    [
                        sys.executable,
                        "-m",
                        "duel.inject_b4",
                        "--cell",
                        f"{env} x {flow}",
                        "--axis",
                        axis,
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
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    todo = jobs()
    print(f"launching {len(todo)} jobs on {workers} workers", flush=True)
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
