"""Run S2 over the nine confirmatory cells in parallel.

Each cell tunes B4 (k, a, b) on the evaluation split and scores it on the
same split (duel.b4 --oracle --name b4oracle -> b4oracle_*).  This is not a
deployable rule: it uses no holdout, so it is an upper bound on what any
search over the declared grid could reach.  Resumable: a cell whose output
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


def jobs():
    out = []
    for env, flow, seed in CELLS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if (ROOT / "results" / f"b4oracle_{tag}.json").exists():
            continue
        out.append(
            (
                f"b4oracle_{tag}",
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
                    "--oracle",
                    "--name",
                    "b4oracle",
                    "--out",
                    "results",
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
