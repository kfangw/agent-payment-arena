"""Run S3: the four injection axes against B4 on the two main-narrative cells.

Each (cell, axis) is an independent job at the base cell's seed and full
evaluation size, so the no-injection level reproduces the Table 9 A2 - B4.
Eight jobs run in parallel; a job whose output exists is skipped.  Per-job
stdout goes to reports/_logs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arena.experiments.runner import Job, run_jobs

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
                Job.python(
                    f"injb4_{tag}",
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
                )
            )
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    raise SystemExit(run_jobs(jobs(), root=ROOT, logs=LOGS, workers=workers).exit_code)


if __name__ == "__main__":
    main()
