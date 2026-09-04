"""Run every remaining R1 (B4) and R2 (delta channels) cell in parallel.

Each cell is an independent process; a thread pool caps concurrency at the
core count.  Cells whose result file already exists are skipped, so the
launcher is resumable.  Per-cell stdout goes to reports/_logs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arena.experiments.runner import Job, default_workers, run_jobs
from arena.experiments.settlement.design import CONFIRMATORY_RUNS

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"

RSHIFT_CELLS = [("E-slow x F2", 2004), ("E-outage x F2", 2006)]


def jobs():
    out = []
    for env, flow, seed in CONFIRMATORY_RUNS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if (ROOT / "results" / f"b4_{tag}.json").exists():
            continue
        out.append(
            Job.python(
                f"b4_{tag}",
                "-m",
                "arena.experiments.settlement.b4",
                "--env",
                env,
                "--flow",
                flow,
                "--seed",
                str(seed),
                "--out",
                "results",
            )
        )
    for cell, seed in RSHIFT_CELLS:
        e, f = (s.strip() for s in cell.split("x"))
        tag = f"{e}_{f}_delta_s{seed}"
        if (ROOT / "results_inject" / f"rshift_{tag}.json").exists():
            continue
        out.append(
            Job.python(
                f"rshift_{tag}",
                "-m",
                "arena.experiments.settlement.rshift",
                "--cell",
                cell,
                "--seed",
                str(seed),
                "--out",
                "results_inject",
            )
        )
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else default_workers()
    todo = jobs()
    for job in todo:
        print(f"queued {job.name}", flush=True)
    raise SystemExit(run_jobs(todo, root=ROOT, logs=LOGS, workers=workers, label="cells").exit_code)


if __name__ == "__main__":
    main()
