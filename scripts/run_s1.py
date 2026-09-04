"""Run S1 over the nine confirmatory cells in parallel.

Two independent jobs per cell: the B3 grid sweep (arena.experiments.settlement.grid -> settlement_gridN)
and the B4 retune at the finest suspicion grid (arena.experiments.settlement.b4 --b3-n 161 ->
b4_gridN).  Both reuse the base cell's draws and read A2 from the base
result, so neither recomputes family A.  Resumable: a cell whose output
exists is skipped.  Per-cell stdout goes to reports/_logs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arena.experiments.runner import Job, default_workers, run_jobs
from arena.experiments.settlement.design import CONFIRMATORY_RUNS

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"

GRID_N = 161


def jobs():
    out = []
    for env, flow, seed in CONFIRMATORY_RUNS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if not (ROOT / "results" / f"settlement_gridN_{tag}.json").exists():
            out.append(
                Job.python(
                    f"grid_{tag}",
                    "-m",
                    "arena.experiments.settlement.grid",
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
        if not (ROOT / "results" / f"b4_gridN_{tag}.json").exists():
            out.append(
                Job.python(
                    f"b4gridN_{tag}",
                    "-m",
                    "arena.experiments.settlement.b4",
                    "--env",
                    env,
                    "--flow",
                    flow,
                    "--seed",
                    str(seed),
                    "--b3-n",
                    str(GRID_N),
                    "--name",
                    "b4_gridN",
                    "--out",
                    "results",
                )
            )
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else default_workers()
    raise SystemExit(run_jobs(jobs(), root=ROOT, logs=LOGS, workers=workers).exit_code)


if __name__ == "__main__":
    main()
