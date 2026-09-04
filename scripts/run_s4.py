"""Run S4: the deeper-settlement cell E-slow-deep across the three flows.

Each flow first runs the base comparison (duel.run -> duel_E-slow-deep_*) and
then B4 on a dense k ladder {0..12} (duel.b4 --k-grid -> b4_E-slow-deep_*),
which must run after its base file exists.  The three flows are independent
and run in parallel.  Resumable: a stage whose output exists is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arena.experiments.runner import PipelineJob, run_jobs

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"

# E-slow-deep is a new cell; its seeds sit in the exp1 band, disjoint from 1-9.
CELL_SEED = [("E-slow-deep", "F1", 10), ("E-slow-deep", "F2", 11), ("E-slow-deep", "F3", 12)]
K_DENSE = ",".join(str(k) for k in range(13))  # {0,1,...,12}


def jobs():
    out = []
    for env, flow, seed in CELL_SEED:
        tag = f"{env}_{flow}_mid_s{seed}"
        steps = []
        if not (ROOT / "results" / f"duel_{tag}.json").exists():
            steps.append(
                (
                    "-m",
                    "duel.run",
                    "--env",
                    env,
                    "--flow",
                    flow,
                    "--cw",
                    "mid",
                    "--n-eval",
                    "5663400",
                    "--n-tune",
                    "200000",
                    "--seed",
                    str(seed),
                    "--out",
                    "results",
                )
            )
        if not (ROOT / "results" / f"b4_{tag}.json").exists():
            steps.append(
                (
                    "-m",
                    "duel.b4",
                    "--env",
                    env,
                    "--flow",
                    flow,
                    "--seed",
                    str(seed),
                    "--k-grid",
                    K_DENSE,
                    "--out",
                    "results",
                )
            )
        if steps:
            out.append(PipelineJob.python(f"s4_{tag}", *steps))
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    raise SystemExit(run_jobs(jobs(), root=ROOT, logs=LOGS, workers=workers).exit_code)


if __name__ == "__main__":
    main()
