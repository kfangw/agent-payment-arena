"""Run S2 over the nine confirmatory cells in parallel.

Each cell tunes B4 (k, a, b) on the evaluation split and scores it on the
same split (duel.b4 --oracle --name b4oracle -> b4oracle_*).  This is not a
deployable rule: it uses no holdout, so it is an upper bound on what any
search over the declared grid could reach.  Resumable: a cell whose output
exists is skipped.  Per-cell stdout goes to reports/_logs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from duel.design import CONFIRMATORY_RUNS
from duel.runner import Job, default_workers, run_jobs

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"


def jobs():
    out = []
    for env, flow, seed in CONFIRMATORY_RUNS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if (ROOT / "results" / f"b4oracle_{tag}.json").exists():
            continue
        out.append(
            Job.python(
                f"b4oracle_{tag}",
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
            )
        )
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else default_workers()
    raise SystemExit(run_jobs(jobs(), root=ROOT, logs=LOGS, workers=workers).exit_code)


if __name__ == "__main__":
    main()
