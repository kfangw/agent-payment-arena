"""Run R4 (residual diagnosis) on the three E-outage cells in parallel.

Each cell reproduces family A, marginalizes it over each hidden coordinate,
replays every ablation and B3 (n=161) on the confirmatory evaluation split,
and records the arrival-state disagreement map.  E-outage replay stays
tractable at the full sample (tau=10), so R4 runs at 5.66M to match the
residual it explains.  Resumable; numpy threads pinned; logs in reports/_logs.
"""

from __future__ import annotations

from pathlib import Path

from arena.experiments.runner import Job, run_jobs

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"
CELLS = [("E-outage", "F1", 7), ("E-outage", "F2", 8), ("E-outage", "F3", 9)]


def jobs():
    out = []
    for env, flow, seed in CELLS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if (ROOT / "results_r4" / f"r4_{tag}.json").exists():
            continue
        out.append(
            Job.python(
                f"r4_{tag}",
                "-m",
                "arena.experiments.settlement.residual",
                "--env",
                env,
                "--flow",
                flow,
                "--seed",
                str(seed),
                "--results",
                "results",
                "--results-grid",
                "results",
                "--out",
                "results_r4",
            )
        )
    return out


def main():
    raise SystemExit(run_jobs(jobs(), root=ROOT, logs=LOGS, workers=3).exit_code)


if __name__ == "__main__":
    main()
