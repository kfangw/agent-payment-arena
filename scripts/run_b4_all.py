"""Run B4 across the nine confirmatory cells, skipping any already done.

Cell-to-seed follows the main run's assignment so B4 sits on the same
draws as its base cell.  Each cell writes results/b4_{tag}.json; a cell
whose file already exists is skipped, so the driver is resumable.
"""

from __future__ import annotations

from pathlib import Path

from arena.experiments.runner import Job, run_jobs

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"
CELL_SEED = [
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


def jobs(out: str = "results") -> list[Job]:
    pending = []
    for env, flow, seed in CELL_SEED:
        tag = f"{env}_{flow}_mid_s{seed}"
        path = ROOT / out / f"b4_{tag}.json"
        if path.exists():
            print(f"skip {tag} (exists)", flush=True)
            continue
        pending.append(
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
                out,
            )
        )
    return pending


def run(out: str = "results") -> int:
    """Run pending cells sequentially and return a shell exit code."""
    return run_jobs(jobs(out), root=ROOT, logs=LOGS, workers=1).exit_code


if __name__ == "__main__":
    import sys

    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "results"))
