"""Process execution for resumable experiment batches."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    """One named experiment process."""

    name: str
    command: tuple[str, ...]

    @classmethod
    def python(cls, name: str, *args: str) -> Job:
        """Create a job that uses the active Python interpreter."""
        return cls(name=name, command=(sys.executable, *args))


@dataclass(frozen=True)
class BatchResult:
    """Successful and failed job names from a completed batch."""

    succeeded: tuple[str, ...]
    failed: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        """Return a shell exit code for the batch."""
        return 1 if self.failed else 0


def default_workers(limit: int = 8) -> int:
    """Choose a conservative worker count for CPU-heavy jobs."""
    return min(limit, os.cpu_count() or 1)


def _process_env(root: Path) -> dict[str, str]:
    """Create a child environment that can import the checkout."""
    current = os.environ.get("PYTHONPATH")
    pythonpath = str(root) if not current else os.pathsep.join((str(root), current))
    return {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def run_job(job: Job, *, root: Path, logs: Path) -> tuple[str, int]:
    """Run one job and write its combined output to a stable log file."""
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{job.name}.log").open("w") as stream:
        completed = subprocess.run(
            job.command,
            cwd=root,
            env=_process_env(root),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return job.name, completed.returncode


def run_jobs(
    jobs: Sequence[Job],
    *,
    root: Path,
    logs: Path,
    workers: int,
    label: str = "jobs",
) -> BatchResult:
    """Run independent jobs concurrently and summarize their exit codes."""
    if workers < 1:
        raise ValueError("workers must be positive")

    queue = tuple(jobs)
    print(f"launching {len(queue)} {label} on {workers} workers", flush=True)
    succeeded: list[str] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_job, job, root=root, logs=logs): job.name for job in queue}
        for future in as_completed(futures):
            name, returncode = future.result()
            (succeeded if returncode == 0 else failed).append(name)
            completed = len(succeeded) + len(failed)
            print(f"done {name} rc={returncode} ({completed}/{len(queue)})", flush=True)

    print(f"completed ok={len(succeeded)} failed={len(failed)}", flush=True)
    if failed:
        print(f"FAILED: {sorted(failed)}", flush=True)
    return BatchResult(tuple(sorted(succeeded)), tuple(sorted(failed)))
