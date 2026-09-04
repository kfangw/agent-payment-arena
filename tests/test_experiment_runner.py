"""Regression tests for the shared experiment process runner."""

from pathlib import Path

import pytest

from arena.experiments.runner import Job, run_jobs


def test_run_jobs_records_output_and_failure(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    result = run_jobs(
        (
            Job.python("success", "-c", "print('recorded')"),
            Job.python("failure", "-c", "raise SystemExit(3)"),
        ),
        root=tmp_path,
        logs=logs,
        workers=2,
    )

    assert result.succeeded == ("success",)
    assert result.failed == ("failure",)
    assert result.exit_code == 1
    assert (logs / "success.log").read_text().strip() == "recorded"


def test_run_jobs_rejects_nonpositive_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        run_jobs((), root=tmp_path, logs=tmp_path / "logs", workers=0)
