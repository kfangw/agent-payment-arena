"""Regression tests for the shared duel batch runner."""

from pathlib import Path

from duel.runner import Job, run_jobs


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
