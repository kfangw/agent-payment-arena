"""Regression tests for the shared experiment process runner."""

from pathlib import Path

import pytest

from arena.experiments.runner import Job, PipelineJob, run_jobs


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


def test_pipeline_runs_steps_in_order(tmp_path: Path) -> None:
    output = tmp_path / "order.txt"
    result = run_jobs(
        (
            PipelineJob.python(
                "pipeline",
                ("-c", f"from pathlib import Path; Path({str(output)!r}).write_text('first')"),
                (
                    "-c",
                    "from pathlib import Path; "
                    f"p=Path({str(output)!r}); "
                    "p.write_text(p.read_text() + '-second')",
                ),
            ),
        ),
        root=tmp_path,
        logs=tmp_path / "logs",
        workers=1,
    )

    assert result.exit_code == 0
    assert output.read_text() == "first-second"


def test_pipeline_stops_after_a_failed_step(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    result = run_jobs(
        (
            PipelineJob.python(
                "pipeline",
                ("-c", "raise SystemExit(4)"),
                ("-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            ),
        ),
        root=tmp_path,
        logs=tmp_path / "logs",
        workers=1,
    )

    assert result.failed == ("pipeline",)
    assert not marker.exists()
