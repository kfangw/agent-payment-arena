"""Tests for repeated evaluation reporting."""

from pathlib import Path

import pytest

from arena.evaluation import run_minimum_suite
from arena.report import build_report, render_markdown, write_report


def test_report_requires_repetitions_and_carries_intervals() -> None:
    with pytest.raises(ValueError, match="at least two repetitions"):
        build_report(run_minimum_suite(1))

    report = build_report(run_minimum_suite(2))
    assert len(report.rows) == 4
    assert report.rows[0].metrics["unauthorized_spend"].lower >= 0
    rendered = render_markdown(report)
    assert "unauthorized spend" in rendered
    assert report.created_utc in rendered
    assert "Models: none (scripted baselines)" in rendered


def test_report_writes_json_and_markdown_once(tmp_path: Path) -> None:
    report = build_report(run_minimum_suite(2))
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_report(report, json_path, markdown_path)
    assert json_path.exists() and markdown_path.exists()
    with pytest.raises(FileExistsError):
        write_report(report, json_path, markdown_path)
