"""Aggregate repeated evaluations into JSON and Markdown reports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from arena.evaluation import EvaluationResult
from arena.experiments.artifacts import git_revision, write_json_once

METRICS = (
    "unauthorized_spend",
    "benign_tasks_blocked",
    "escalations",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
)


@dataclass(frozen=True)
class Estimate:
    """Mean and bootstrap interval across repetitions."""

    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ReportRow:
    """Complete metric vector for one agent-policy configuration."""

    agent_id: str
    policy_id: str
    metrics: dict[str, Estimate]


@dataclass(frozen=True)
class EvaluationReport:
    """Versioned aggregate report."""

    suite: str
    repetitions: int
    seed: int
    code_revision: str | None
    rows: tuple[ReportRow, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report mapping."""
        return asdict(self)


def build_report(result: EvaluationResult) -> EvaluationReport:
    """Aggregate each metric by repetition and attach intervals."""
    if result.repetitions < 2:
        raise ValueError("reports require at least two repetitions")
    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in result.records:
        totals = grouped[(record.agent_id, record.policy_id, record.repetition)]
        for name in METRICS:
            totals[name] += float(getattr(record.metrics, name))
    configurations = sorted({(agent, policy) for agent, policy, _ in grouped})
    rows = []
    for agent, policy in configurations:
        estimates = {}
        for name in METRICS:
            values = np.array(
                [
                    grouped[(agent, policy, repetition)][name]
                    for repetition in range(result.repetitions)
                ]
            )
            estimates[name] = _estimate(values, result.seed)
        rows.append(ReportRow(agent, policy, estimates))
    return EvaluationReport(
        result.suite,
        result.repetitions,
        result.seed,
        git_revision(),
        tuple(rows),
    )


def _estimate(values: NDArray[np.float64], seed: int) -> Estimate:
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(2_000, len(values)))
    means = values[selected].mean(axis=1)
    return Estimate(
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def render_markdown(report: EvaluationReport) -> str:
    """Render the primary metrics as a compact comparison table."""
    lines = [
        f"# Evaluation report: {report.suite}",
        "",
        f"Repetitions: {report.repetitions}; seed: {report.seed}; code: {report.code_revision}",
        "",
        "| agent | policy | unauthorized spend | benign blocked | escalations "
        "| tokens | latency ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.rows:
        metrics = row.metrics
        tokens = metrics["prompt_tokens"].mean + metrics["completion_tokens"].mean
        lines.append(
            f"| {row.agent_id} | {row.policy_id} | "
            f"{_format(metrics['unauthorized_spend'])} | "
            f"{_format(metrics['benign_tasks_blocked'])} | "
            f"{_format(metrics['escalations'])} | {tokens:.1f} | "
            f"{metrics['latency_ms'].mean:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _format(estimate: Estimate) -> str:
    return f"{estimate.mean:.2f} [{estimate.lower:.2f}, {estimate.upper:.2f}]"


def write_report(report: EvaluationReport, json_path: Path, markdown_path: Path) -> None:
    """Write report artifacts without overwriting existing results."""
    write_json_once(json_path, report.to_dict())
    if markdown_path.exists():
        raise FileExistsError(f"result already exists: {markdown_path}")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report))
