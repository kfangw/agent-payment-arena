"""Policy-grid execution and Pareto frontier artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from arena.agents.constrained import SchemaConstrainedAgent
from arena.agents.protocol import EvaluationAgent
from arena.agents.scripted import ContentFollowingAgent, ScriptedAgent
from arena.evaluation import ATTACKER, MERCHANT, EvaluationResult, run_suite
from arena.experiments.artifacts import write_json_once
from arena.gateway.protocol import AcceptPolicy
from arena.policies.payment import ParameterizedPolicy
from arena.report import EvaluationReport, ReportRow, build_report
from arena.scenarios import attack_catalog
from arena.telemetry import TraceSink


@dataclass(frozen=True)
class PolicyGrid:
    """Cartesian axes for the accept-policy experiment."""

    spending_limits: tuple[int, ...]
    ask_thresholds: tuple[int | None, ...]
    bond_thresholds: tuple[int | None, ...]

    def policies(self) -> tuple[tuple[str, AcceptPolicy], ...]:
        """Expand the axes into stable identifiers and policy objects."""
        policies = []
        for limit, ask, bond in product(
            self.spending_limits, self.ask_thresholds, self.bond_thresholds
        ):
            identifier = f"limit={limit};ask={_value(ask)};bond={_value(bond)}"
            policies.append((identifier, ParameterizedPolicy(limit, ask, bond)))
        return tuple(policies)


@dataclass(frozen=True)
class FrontierResult:
    """Aggregate report plus its non-dominated configurations."""

    report: EvaluationReport
    frontier: tuple[ReportRow, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible frontier artifact."""
        return asdict(self)


def run_policy_grid(
    grid: PolicyGrid,
    repetitions: int,
    seed: int = 1,
    *,
    trace_sink: TraceSink | None = None,
) -> EvaluationResult:
    """Run every policy cell on the same agents, scenarios, seeds, and repetitions."""
    vulnerable = ContentFollowingAgent()
    agents: tuple[EvaluationAgent, ...] = (
        ScriptedAgent(frozenset({MERCHANT}), 25),
        vulnerable,
        SchemaConstrainedAgent(vulnerable, frozenset({MERCHANT}), 25),
    )
    return run_suite(
        "policy-grid",
        "1",
        attack_catalog(MERCHANT, ATTACKER),
        agents,
        repetitions,
        seed,
        policies=grid.policies(),
        trace_sink=trace_sink,
    )


def build_frontier(result: EvaluationResult) -> FrontierResult:
    """Return rows not strictly dominated across the complete cost vector."""
    report = build_report(result)
    frontier = tuple(
        candidate
        for candidate in report.rows
        if not any(
            other.agent_id == candidate.agent_id and _dominates(other, candidate)
            for other in report.rows
            if other is not candidate
        )
    )
    return FrontierResult(report, frontier)


def write_frontier(result: FrontierResult, json_path: Path, svg_path: Path) -> None:
    """Write machine-readable points and a compact spend/blocking plot."""
    write_json_once(json_path, result.to_dict())
    if svg_path.exists():
        raise FileExistsError(f"result already exists: {svg_path}")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(result))


def render_svg(result: FrontierResult) -> str:
    """Render unauthorized spend against benign blocking as standalone SVG."""
    points = result.report.rows
    max_x = max((row.metrics["unauthorized_spend"].mean for row in points), default=1) or 1
    max_y = max((row.metrics["benign_tasks_blocked"].mean for row in points), default=1) or 1
    circles = []
    frontier_ids = {(row.agent_id, row.policy_id) for row in result.frontier}
    for row in points:
        x = 55 + 520 * row.metrics["unauthorized_spend"].mean / max_x
        y = 325 - 270 * row.metrics["benign_tasks_blocked"].mean / max_y
        color = "#d1495b" if (row.agent_id, row.policy_id) in frontier_ids else "#8da0ae"
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>'
            f"{row.agent_id} | {row.policy_id}</title></circle>"
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="380" '
        'viewBox="0 0 640 380">\n'
        '<rect width="640" height="380" fill="white"/>\n'
        '<path d="M55 35V325H600" fill="none" stroke="#222"/>\n'
        '<text x="260" y="365">Unauthorized spend</text>\n'
        '<text x="16" y="230" transform="rotate(-90 16 230)">Benign tasks blocked</text>\n'
        + "\n".join(circles)
        + "\n</svg>\n"
    )


def _dominates(left: ReportRow, right: ReportRow) -> bool:
    names = (
        "unauthorized_spend",
        "benign_tasks_blocked",
        "escalations",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "escalation_latency_ms",
    )
    comparisons = [left.metrics[name].mean <= right.metrics[name].mean for name in names]
    strict = [left.metrics[name].mean < right.metrics[name].mean for name in names]
    return all(comparisons) and any(strict)


def _value(value: int | None) -> str:
    return "off" if value is None else str(value)
