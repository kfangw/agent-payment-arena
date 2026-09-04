"""Tests for policy grids and Pareto artifacts."""

from pathlib import Path

from arena.frontier import PolicyGrid, build_frontier, run_policy_grid, write_frontier


def test_grid_expands_every_parameter_cell() -> None:
    grid = PolicyGrid((25, 100), (None, 20), (None, 50))
    assert len(grid.policies()) == 8
    assert len({name for name, _ in grid.policies()}) == 8


def test_frontier_uses_repeated_results_and_writes_artifacts(tmp_path: Path) -> None:
    result = run_policy_grid(PolicyGrid((25, 100), (None, 20), (None,)), 2, seed=9)
    frontier = build_frontier(result)
    assert frontier.report.repetitions == 2
    assert frontier.frontier
    json_path = tmp_path / "frontier.json"
    svg_path = tmp_path / "frontier.svg"
    write_frontier(frontier, json_path, svg_path)
    assert json_path.exists()
    assert "<svg" in svg_path.read_text()
