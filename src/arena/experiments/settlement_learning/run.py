"""Run a finite exact check and optional synthetic replay from a JSON setting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .model import Request, Setting
from .simulate import replay
from .solver import MODES, Solver


def read_setting(path: Path) -> Setting:
    """Read strict parameter names and independent request distributions."""
    raw = json.loads(path.read_text())
    schedule = tuple(
        tuple(
            (
                float(item["weight"]),
                Request(float(item["amount"]), tuple(item["hazards"])),
            )
            for item in distribution
        )
        for distribution in raw.pop("schedule")
    )
    if len(schedule) > 12:
        raise ValueError(
            "initial exact solver supports at most 12 requests per relation"
        )
    return Setting(schedule=schedule, **raw)


def source_metadata() -> dict:
    """Record source hashes as well as revision, including uncommitted code."""
    folder = Path(__file__).resolve().parent
    from arena.experiments.settlement import core

    files = sorted(folder.glob("*.py")) + [Path(core.__file__)]
    hashes = {
        p.name if p.parent == folder else "settlement/core.py": hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in files
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=folder, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=folder, text=True
            )
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    return dict(
        revision=revision,
        dirty=dirty,
        source_sha256=hashes,
        python=platform.python_version(),
        numpy=version("numpy"),
    )


def main() -> None:
    """Write a new result directory; never overwrite a previous run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--relations", type=int, default=0, help="0 computes exact values only"
    )
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument(
        "--split", choices=("e0", "pilot", "evaluation"), default="pilot"
    )
    args = parser.parse_args()
    if args.relations < 0 or args.relations == 1:
        parser.error("relations must be 0 or at least 2")
    setting = read_setting(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        setting=asdict(setting),
        seed=args.seed,
        split=args.split,
        relations=args.relations,
        data_grade="synthetic",
        clock="stage_full_window",
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        **source_metadata(),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    solvers = {mode: Solver(setting, mode) for mode in MODES}
    exact = {mode: solver.evaluate() for mode, solver in solvers.items()}
    planner = solvers["optimal"].future(0, 0, 0)
    if not math.isclose(planner, exact["optimal"], rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("planning/evaluation mismatch")
    if any(value > exact["optimal"] + 1e-9 for value in exact.values()):
        raise RuntimeError("baseline exceeds computed optimal value")
    rows = {mode: [] for mode in MODES}
    with (
        (args.output / "public-events.jsonl").open("w") as event_file,
        (args.output / "private-metrics.jsonl").open("w") as metric_file,
    ):
        for relation in range(args.relations):
            for mode, solver in solvers.items():
                metrics, events = replay(solver, args.seed, args.split, relation)
                rows[mode].append(metrics)
                metric_file.write(
                    json.dumps(dict(policy=mode, relation_id=relation, **metrics))
                    + "\n"
                )
                for event in events:
                    event_file.write(json.dumps(dict(policy=mode, **event)) + "\n")
    summary = {
        "exact_value": exact,
        "planner_value": planner,
        "sample": {},
        "paired": {},
    }
    if args.relations:
        for mode, records in rows.items():
            rewards = [row["reward"] for row in records]
            summary["sample"][mode] = dict(
                means={
                    key: statistics.mean(row[key] for row in records)
                    for key in records[0]
                },
                reward_se=statistics.stdev(rewards) / math.sqrt(args.relations),
                normal_rejection_rate=(
                    sum(r["normal_rejections"] for r in records)
                    / sum(r["normal_requests"] for r in records)
                    if sum(r["normal_requests"] for r in records)
                    else None
                ),
            )
            difference = [
                a["reward"] - b["reward"]
                for a, b in zip(rows["optimal"], records, strict=True)
            ]
            summary["paired"][mode] = dict(
                mean=statistics.mean(difference),
                se=statistics.stdev(difference) / math.sqrt(args.relations),
            )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
