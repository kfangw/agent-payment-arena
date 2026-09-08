"""Execute or summarize the frozen finite release evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESIGN = REPO / "configs/settlement_learning/evaluation_v1/design.json"


def digest(path: Path) -> str:
    """Hash the exact bytes used by the runner."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict) -> None:
    """Permit identical repeated reporting but reject changed provenance."""
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != data:
            raise RuntimeError(f"existing file differs: {path}")
        return
    with path.open("x") as stream:
        stream.write(data)


def verify_run(path: Path, cell: dict, design: dict, seed: int, count: int) -> dict:
    """Accept only complete runs with the frozen sources and settings."""
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "complete", path
    assert manifest["config_sha256"] == cell["sha256"], path
    assert (manifest["seed"], manifest["relations"], manifest["split"]) == (
        seed,
        count,
        design["split"],
    ), path
    assert manifest["clock"] == "stage_full_window", path
    expected = {}
    for relative, sha in design["source_sha256"].items():
        file = Path(relative)
        key = "settlement/core.py" if file.parent.name == "settlement" else file.name
        expected[key] = sha
    assert manifest["source_sha256"] == expected, path
    result = json.loads((path / "summary.json").read_text())
    assert set(result["exact_value"]) == set(design["policies"]), path
    for value in result["exact_value"].values():
        assert math.isfinite(value), path
    assert math.isclose(
        result["planner_value"],
        result["exact_value"]["optimal"],
        rel_tol=1e-10,
        abs_tol=1e-10,
    ), path
    if count:
        assert set(result["sample"]) == set(design["policies"]), path
    return result


def execute(path: Path, cell: dict, design: dict, seed: int, count: int) -> None:
    """Resume only a previously complete matching run."""
    if path.exists():
        verify_run(path, cell, design, seed, count)
        print(f"verified existing: {path.name}", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "arena.experiments.settlement_learning.run",
            "--config",
            str(REPO / cell["path"]),
            "--output",
            str(path),
            "--relations",
            str(count),
            "--seed",
            str(seed),
            "--split",
            design["split"],
        ],
        cwd=REPO,
        check=True,
    )
    verify_run(path, cell, design, seed, count)


def report(output: Path, design: dict) -> dict:
    """Summarize all cells, including zero and unfavorable baseline differences."""
    exact_rows = []
    sampling_rows = []
    family_size = sum(c["replay"] for c in design["cells"]) * len(design["policies"])
    for cell in design["cells"]:
        values = verify_run(
            output / "exact" / cell["id"], cell, design, design["seeds"][0], 0
        )["exact_value"]
        scale = cell["expected_exposure"]
        tolerance = 1e-10 * (1 + scale)
        comparisons = {}
        for policy in design["policies"]:
            difference = values["optimal"] - values[policy]
            if difference < -tolerance:
                raise RuntimeError(f"baseline exceeds optimum: {cell['id']} {policy}")
            category = (
                "numerical_tie"
                if abs(difference) <= tolerance
                else "below_declared_margin"
                if difference <= cell["practical_delta"]
                else "above_declared_margin"
            )
            comparisons[policy] = dict(
                difference=difference,
                normalized_difference=difference / scale,
                category=category,
            )
        exact_rows.append(
            dict(
                cell=cell["id"],
                groups=cell["groups"],
                values=values,
                comparisons=comparisons,
                practical_delta=cell["practical_delta"],
            )
        )
        if not cell["replay"]:
            continue
        records = {p: {} for p in design["policies"]}
        for seed in design["seeds"]:
            path = output / "replay" / cell["id"] / f"seed-{seed}"
            replay_result = verify_run(
                path, cell, design, seed, design["relations_per_seed"]
            )
            assert all(
                math.isclose(
                    values[p],
                    replay_result["exact_value"][p],
                    abs_tol=tolerance,
                    rel_tol=1e-10,
                )
                for p in values
            )
            per_seed = {p: [] for p in records}
            for line in (path / "private-metrics.jsonl").read_text().splitlines():
                row = json.loads(line)
                policy = row.pop("policy")
                key = (seed, row.pop("relation_id"))
                assert policy in records and key not in records[policy], path
                assert all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in row.values()
                ), path
                records[policy][key] = row
                per_seed[policy].append(row)
            for policy, rows in per_seed.items():
                assert len(rows) == design["relations_per_seed"], path
                for metric, mean in replay_result["sample"][policy]["means"].items():
                    assert math.isclose(
                        statistics.mean(r[metric] for r in rows),
                        mean,
                        abs_tol=1e-10,
                        rel_tol=1e-10,
                    ), path
        config = json.loads((REPO / cell["path"]).read_text())
        lower = sum(
            min(
                -max(1, config["harm"]) * x["amount"]
                - config["query_cost"]
                - config["wait_cost"]
                * x["amount"]
                * (len(x["hazards"]) + config["deadline"])
                for x in dist
            )
            for dist in config["schedule"]
        )
        upper = sum(
            max(config["margin"] * x["amount"] for x in dist)
            for dist in config["schedule"]
        )
        keys = {
            (seed, i)
            for seed in design["seeds"]
            for i in range(design["relations_per_seed"])
        }
        for policy, rows in records.items():
            assert set(rows) == keys
            rewards = [rows[key]["reward"] for key in sorted(keys)]
            assert all(lower - tolerance <= x <= upper + tolerance for x in rewards)
            n = len(rewards)
            mean = statistics.mean(rewards)
            halfwidth = (upper - lower) * math.sqrt(
                math.log(2 * family_size / design["statistical_family_alpha"]) / (2 * n)
            )
            differences = [
                records["optimal"][key]["reward"] - rows[key]["reward"]
                for key in sorted(keys)
            ]
            sampling_rows.append(
                dict(
                    cell=cell["id"],
                    policy=policy,
                    relations=n,
                    exact=values[policy],
                    mean=mean,
                    se=statistics.stdev(rewards) / math.sqrt(n),
                    simultaneous_hoeffding_halfwidth=halfwidth,
                    exact_inside_bound=abs(mean - values[policy])
                    <= halfwidth + tolerance,
                    paired_mean=statistics.mean(differences),
                    paired_se=statistics.stdev(differences) / math.sqrt(n),
                    normal_rejection_rate=(
                        sum(row["normal_rejections"] for row in rows.values())
                        / sum(row["normal_requests"] for row in rows.values())
                        if sum(row["normal_requests"] for row in rows.values())
                        else None
                    ),
                    metric_means={
                        metric: statistics.mean(row[metric] for row in rows.values())
                        for metric in next(iter(rows.values()))
                    },
                )
            )
    return dict(
        exact=exact_rows,
        sampling=sampling_rows,
        family_size=family_size,
        family_alpha=design["statistical_family_alpha"],
        passed_sampling_bounds=all(r["exact_inside_bound"] for r in sampling_rows),
    )


def main() -> None:
    """Run each phase explicitly; do not launch replay during exact checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("exact", "replay", "report"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("do not disable assertions with python -O")
    design = json.loads(DESIGN.read_text())
    for relative, expected in design["source_sha256"].items():
        if digest(REPO / relative) != expected:
            raise RuntimeError(f"source changed after design freeze: {relative}")
    for cell in design["cells"]:
        if digest(REPO / cell["path"]) != cell["sha256"]:
            raise RuntimeError(f"configuration changed: {cell['id']}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_once(
        output / "suite.json",
        dict(
            design_sha256=digest(DESIGN),
            runner_sha256=digest(Path(__file__)),
            design=design,
        ),
    )
    if args.phase == "exact":
        for i, cell in enumerate(design["cells"], 1):
            print(f"exact {i}/{len(design['cells'])}: {cell['id']}", flush=True)
            execute(output / "exact" / cell["id"], cell, design, design["seeds"][0], 0)
    elif args.phase == "replay":
        for cell in design["cells"]:
            verify_run(
                output / "exact" / cell["id"], cell, design, design["seeds"][0], 0
            )
        for cell in design["cells"]:
            if cell["replay"]:
                for seed in design["seeds"]:
                    print(f"replay: {cell['id']} seed {seed}", flush=True)
                    execute(
                        output / "replay" / cell["id"] / f"seed-{seed}",
                        cell,
                        design,
                        seed,
                        design["relations_per_seed"],
                    )
    else:
        result = report(output, design)
        write_once(output / "report.json", result)
        print(output / "report.json")
        if not result["passed_sampling_bounds"]:
            raise SystemExit(
                "one or more sampling checks need investigation; report retained"
            )


if __name__ == "__main__":
    main()
