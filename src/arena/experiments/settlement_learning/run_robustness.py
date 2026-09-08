"""Run a frozen exact robustness suite with resumable, separate result files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from .model import Request
from .robustness import Evaluator, Planner, tune
from .run import read_setting, source_metadata
from .solver import MODES


def with_hazards(setting, change):
    """Change only hazard values, preserving observable support."""
    return replace(
        setting,
        schedule=tuple(
            tuple((w, Request(r.amount, change(r.hazards))) for w, r in dist)
            for dist in setting.schedule
        ),
    )


def design(config_dir):
    """Build the predeclared 18 bases and deduplicated misspecification cases."""
    cases = []
    for anchor in ("uncertain", "pilot"):
        original = read_setting(config_dir / f"{anchor}.json")
        for rail, hazards in (
            ("final", ()),
            ("fast", (0.005,) * 4),
            ("persistent", tuple(0.06 * 0.5**i for i in range(4))),
        ):
            for harm in (0.5, 1.0, 2.0):
                actual = replace(with_hazards(original, lambda _: hazards), harm=harm)
                base = f"{anchor}-{rail}-h{harm:g}"
                variants = [
                    ("matched", actual),
                    ("prior025", replace(actual, prior=0.25)),
                    ("prior075", replace(actual, prior=0.75)),
                    (
                        "type_compressed",
                        replace(
                            actual,
                            low=(actual.low + 0.5) / 2,
                            high=(actual.high + 0.5) / 2,
                        ),
                    ),
                    (
                        "response05",
                        replace(actual, response_rate=actual.response_rate * 0.5),
                    ),
                    (
                        "response15",
                        replace(
                            actual, response_rate=min(1, actual.response_rate * 1.5)
                        ),
                    ),
                    (
                        "hazard05",
                        with_hazards(actual, lambda fs: tuple(f * 0.5 for f in fs)),
                    ),
                    (
                        "hazard2",
                        with_hazards(
                            actual, lambda fs: tuple(min(1, f * 2) for f in fs)
                        ),
                    ),
                ]
                unique = {}
                for name, assumed in variants:
                    if assumed in unique:
                        unique[assumed]["tags"].append(name)
                    else:
                        item = {
                            "id": f"{base}-{name}",
                            "base": base,
                            "tags": [name],
                            "actual": actual,
                            "assumed": assumed,
                            "actual_accounting": "basic",
                            "planning_accounting": "basic",
                        }
                        unique[assumed] = item
                        cases.append(item)
                for kind in ("refit", "held"):
                    cases.append(
                        {
                            "id": f"{base}-additive-{kind}",
                            "base": base,
                            "tags": [f"additive_{kind}"],
                            "actual": actual,
                            "assumed": actual,
                            "actual_accounting": "additive",
                            "planning_accounting": "additive"
                            if kind == "refit"
                            else "basic",
                        }
                    )
    return cases


def serial(case):
    """Serialize both models so later runs do not infer actual from assumed."""
    return {k: asdict(v) if k in ("actual", "assumed") else v for k, v in case.items()}


def save(path, value):
    """Replace only a run's own control file atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    """Explicit execution entry point; importing this module runs no evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", type=Path, default=Path("configs/settlement_learning")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", nargs="+", type=int, default=[21, 161])
    parser.add_argument("--case-prefix", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write configuration inventory without evaluation",
    )
    args = parser.parse_args()
    if any(p < 2 for p in args.points) or len(set(args.points)) != len(args.points):
        parser.error("grid sizes must be distinct integers >= 2")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    cases = [c for c in design(args.configs) if c["id"].startswith(args.case_prefix)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("no cases selected")
    payload = {
        "cases": [serial(c) for c in cases],
        "points": args.points,
        "policies": [*MODES, "no_verify", "no_wait"],
        "threshold_families": [
            "B1",
            "B2",
            "B3_prior",
            "B3_posterior",
            "B4_prior",
            "B4_posterior",
        ],
        "candidate_ties": "within 1e-10*(1+expected exposure) of maximum",
        "selection": "largest computed reward; first encountered exact tie",
        "data_grade": "synthetic_exact",
        "new_replay": False,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    metadata = source_metadata()
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if (
            manifest["design_sha256"] != digest
            or manifest["source_sha256"] != metadata["source_sha256"]
        ):
            parser.error("resume rejected: source or design changed")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "design.json", payload)
        manifest = dict(
            status="planned", design_sha256=digest, completed_cases=0, **metadata
        )
        save(args.output / "manifest.json", manifest)
    if args.plan_only:
        print(f"{len(cases)} cases; grids {args.points}; no evaluation", flush=True)
        return
    manifest["status"] = "running"
    save(args.output / "manifest.json", manifest)
    tuned = {}
    oracles = {}
    current_base = None
    try:
        for index, case in enumerate(cases):
            if current_base != case["base"]:
                tuned.clear()
                current_base = case["base"]
            path = args.output / (case["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("status") != "complete":
                    raise ValueError(f"incomplete result file: {path}")
                continue
            start = time.monotonic()
            actual, assumed = case["actual"], case["assumed"]
            accounting, planned = case["actual_accounting"], case["planning_accounting"]
            exposure = sum(sum(w * r.amount for w, r in d) for d in actual.schedule)
            oracle_key = (actual, accounting)
            if oracle_key not in oracles:
                oracle = Planner(actual, accounting=accounting)
                truth = Evaluator(actual, oracle, accounting)
                optimal = truth.future()[0]
                if not math.isclose(
                    optimal, oracle.future(0, 0, 0), abs_tol=1e-10, rel_tol=1e-10
                ):
                    raise RuntimeError("oracle planning and evaluation disagree")
                oracles[oracle_key] = optimal
            optimal = oracles[oracle_key]
            rows = []
            ties = {}

            def record(name, policy, grid=None, fit=None):
                values = Evaluator(actual, policy, accounting).report()
                gap = optimal - values["reward"]
                if gap < -1e-9 * (1 + exposure):
                    raise RuntimeError(f"{name} exceeds actual-model optimum")
                rows.append(
                    dict(
                        policy=name,
                        grid_points=grid,
                        **values,
                        oracle_regret=gap,
                        normalized_regret=gap / exposure,
                        fit=fit,
                    )
                )

            for mode in (*MODES, "no_verify", "no_wait"):
                record(mode, Planner(assumed, mode, planned))
            for points in args.points:
                for family in ("B1", "B2", "B3", "B4"):
                    for posterior in (
                        (False, True) if family in ("B3", "B4") else (False,)
                    ):
                        name = family + ("_posterior" if posterior else "_prior")
                        key = (assumed, family, posterior, points, planned)
                        if key in tuned:
                            policy, fit = tuned[key]
                        else:
                            policy, fit = tune(
                                assumed, family, posterior, points, planned
                            )
                            if assumed == actual and planned == "basic":
                                tuned[key] = (policy, fit)
                        ident = f"{name}-{points}"
                        ties[ident] = fit["tied_parameters"]
                        compact = {
                            k: v for k, v in fit.items() if k != "tied_parameters"
                        }
                        compact["tied_parameter_count"] = len(fit["tied_parameters"])
                        record(name, policy, points, compact)
            with gzip.open(
                args.output / (case["id"] + "-ties.json.gz"), "wt"
            ) as stream:
                json.dump(ties, stream, allow_nan=False)
            output = {
                "status": "complete",
                "case": serial(case),
                "oracle_reward": optimal,
                "exposure": exposure,
                "rows": rows,
                "elapsed_seconds": time.monotonic() - start,
            }
            save(path, output)
            manifest["completed_cases"] = index + 1
            save(args.output / "manifest.json", manifest)
            print(
                f"{index + 1}/{len(cases)} {case['id']}: {output['elapsed_seconds']:.2f}s",
                flush=True,
            )
        manifest["status"] = "complete"
        manifest["completed_cases"] = len(cases)
        save(args.output / "manifest.json", manifest)
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        save(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
