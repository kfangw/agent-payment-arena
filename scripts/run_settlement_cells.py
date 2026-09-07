"""Run the core settlement cells from one resumable command."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.experiments.runner import PipelineJob, default_workers, run_jobs
from arena.experiments.settlement.design import CONFIRMATORY_RUNS

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"


def jobs(profile: str, output: str) -> list[PipelineJob]:
    """Build pending base and B5 jobs for one sample-size profile."""
    n_eval, n_tune = ((2_000, 500) if profile == "smoke" else (5_663_400, 200_000))
    b3_n = 21 if profile == "smoke" else 161
    boot = 100 if profile == "smoke" else 20_000
    perm = 100 if profile == "smoke" else 10_000
    root = ROOT / output
    pending: list[PipelineJob] = []
    for env, flow, seed in CONFIRMATORY_RUNS:
        tag = f"{env}_{flow}_mid_s{seed}"
        steps: list[tuple[str, ...]] = []
        base = root / "base" / f"settlement_{tag}.json"
        if not base.exists():
            steps.append((
                "-m", "arena.experiments.settlement.run", "--env", env,
                "--flow", flow, "--seed", str(seed), "--n-eval", str(n_eval),
                "--n-tune", str(n_tune), "--out", f"{output}/base",
            ))
        if env == "E-outage":
            b5 = root / "b5" / f"b5_E-outage_{flow}_mid_s{seed}.json"
            if not b5.exists():
                steps.append((
                    "-m", "arena.experiments.settlement.b5", "--flow", flow,
                    "--seed", str(seed), "--n-eval", str(n_eval),
                    "--n-tune", str(n_tune), "--b3-n", str(b3_n),
                    "--n-boot", str(boot), "--n-perm", str(perm),
                    "--out", f"{output}/b5",
                ))
        if steps:
            pending.append(PipelineJob.python(f"settlement_{tag}", *steps))
    return pending


def main() -> None:
    """Run pending cells with a bounded worker pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", default="results_settlement")
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()
    todo = jobs(args.profile, args.output)
    if not todo:
        print("all requested artifacts already exist")
        return
    raise SystemExit(run_jobs(todo, root=ROOT, logs=LOGS, workers=args.workers).exit_code)


if __name__ == "__main__":
    main()
