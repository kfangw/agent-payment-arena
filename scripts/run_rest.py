"""Run S2, S3, and S4 under one shared worker pool, at a reduced evaluation
sample so pure-Python replay stays tractable.

The confirmatory sample (5.66M) makes a single family-A replay minutes long
because the verify window iterates to tau; at n_eval=100,000 the same replay
is seconds.  S2, S3, and S4 therefore run at 100,000 eval / 50,000 tune
(inject.py's own defaults).  The intervals are wider than the confirmatory
ones and the levels are re-anchored to this sample, so each item's identity
check is against a base computed at the SAME size, not the 5.66M Table 9.

  S2  self-contained in results_s2/: arena.experiments.settlement.run (reduced base) -> b4 (holdout)
      -> b4 --oracle, all reading that reduced base so the k=0 identity holds.
  S4  in results/: arena.experiments.settlement.run E-slow-deep + b4 on the dense k ladder.
  S3  in results_inject/: inject_b4 (builds its own context, no base file).

One pool, numpy threads pinned to one per process, heaviest jobs first.
Resumable: a job whose final output exists is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arena.experiments.runner import PipelineJob, default_workers, run_jobs
from arena.experiments.settlement.design import CONFIRMATORY_RUNS

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"
N_EVAL, N_TUNE = "100000", "50000"

S3_CELLS = [("E-slow", "F2", 5), ("E-outage", "F2", 8)]
S3_AXES = ["delta", "kappa", "lambda", "noise"]
S4_CELLS = [("E-slow-deep", "F1", 10), ("E-slow-deep", "F2", 11), ("E-slow-deep", "F3", 12)]
K_DENSE = ",".join(str(k) for k in range(13))


def _exists(sub, name):
    return (ROOT / sub / f"{name}.json").exists()


def _duel(env, flow, seed, out):
    return (
        "-m",
        "arena.experiments.settlement.run",
        "--env",
        env,
        "--flow",
        flow,
        "--cw",
        "mid",
        "--n-eval",
        N_EVAL,
        "--n-tune",
        N_TUNE,
        "--seed",
        str(seed),
        "--out",
        out,
    )


def jobs():
    out = []
    # S2: reduced base + holdout B4 + oracle B4, matched draws in results_s2
    for env, flow, seed in CONFIRMATORY_RUNS:
        tag = f"{env}_{flow}_mid_s{seed}"
        if _exists("results_s2", f"b4oracle_{tag}"):
            continue
        steps = [
            _duel(env, flow, seed, "results_s2"),
            (
                "-m",
                "arena.experiments.settlement.b4",
                "--env",
                env,
                "--flow",
                flow,
                "--seed",
                str(seed),
                "--n-eval",
                N_EVAL,
                "--n-tune",
                N_TUNE,
                "--results",
                "results_s2",
                "--out",
                "results_s2",
            ),
            (
                "-m",
                "arena.experiments.settlement.b4",
                "--env",
                env,
                "--flow",
                flow,
                "--seed",
                str(seed),
                "--n-eval",
                N_EVAL,
                "--n-tune",
                N_TUNE,
                "--oracle",
                "--name",
                "b4oracle",
                "--results",
                "results_s2",
                "--out",
                "results_s2",
            ),
        ]
        out.append(PipelineJob.python(f"s2_{tag}", *steps))
    # S3: injection axes against B4 (self-contained, reduced)
    for env, flow, seed in S3_CELLS:
        for axis in S3_AXES:
            tag = f"{env}_{flow}_{axis}_s{seed}"
            if not _exists("results_inject", f"b4_{tag}"):
                out.append(
                    PipelineJob.python(
                        f"s3_{tag}",
                        (
                            "-m",
                            "arena.experiments.settlement.inject_b4",
                            "--cell",
                            f"{env} x {flow}",
                            "--axis",
                            axis,
                            "--n-eval",
                            N_EVAL,
                            "--n-tune",
                            N_TUNE,
                            "--seed",
                            str(seed),
                            "--out",
                            "results_inject",
                        ),
                    )
                )
    # S4: E-slow-deep reduced base then dense-k B4
    for env, flow, seed in S4_CELLS:
        tag = f"{env}_{flow}_mid_s{seed}"
        steps = []
        if not _exists("results", f"settlement_{tag}"):
            steps.append(_duel(env, flow, seed, "results"))
        if not _exists("results", f"b4_{tag}"):
            steps.append(
                (
                    "-m",
                    "arena.experiments.settlement.b4",
                    "--env",
                    env,
                    "--flow",
                    flow,
                    "--seed",
                    str(seed),
                    "--n-eval",
                    N_EVAL,
                    "--n-tune",
                    N_TUNE,
                    "--k-grid",
                    K_DENSE,
                    "--out",
                    "results",
                )
            )
        if steps:
            out.append(PipelineJob.python(f"s4_{tag}", *steps))

    def cost(name):
        outage = "E-outage" in name or "outage" in name
        if name.startswith("s3") and "noise" in name:
            return 100 if not outage else 60  # chain noise (tau=300) is heaviest
        if name.startswith("s3"):
            return 40 if outage else 20
        if name.startswith("s2"):
            return 30 if outage else 15
        return 10  # s4

    out.sort(key=lambda job: cost(job.name), reverse=True)
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else default_workers()
    raise SystemExit(run_jobs(jobs(), root=ROOT, logs=LOGS, workers=workers).exit_code)


if __name__ == "__main__":
    main()
