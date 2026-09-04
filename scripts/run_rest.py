"""Run S2, S3, and S4 under one shared worker pool, at a reduced evaluation
sample so pure-Python replay stays tractable.

The confirmatory sample (5.66M) makes a single family-A replay minutes long
because the verify window iterates to tau; at n_eval=100,000 the same replay
is seconds.  S2, S3, and S4 therefore run at 100,000 eval / 50,000 tune
(inject.py's own defaults).  The intervals are wider than the confirmatory
ones and the levels are re-anchored to this sample, so each item's identity
check is against a base computed at the SAME size, not the 5.66M Table 9.

  S2  self-contained in results_s2/: duel.run (reduced base) -> b4 (holdout)
      -> b4 --oracle, all reading that reduced base so the k=0 identity holds.
  S4  in results/: duel.run E-slow-deep + b4 on the dense k ladder.
  S3  in results_inject/: inject_b4 (builds its own context, no base file).

One pool, numpy threads pinned to one per process, heaviest jobs first.
Resumable: a job whose final output exists is skipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"
PY = sys.executable
N_EVAL, N_TUNE = "100000", "50000"

NINE = [
    ("E-fast", "F1", 1),
    ("E-fast", "F2", 2),
    ("E-fast", "F3", 3),
    ("E-slow", "F1", 4),
    ("E-slow", "F2", 5),
    ("E-slow", "F3", 6),
    ("E-outage", "F1", 7),
    ("E-outage", "F2", 8),
    ("E-outage", "F3", 9),
]
S3_CELLS = [("E-slow", "F2", 5), ("E-outage", "F2", 8)]
S3_AXES = ["delta", "kappa", "lambda", "noise"]
S4_CELLS = [("E-slow-deep", "F1", 10), ("E-slow-deep", "F2", 11), ("E-slow-deep", "F3", 12)]
K_DENSE = ",".join(str(k) for k in range(13))


def _exists(sub, name):
    return (ROOT / sub / f"{name}.json").exists()


def _duel(env, flow, seed, out):
    return [
        PY,
        "-m",
        "duel.run",
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
    ]


def jobs():
    out = []
    # S2: reduced base + holdout B4 + oracle B4, matched draws in results_s2
    for env, flow, seed in NINE:
        tag = f"{env}_{flow}_mid_s{seed}"
        if _exists("results_s2", f"b4oracle_{tag}"):
            continue
        seq = [
            _duel(env, flow, seed, "results_s2"),
            [
                PY,
                "-m",
                "duel.b4",
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
            ],
            [
                PY,
                "-m",
                "duel.b4",
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
            ],
        ]
        out.append((f"s2_{tag}", seq))
    # S3: injection axes against B4 (self-contained, reduced)
    for env, flow, seed in S3_CELLS:
        for axis in S3_AXES:
            tag = f"{env}_{flow}_{axis}_s{seed}"
            if not _exists("results_inject", f"b4_{tag}"):
                out.append(
                    (
                        f"s3_{tag}",
                        [
                            [
                                PY,
                                "-m",
                                "duel.inject_b4",
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
                            ]
                        ],
                    )
                )
    # S4: E-slow-deep reduced base then dense-k B4
    for env, flow, seed in S4_CELLS:
        tag = f"{env}_{flow}_mid_s{seed}"
        seq = []
        if not _exists("results", f"duel_{tag}"):
            seq.append(_duel(env, flow, seed, "results"))
        if not _exists("results", f"b4_{tag}"):
            seq.append(
                [
                    PY,
                    "-m",
                    "duel.b4",
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
                ]
            )
        if seq:
            out.append((f"s4_{tag}", seq))

    def cost(name):
        outage = "E-outage" in name or "outage" in name
        if name.startswith("s3") and "noise" in name:
            return 100 if not outage else 60  # chain noise (tau=300) is heaviest
        if name.startswith("s3"):
            return 40 if outage else 20
        if name.startswith("s2"):
            return 30 if outage else 15
        return 10  # s4

    out.sort(key=lambda j: cost(j[0]), reverse=True)
    return out


def run_one(name, cmds):
    penv = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{name}.log").open("w") as fh:
        for cmd in cmds:
            rc = subprocess.run(
                cmd, cwd=ROOT, env=penv, stdout=fh, stderr=subprocess.STDOUT
            ).returncode
            if rc != 0:
                return name, rc
    return name, 0


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    todo = jobs()
    print(f"launching {len(todo)} jobs on {workers} workers", flush=True)
    ok, bad = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, n, c): n for n, c in todo}
        for fut in as_completed(futs):
            name, rc = fut.result()
            (ok if rc == 0 else bad).append(name)
            print(f"done {name} rc={rc}  ({len(ok) + len(bad)}/{len(todo)})", flush=True)
    print(f"\ncompleted ok={len(ok)} failed={len(bad)}", flush=True)
    if bad:
        print("FAILED:", bad, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
