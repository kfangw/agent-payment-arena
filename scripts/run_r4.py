"""Run R4 (residual diagnosis) on the three E-outage cells in parallel.

Each cell reproduces family A, marginalizes it over each hidden coordinate,
replays every ablation and B3 (n=161) on the confirmatory evaluation split,
and records the arrival-state disagreement map.  E-outage replay stays
tractable at the full sample (tau=10), so R4 runs at 5.66M to match the
residual it explains.  Resumable; numpy threads pinned; logs in reports/_logs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "_logs"
CELLS = [("E-outage", "F1", 7), ("E-outage", "F2", 8), ("E-outage", "F3", 9)]


def run_one(env, flow, seed):
    tag = f"{env}_{flow}_mid_s{seed}"
    if (ROOT / "results_r4" / f"r4_{tag}.json").exists():
        return tag, 0
    penv = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "duel.residual", "--env", env, "--flow", flow,
           "--seed", str(seed), "--results", "results", "--results-grid",
           "results", "--out", "results_r4"]
    with (LOGS / f"r4_{tag}.log").open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=penv, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    return tag, rc


def main():
    ok, bad = [], []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_one, e, f, s): f"{e}_{f}" for e, f, s in CELLS}
        for fut in as_completed(futs):
            tag, rc = fut.result()
            (ok if rc == 0 else bad).append(tag)
            print(f"done {tag} rc={rc}", flush=True)
    print(f"completed ok={len(ok)} failed={len(bad)}", flush=True)
    if bad:
        print("FAILED:", bad, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
