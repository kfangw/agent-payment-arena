"""Run T5, T6, and T7 under one shared, thread-pinned worker pool.

T5: (C, h, m) one-at-a-time on the three E-outage cells (B4 competitor).
T6: (f0, gamma) grid on the three E-slow cells (B1 competitor).
T7: three arrival shapes on the two main cells.
All at the reduced sample; runners refuse to start on a dirty tree, so this
driver must be committed and run from a clean tree. Resumable.
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

OUTAGE = [("E-outage", "F1", 7), ("E-outage", "F2", 8), ("E-outage", "F3", 9)]
ESLOW = [("E-slow", "F1", 4), ("E-slow", "F2", 5), ("E-slow", "F3", 6)]
MAIN = [("E-slow x F2", 5), ("E-outage x F2", 8)]

# T5: base plus one-at-a-time; base is (0.5, 1.0, 0.35)
T5_POINTS = [dict(C=0.5, h=1.0, m=0.35),
             dict(C=0.25, h=1.0, m=0.35), dict(C=1.0, h=1.0, m=0.35),
             dict(C=0.5, h=0.5, m=0.35), dict(C=0.5, h=2.0, m=0.35),
             dict(C=0.5, h=1.0, m=0.25), dict(C=0.5, h=1.0, m=0.50)]
T6_POINTS = [dict(f0=f0, gamma=g) for f0 in (0.004, 0.015, 0.06)
             for g in (0.1, 0.3, 0.5)]
SHAPES = ["geometric", "uniform", "bimodal"]


def _out(sub, name):
    return (ROOT / sub / f"{name}.json").exists()


def jobs():
    out = []
    for env, flow, seed in OUTAGE:
        for p in T5_POINTS:
            tag = f"{env}_{flow}_C{p['C']}_h{p['h']}_m{p['m']}_f0.06_g0.5_s{seed}"
            if _out("results_t5", f"sweep_{tag}"):
                continue
            out.append((f"t5_{env}_{flow}_C{p['C']}h{p['h']}m{p['m']}",
                        [PY, "-m", "duel.sweep", "--env", env, "--flow", flow,
                         "--seed", str(seed), "--C", str(p["C"]), "--h",
                         str(p["h"]), "--m", str(p["m"]), "--competitor", "B4",
                         "--out", "results_t5"]))
    for env, flow, seed in ESLOW:
        for p in T6_POINTS:
            tag = f"{env}_{flow}_C0.5_h1.0_m0.35_f{p['f0']}_g{p['gamma']}_s{seed}"
            if _out("results_t6", f"sweep_{tag}"):
                continue
            out.append((f"t6_{env}_{flow}_f{p['f0']}g{p['gamma']}",
                        [PY, "-m", "duel.sweep", "--env", env, "--flow", flow,
                         "--seed", str(seed), "--f0", str(p["f0"]), "--gamma",
                         str(p["gamma"]), "--competitor", "B1", "--out",
                         "results_t6"]))
    for cell, seed in MAIN:
        e, f = (s.strip() for s in cell.split("x"))
        for shape in SHAPES:
            if _out("results_t7", f"shape_{e}_{f}_{shape}_s{seed}"):
                continue
            out.append((f"t7_{e}_{f}_{shape}",
                        [PY, "-m", "duel.shape", "--cell", cell, "--shape",
                         shape, "--seed", str(seed), "--out", "results_t7"]))
    return out


def run_one(name, cmd):
    penv = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1")
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{name}.log").open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=penv, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    return name, rc


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
            print(f"done {name} rc={rc}  ({len(ok)+len(bad)}/{len(todo)})",
                  flush=True)
    print(f"\ncompleted ok={len(ok)} failed={len(bad)}", flush=True)
    if bad:
        print("FAILED:", bad, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
