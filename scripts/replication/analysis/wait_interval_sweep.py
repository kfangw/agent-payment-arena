#!/usr/bin/env python3
"""Random-instance sweep of the waiting set on the stage-indexed chain.

Theorem 1 of the main article says nothing about the optimal set of the
wait action.  This script draws random stage chains, solves each one on a
501-point suspicion grid with the article's solver (core.best_action), and
counts the stage rows at which the set where waiting is optimal is not one
connected interval.  It also re-checks the three terminal clauses of the
theorem (grant a prefix, reject a nonempty suffix, verify an interval off
both endpoints) on the same rows.

Ranges (all uniform unless stated): stage count 1 to 4; each stage hazard
in [0, 0.2]; exposure log-uniform on [$0.05, $200]; m in [0.05, 1.5];
h in [0.1, 4]; C in [0.005, 1]; c_w in [0, 0.2]; rho in [0.05, 0.95];
tau an integer in 1..11.  Waiting is disallowed at FINAL, so FINAL rows
are not counted.

Run from code/:  PYTHONPATH=. python3 analysis/wait_interval_sweep.py [n] [seed]
"""
import json, sys, time
import numpy as np
from arena.experiments.settlement.core import best_action, GRANT, REJECT, VERIFY, WAIT

PI = np.linspace(0.0, 1.0, 501)


def is_interval(mask):
    idx = np.flatnonzero(mask)
    return len(idx) == 0 or idx[-1] - idx[0] + 1 == len(idx)


def main(n=60_000, seed=20260917):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    rows = wait_rows = wait_bad = grant_bad = reject_bad = verify_bad = 0
    stages_seen = np.zeros(5, dtype=int)
    for k in range(n):
        N = int(rng.integers(1, 5))
        f = rng.uniform(0.0, 0.2, size=N)
        v = float(np.exp(rng.uniform(np.log(0.05), np.log(200.0))))
        p = dict(m=float(rng.uniform(0.05, 1.5)), h=float(rng.uniform(0.1, 4.0)),
                 C=float(rng.uniform(0.005, 1.0)), cw=float(rng.uniform(0.0, 0.2)),
                 rho=float(rng.uniform(0.05, 0.95)), tau=int(rng.integers(1, 12)))
        lab, _ = best_action(f, v, PI, p)
        stages_seen[N] += 1
        for i in range(N):                # stages 0..N-1; FINAL (index N) excluded
            rows += 1
            w = lab[i] == WAIT
            g = lab[i] == GRANT
            r = lab[i] == REJECT
            ver = lab[i] == VERIFY
            if w.any():
                wait_rows += 1
                if not is_interval(w):
                    wait_bad += 1
            if g.any() and not (is_interval(g) and g[0]):
                grant_bad += 1
            if not (r.any() and is_interval(r) and r[-1]):
                reject_bad += 1
            if ver.any() and not (is_interval(ver) and not ver[0] and not ver[-1]):
                verify_bad += 1
    out = dict(instances=n, seed=seed, stage_rows=rows, rows_with_waiting=wait_rows,
               waiting_not_interval=wait_bad, grant_not_prefix=grant_bad,
               reject_not_suffix=reject_bad, verify_not_interior_interval=verify_bad,
               instances_by_stage_count={str(i): int(c) for i, c in enumerate(stages_seen) if c},
               pi_grid=len(PI), seconds=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260917
    main(n, seed)
