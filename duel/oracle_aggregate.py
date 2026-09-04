"""Aggregate S2: A2 - B4_oracle and the winner's curse B4_oracle - B4.

B4_oracle tunes (k, a, b) on the evaluation split, so it is an upper bound,
not a policy: no holdout, not deployable.  Two quantities per cell.  First,
A2 - B4_oracle under the same nine-cell simultaneous interval as S1: if the
edge survives against the best the grid can do on the eval split itself, it
is not an artifact of the small tuning split.  Second, B4_oracle - B4, the
realized size of the winner's curse, which must be non-negative in every
cell (the oracle maximizes the same eval mean B4 is scored on).

Run:  python -m duel.oracle_aggregate --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b4_aggregate import CELL_SEED, split_cell
from .design import NINE_CELLS, eps_for
from .grid_aggregate import N_BOOT, SIM_LEVEL
from .naming import canon_keys
from .stats import boot_ci, holm, perm_p, ratio_mean, units, verdict


def _load(cell, results):
    env, flow = split_cell(cell)
    seed = CELL_SEED[cell]
    base = json.load(open(Path(results) / f"duel_{env}_{flow}_mid_s{seed}.json"))
    orc = json.load(open(Path(results) / f"b4oracle_{env}_{flow}_mid_s{seed}.json"))
    b4 = json.load(open(Path(results) / f"b4_{env}_{flow}_mid_s{seed}.json"))
    bp, op, hp = base["payload"], orc["payload"], b4["payload"]
    return dict(
        cell=cell,
        seed=seed,
        a2=np.asarray(canon_keys(bp["policies"])["A"]["block_sums"], dtype=float),
        oracle=np.asarray(op["b4_block_sums"], dtype=float),
        b4=np.asarray(hp["b4_block_sums"], dtype=float),
        counts=np.asarray(op["block_counts"], dtype=float),
        A2=canon_keys(bp["means"])["A"],
        B4=hp["b4_mean"],
        B4_oracle=op["b4_mean"],
        k=op["b4"]["k"],
        a=op["b4"]["a"],
        b=op["b4"]["b"],
        k_holdout=hp["b4"]["k"],
        k0_gap=op["k0_identity_max_gap"],
        mean_exposure=op["mean_exposure"],
        split=op.get("tune_split"),
        base_hash=base["params_hash"],
        oracle_hash=orc["params_hash"],
        oracle_code=orc["code"],
        oracle_path=str(Path(results) / f"b4oracle_{env}_{flow}_mid_s{seed}.json"),
    )


def build(results="results_s2"):
    rows = [_load(c, results) for c in NINE_CELLS]
    perm = []
    for r in rows:
        diff = r["a2"] - r["oracle"]
        r["diff"] = ratio_mean(diff, r["counts"])
        r["ci"] = list(boot_ci(diff, r["counts"], n_boot=N_BOOT, seed=r["seed"], level=SIM_LEVEL))
        r["perm_p"] = perm_p(diff, r["counts"], seed=r["seed"] + 1)
        r["bp"] = units(r["diff"], r["mean_exposure"])["bp"]
        r["curse"] = r["B4_oracle"] - r["B4"]
        r["curse_bp"] = units(r["curse"], r["mean_exposure"])["bp"]
        perm.append(r["perm_p"])
    for r, ph in zip(rows, holm(perm)):
        eps = eps_for(r["mean_exposure"])
        r["holm_p"] = ph
        r["eps"] = eps
        r["verdict"] = verdict(r["ci"][0], r["ci"][1], ph, eps)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_s2")
    args = ap.parse_args(argv)
    rows = build(args.results)
    hdr = ("cell", "A2", "B4", "B4orc", "k", "curse", "cbp", "A2-B4orc", "sim95", "bp", "verdict")
    print("%-14s %7s %7s %7s %3s %8s %6s %9s %18s %7s %s" % hdr)
    for r in rows:
        print(
            "%-14s %7.3f %7.3f %7.3f %3d %8.4f %6.2f %9.4f [%7.4f,%7.4f] %7.2f %s"
            % (
                r["cell"],
                r["A2"],
                r["B4"],
                r["B4_oracle"],
                r["k"],
                r["curse"],
                r["curse_bp"],
                r["diff"],
                r["ci"][0],
                r["ci"][1],
                r["bp"],
                r["verdict"],
            )
        )
    neg = [(r["cell"], r["curse"]) for r in rows if r["curse"] < -1e-12]
    print(
        f"\nB4_oracle >= B4 in all nine cells: {not neg}" + (f"  VIOLATIONS {neg}" if neg else "")
    )
    print(f"max winner's curse: {max(r['curse_bp'] for r in rows):.2f} bp")
    print(f"k=0 identity max gap over cells: {max(r['k0_gap'] for r in rows):.3e}")


if __name__ == "__main__":
    main()
