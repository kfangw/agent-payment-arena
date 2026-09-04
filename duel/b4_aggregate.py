"""Aggregate the nine B4 cells into the R1 table.

Reads b4_*.json for B4 and the A2 - B4 paired statistic, and the base
duel_*.json for the A2, B1, B3 means at the same seed.  Holm correction
is applied to the nine A2 - B4 comparisons as their own family, separate
from A2 - B1 (confirmatory) and A2 - B3 (exploratory): B4 is an added
exploratory contender, so its multiplicity is not pooled with theirs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .design import NINE_CELLS, eps_for
from .naming import canon_keys
from .stats import holm, units, verdict

# cell -> seed, fixed by the main run's assignment
CELL_SEED = {
    "E-fast x F1": 1,
    "E-fast x F2": 2,
    "E-fast x F3": 3,
    "E-slow x F1": 4,
    "E-slow x F2": 5,
    "E-slow x F3": 6,
    "E-outage x F1": 7,
    "E-outage x F2": 8,
    "E-outage x F3": 9,
}


def _tag(cell):
    env, flow = (s.strip() for s in cell.split("x"))
    return env, flow


def load(cell, results, results_b4):
    env, flow = _tag(cell)
    seed = CELL_SEED[cell]
    base = json.load(open(Path(results) / f"duel_{env}_{flow}_mid_s{seed}.json"))
    b4 = json.load(open(Path(results_b4) / f"b4_{env}_{flow}_mid_s{seed}.json"))
    bp = base["payload"]
    pp = b4["payload"]
    means = canon_keys(bp["means"])
    return dict(
        cell=cell,
        seed=seed,
        A2=means["A"],
        B1=means["B1"],
        B3=means["B3"],
        B4=pp["b4_mean"],
        k=pp["b4"]["k"],
        a=pp["b4"]["a"],
        b=pp["b4"]["b"],
        horizon=pp["horizon"],
        k_grid=pp["k_grid"],
        k0_gap=pp["k0_identity_max_gap"],
        diff=pp["a2_minus_b4"]["mean"],
        ci=pp["a2_minus_b4"]["ci95"],
        perm_p=pp["a2_minus_b4"]["perm_p"],
        mean_exposure=pp["mean_exposure"],
        base_code=base["code"],
        b4_code=b4["code"],
        base_hash=base["params_hash"],
        b4_hash=b4["params_hash"],
        b4_path=str(Path(results_b4) / f"b4_{env}_{flow}_mid_s{seed}.json"),
    )


def build(results="results", results_b4="results"):
    rows = [load(c, results, results_b4) for c in NINE_CELLS]
    adj = holm([r["perm_p"] for r in rows])
    for r, p in zip(rows, adj):
        r["holm_p"] = p
        eps = eps_for(r["mean_exposure"])
        r["eps"] = eps
        r["bp"] = units(r["diff"], r["mean_exposure"])["bp"]
        r["verdict"] = verdict(r["ci"][0], r["ci"][1], p, eps)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--results-b4", default="results")
    args = ap.parse_args(argv)
    rows = build(args.results, args.results_b4)
    hdr = ("cell", "A", "B1", "B3", "B4", "k*", "A-B4", "ci95", "bp", "holm_p", "verdict")
    print("%-14s %7s %7s %7s %7s %4s %8s %18s %7s %8s %s" % hdr)
    for r in rows:
        print(
            "%-14s %7.3f %7.3f %7.3f %7.3f %4d %8.4f [%7.4f,%7.4f] %7.2f %8.4f %s"
            % (
                r["cell"],
                r["A2"],
                r["B1"],
                r["B3"],
                r["B4"],
                r["k"],
                r["diff"],
                r["ci"][0],
                r["ci"][1],
                r["bp"],
                r["holm_p"],
                r["verdict"],
            )
        )
    gaps = [r["k0_gap"] for r in rows]
    print(f"\nk=0 identity max gap over cells: {max(gaps):.3e}")
    codes = {r["b4_code"] for r in rows}
    print(f"b4 code revisions: {codes}")


if __name__ == "__main__":
    main()
