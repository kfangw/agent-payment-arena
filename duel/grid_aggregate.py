"""Aggregate S1: the A2 - B4 table at the refined suspicion grid.

Reads the base duel_*.json for A2 block sums (family A is not recomputed),
the b4_gridN_*.json for B4 at the finest grid, and the original b4_*.json
for the holdout-tuned B4 at n = 21, so the verdict change from grid
refinement can be read off directly.  The interval is the nine-cell
simultaneous 95% (Bonferroni: per-cell percentiles at alpha/2 = 0.278% and
99.722%, bootstrap 20,000), which is wider than the per-cell interval, so a
verdict that survives it is the stronger claim.  Holm still corrects the
nine A2 - B4 permutation p values as their own family.

Run:  python -m duel.grid_aggregate --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b4_aggregate import CELL_SEED, _tag
from .design import NINE_CELLS, eps_for
from .naming import canon_keys
from .stats import boot_ci, holm, perm_p, ratio_mean, units, verdict

ALPHA = 0.05
N_CELLS = 9
SIM_LEVEL = 1.0 - ALPHA / N_CELLS      # per-cell coverage for Bonferroni
N_BOOT = 20_000


def _load(cell, results, results_b4):
    env, flow = _tag(cell)
    seed = CELL_SEED[cell]
    base = json.load(open(Path(results) / f"duel_{env}_{flow}_mid_s{seed}.json"))
    b4g = json.load(open(Path(results_b4) / f"b4_gridN_{env}_{flow}_mid_s{seed}.json"))
    b4o = json.load(open(Path(results_b4) / f"b4_{env}_{flow}_mid_s{seed}.json"))
    grid = json.load(open(Path(results) / f"duel_gridN_{env}_{flow}_mid_s{seed}.json"))
    bp, gp, op = base["payload"], b4g["payload"], b4o["payload"]
    a2 = np.asarray(canon_keys(bp["policies"])["A"]["block_sums"], dtype=float)
    b4 = np.asarray(gp["b4_block_sums"], dtype=float)
    b4orig = np.asarray(op["b4_block_sums"], dtype=float)
    counts = np.asarray(gp["block_counts"], dtype=float)
    return dict(
        cell=cell, seed=seed, a2=a2, b4=b4, b4orig=b4orig, counts=counts,
        A2=canon_keys(bp["means"])["A"], B1=bp["means"]["B1"],
        B4=gp["b4_mean"], k=gp["b4"]["k"], a=gp["b4"]["a"], b=gp["b4"]["b"],
        b3_n=gp.get("b3_n"), k0_gap=gp["k0_identity_max_gap"],
        mean_exposure=gp["mean_exposure"],
        orig_verdict=None,  # filled after the original run is re-judged below
        orig_B4=op["b4_mean"], orig_k=op["b4"]["k"],
        grid_curve=gp["b4"], b3_curve=grid["payload"]["curve"],
        base_hash=base["params_hash"], b4g_hash=b4g["params_hash"],
        b4g_code=b4g["code"],
        b4g_path=str(Path(results_b4) / f"b4_gridN_{env}_{flow}_mid_s{seed}.json"),
    )


def build(results="results", results_b4="results"):
    rows = [_load(c, results, results_b4) for c in NINE_CELLS]
    # simultaneous interval and permutation p on the refined-grid A2 - B4
    perm = []
    for r in rows:
        diff = r["a2"] - r["b4"]
        r["diff"] = ratio_mean(diff, r["counts"])
        r["ci"] = list(boot_ci(diff, r["counts"], n_boot=N_BOOT,
                               seed=r["seed"], level=SIM_LEVEL))
        r["perm_p"] = perm_p(diff, r["counts"], seed=r["seed"] + 1)
        perm.append(r["perm_p"])
    for r, ph in zip(rows, holm(perm)):
        eps = eps_for(r["mean_exposure"])
        r["holm_p"] = ph
        r["eps"] = eps
        r["bp"] = units(r["diff"], r["mean_exposure"])["bp"]
        r["verdict"] = verdict(r["ci"][0], r["ci"][1], ph, eps)
    # original (n=21) A2 - B4 under the SAME simultaneous rule, so the only
    # thing that differs between the two verdicts is the grid resolution.
    orig_perm = []
    for r in rows:
        d0 = r["a2"] - r["b4orig"]
        r["orig_diff"] = ratio_mean(d0, r["counts"])
        r["orig_ci"] = list(boot_ci(d0, r["counts"], n_boot=N_BOOT,
                                    seed=r["seed"], level=SIM_LEVEL))
        r["orig_bp"] = units(r["orig_diff"], r["mean_exposure"])["bp"]
        orig_perm.append(perm_p(d0, r["counts"], seed=r["seed"] + 1))
    for r, ph in zip(rows, holm(orig_perm)):
        r["orig_holm_p"] = ph
        r["orig_verdict"] = verdict(r["orig_ci"][0], r["orig_ci"][1], ph,
                                    r["eps"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--results-b4", default="results")
    args = ap.parse_args(argv)
    rows = build(args.results, args.results_b4)
    hdr = ("cell", "A2", "B1", "B4", "k*", "A2-B4", "sim95", "bp",
           "holm_p", "verdict", "was(n21)")
    print("%-14s %7s %7s %7s %3s %8s %18s %7s %8s %-11s %s" % hdr)
    for r in rows:
        print("%-14s %7.3f %7.3f %7.3f %3d %8.4f [%7.4f,%7.4f] %7.2f %8.4f %-11s %s"
              % (r["cell"], r["A2"], r["B1"], r["B4"], r["k"], r["diff"],
                 r["ci"][0], r["ci"][1], r["bp"], r["holm_p"], r["verdict"],
                 r["orig_verdict"]))
    print(f"\nsimultaneous level per cell = {SIM_LEVEL:.5f} "
          f"(percentiles {(1-SIM_LEVEL)/2*100:.3f}% / {(1+SIM_LEVEL)/2*100:.3f}%), "
          f"boot {N_BOOT}")
    gaps = [r["k0_gap"] for r in rows]
    print(f"k=0 identity max gap over cells: {max(gaps):.3e}")
    flips = [(r["cell"], r["orig_verdict"], r["verdict"])
             for r in rows if r["orig_verdict"] != r["verdict"]]
    print(f"verdict changes: {flips if flips else 'none'}")


if __name__ == "__main__":
    main()
