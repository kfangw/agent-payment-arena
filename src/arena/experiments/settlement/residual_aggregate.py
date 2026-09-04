"""Aggregate R4: the coordinate-ablation table and the disagreement map.

Reads r4_{cell}.json for the three E-outage cells.  For R4a each ablation
loss A - A_x is a paired block-sum difference; the interval is the
three-cell simultaneous 95% (Bonferroni).  The decomposition A - A-ilrv
(extra coordinates) plus A-ilrv - B3 (compilation of the same information)
equals A - B3 by construction; the leave-one-out losses do not add up, and
that gap is reported rather than hidden.  R4b reports the disagreement mass
and the top residual cells.

Run:  python -m arena.experiments.settlement.residual_aggregate --results results_r4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stats import boot_ci, ratio_mean, units

CELL_SEED = {"E-outage x F1": 7, "E-outage x F2": 8, "E-outage x F3": 9}
ABL = ["A-r", "A-l", "A-i", "A-v", "A-ilr", "A-ilrv"]
ACT = ["grant", "reject", "verify", "wait"]
ALPHA, N_CELLS, N_BOOT = 0.05, 3, 20_000
SIM_LEVEL = 1.0 - ALPHA / N_CELLS


def _load(cell, results):
    seed = CELL_SEED[cell]
    e, f = (s.strip() for s in cell.split("x"))
    d = json.load(open(Path(results) / f"r4_{e}_{f}_mid_s{seed}.json"))
    p = d["payload"]
    b = {k: np.asarray(v, dtype=float) for k, v in p["block_sums"].items()}
    counts = np.asarray(p["block_counts"], dtype=float)
    return dict(
        cell=cell,
        seed=seed,
        b=b,
        counts=counts,
        mexp=p["mean_exposure"],
        repro=p["a2_base_repro_gap"],
        r4b=p["r4b"],
        hash=d["params_hash"],
        code=d["code"],
        path=str(Path(results) / f"r4_{e}_{f}_mid_s{seed}.json"),
    )


def _loss(r, minus, sub):
    """mean and simultaneous CI of (minus - sub) block-sum difference."""
    diff = r["b"][minus] - r["b"][sub]
    mean = ratio_mean(diff, r["counts"])
    lo, hi = boot_ci(diff, r["counts"], n_boot=N_BOOT, seed=r["seed"], level=SIM_LEVEL)
    return dict(mean=mean, ci=[lo, hi], bp=units(mean, r["mexp"])["bp"])


def build(results="results_r4"):
    rows = []
    for cell in CELL_SEED:
        r = _load(cell, results)
        r["resid"] = _loss(r, "A", "B3")
        r["losses"] = {x: _loss(r, "A", x) for x in ABL}
        r["extra"] = _loss(r, "A", "A-ilrv")  # A - A-ilrv
        r["compile"] = _loss(r, "A-ilrv", "B3")  # A-ilrv - B3
        # additivity of the decomposition (identity) and leave-one-out gap
        r["decomp_sum"] = r["extra"]["mean"] + r["compile"]["mean"]
        r["loo_sum"] = sum(r["losses"][x]["mean"] for x in ("A-r", "A-l", "A-i", "A-v"))
        rows.append(r)
    return rows


def top_cells(r4b, k=10):
    cells = sorted(r4b["cells"], key=lambda c: abs(c["sum_diff"]), reverse=True)
    return cells[:k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_r4")
    args = ap.parse_args(argv)
    rows = build(args.results)

    print(
        "== R4a: ablation losses (A - A_x), $ per payment and bp, "
        f"3-cell simultaneous {SIM_LEVEL:.4f} =="
    )
    hdr = ["cell", "resid", "bp"] + ABL
    print("%-14s %8s %6s" % (hdr[0], "A-B3", "bp") + "".join("%9s" % x for x in ABL))
    for r in rows:
        print(
            "%-14s %8.4f %6.1f" % (r["cell"], r["resid"]["mean"], r["resid"]["bp"])
            + "".join("%9.4f" % r["losses"][x]["mean"] for x in ABL)
        )
    print("\n-- decomposition (identity): (A - A-ilrv) + (A-ilrv - B3) = A - B3 --")
    for r in rows:
        print(
            f"{r['cell']:14} extra(A-A_ilrv)={r['extra']['mean']:.4f} "
            f"[{r['extra']['ci'][0]:.4f},{r['extra']['ci'][1]:.4f}] {r['extra']['bp']:.1f}bp | "
            f"compile(A_ilrv-B3)={r['compile']['mean']:.4f} {r['compile']['bp']:.1f}bp | "
            f"sum={r['decomp_sum']:.4f} vs A-B3={r['resid']['mean']:.4f}"
        )
    print("\n-- leave-one-out sum (NOT additive, expected) --")
    for r in rows:
        print(
            f"{r['cell']:14} loo_sum(r+l+i+v)={r['loo_sum']:.4f} vs A-B3={r['resid']['mean']:.4f}"
        )

    print("\n== R4b: disagreement map ==")
    for r in rows:
        m = r["r4b"]
        print(
            f"\n{r['cell']}: disagree_mass={m['disagree_mass'] * 100:.1f}%  "
            f"disagree_resid={m['disagree_resid']:.2f}  "
            f"total_resid={m['total_resid']:.2f}  "
            f"share={100 * m['disagree_resid'] / m['total_resid']:.1f}%  "
            f"(diag_gap={m['diag_gap']:.1e}, repro_gap={r['repro']:.1e})"
        )
        print("  top cells by |sum_diff| (r,vbin,pibin | A vs B3 | n, sum_diff):")
        for c in top_cells(m):
            print(
                f"    r={c['r']} v{c['v_bin']} pi{c['pi_bin']} | "
                f"A={ACT[c['a_A']]:6} B3={ACT[c['a_B3']]:6} | "
                f"n={c['n']:8d} sum_diff={c['sum_diff']:+.3f}"
            )


if __name__ == "__main__":
    main()
