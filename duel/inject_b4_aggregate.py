"""Aggregate S3: retention of A2 - B4 along each injection axis.

Reads the b4_{cell}_{axis}_s{seed}.json files from results_inject for the two
main cells and reports, per axis, the advantage curve in absolute dollars
with its interval and the retention ratio (advantage at a level over the
no-injection advantage).  The no-injection level must reproduce the base
A2 - B4; retention is reported next to the absolute edge, never alone,
because the pre-injection edge is small.  Axis robustness is ranked by the
retention at the far end of each branch.

Run:  python -m duel.inject_b4_aggregate --results results_inject
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inject import IDENTITY

CELL_SEED = {"E-slow x F2": 5, "E-outage x F2": 8}
AXES = ["delta", "kappa", "lambda", "noise"]
# base A2 - B4 (n=21) from Table 9, for the identity-level check
TABLE9 = {"E-slow x F2": 0.0624, "E-outage x F2": 0.0713}


def _load(cell, axis, results):
    env, flow = (s.strip() for s in cell.split("x"))
    seed = CELL_SEED[cell]
    p = Path(results) / f"b4_{env}_{flow}_{axis}_s{seed}.json"
    d = json.load(open(p))
    return d, str(p)


def rows_for(cell, axis, results):
    d, path = _load(cell, axis, results)
    pay = d["payload"]
    adv = {a["level"]: a for a in pay["advantage"]}
    g0 = pay["no_injection"]["mean"]
    ident = IDENTITY[axis]
    return dict(
        cell=cell, axis=axis, path=path, size=Path(path).stat().st_size,
        params_hash=d["params_hash"], code=d["code"],
        levels=pay["levels"], advantage=adv, g0=g0, ident=ident,
        eps=pay["eps"], mean_exposure=pay["mean_exposure"],
        replicates=pay["replicates"], b4_level0=pay["b4_level0"],
        identity_mean=adv[ident]["mean"], table9=TABLE9[cell],
    )


def build(results="results_inject"):
    out = {}
    for cell in CELL_SEED:
        out[cell] = {ax: rows_for(cell, ax, results) for ax in AXES}
    return out


def _fmt_row(a, eps):
    bp = a["mean"] / eps * 10.0 if eps else 0.0     # eps is 10 bp in dollars
    ret = a["retention"]
    rets = f"{ret*100:6.1f}%" if ret is not None else "   n/a"
    return (f"  lvl {a['level']:+6.2f}  A2-B4 {a['mean']:+.5f} "
            f"[{a['ci95'][0]:+.5f},{a['ci95'][1]:+.5f}]  {bp:+6.2f}bp  ret {rets}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_inject")
    args = ap.parse_args(argv)
    data = build(args.results)
    for cell, axes in data.items():
        any_r = next(iter(axes.values()))
        print(f"\n===== {cell}  (base A2-B4 Table9={any_r['table9']:.4f}, "
              f"eps={any_r['eps']:.5f}) =====")
        # identity check
        for ax, r in axes.items():
            gap = abs(r["identity_mean"] - r["table9"])
            flag = "OK" if gap < 5e-4 else f"GAP {gap:.4f}"
            print(f"[{ax}] identity A2-B4 {r['identity_mean']:.5f} vs Table9 "
                  f"{r['table9']:.4f} -> {flag}; B4_0 k={r['b4_level0']['k']}")
        for ax, r in axes.items():
            print(f"-- axis {ax} (replicates={r['replicates']}) --")
            for lv in r["levels"]:
                print(_fmt_row(r["advantage"][lv], r["eps"]))
        # rank axes by retention at the far positive end
        far = {}
        for ax, r in axes.items():
            hi = max(r["levels"])
            far[ax] = r["advantage"][hi]["retention"]
        order = sorted(far, key=lambda a: (far[a] if far[a] is not None else 9))
        print(f"axis robustness (retention at far end, ascending = drops most "
              f"first): {[(a, round(far[a], 3) if far[a] is not None else None) for a in order]}")


if __name__ == "__main__":
    main()
