"""Summarise the per-draw npz files written by run_queue.py.

    python3 scripts/analyze.py            (works on whatever is done; from the package root)

Per (flow, exp, mode, eta):  for each draw r
    D_r = J(Comp_r) - J(B4)      L_r = J(nominal) - J(Comp_r)
in basis points of the mean exposure $49.50: sum over blocks / sum(block_counts)
/ 49.5 * 1e4 (same bp convention in every analysis script).  B4 = b4ref_E-fitted_F*_mid_s*.json.
Per-draw paired 95% block bootstrap of D_r (percentile, ratio of sums): 2,000 resamples
per flow, the SAME resamples for every draw of that flow, seed 20260929.
C: interaction L_joint - L_q - L_haz per matching (eta, r).
Writes summary.json and summary.md into results/draw_injection/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paths import DRAW_INJECTION, REFERENCE_B4

OUT = DRAW_INJECTION
NPZ = OUT / "npz"
B4 = REFERENCE_B4
SEEDS = {"F1": 7, "F2": 8, "F3": 9}
DEN, NB, THR = 49.5, 2_000, -10.0
rng = np.random.default_rng(20260929)


def bp(x, csum):
    return float(x) / csum / DEN * 1e4


def boot_lower(series, c, nb=NB, chunk=20):
    """series: (k, n) block-level differences. Returns per-series 2.5% bound in bp."""
    n = len(c)
    M = np.column_stack([series.T, c])                     # (n, k+1)
    stats = np.empty((nb, series.shape[0]))
    off = None
    for j0 in range(0, nb, chunk):
        m = min(chunk, nb - j0)
        idx = rng.integers(0, n, (m, n))
        if off is None or len(off) != m:
            off = (np.arange(m) * n)[:, None]
        W = np.bincount((idx + off).ravel(), minlength=m * n).reshape(m, n).astype(float)
        P = W @ M                                          # (m, k+1)
        stats[j0:j0 + m] = P[:, :-1] / P[:, -1:]
    lo = np.percentile(stats, 2.5, axis=0)
    return lo / DEN * 1e4


def summarise(vals):
    v = np.asarray(vals, float)
    return dict(count=int(len(v)), min=float(v.min()), median=float(np.median(v)),
                max=float(v.max()))


def main():
    files = sorted(NPZ.glob("*/*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    by_flow = defaultdict(list)
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        by_flow[meta["flow"]].append((meta, z["block_sums"]))
    summary = {"n_files": len(files), "bootstrap": dict(resamples=NB, seed=20260929,
               note="paired within flow"), "groups": []}
    for flow, items in sorted(by_flow.items()):
        b = json.loads((B4 / f"b4ref_E-fitted_{flow}_mid_s{SEEDS[flow]}.json").read_text())["payload"]
        c = np.asarray(b["block_counts"], float)
        b4 = np.asarray(b["b4_block_sums"], float)
        csum = c.sum()
        nom = [bs for m, bs in items if m["exp"] == "nominal"]
        nom = nom[0] if nom else None
        comps = [(m, bs) for m, bs in items if m["exp"] != "nominal"]
        rows = [(m, bs) for m, bs in items]
        for m, bs in rows:
            assert len(bs) == len(c), (flow, m["exp"], len(bs), len(c))
        lo = boot_lower(np.vstack([bs - b4 for _, bs in rows]), c) if rows else []
        per = []
        for (m, bs), l in zip(rows, lo):
            D = bp((bs - b4).sum(), csum)
            L = bp((nom - bs).sum(), csum) if nom is not None else None
            per.append(dict(exp=m["exp"], mode=m["mode"], eta=m["eta"], r=m["r"],
                            D=D, L=L, D_lo=float(l)))
        groups = defaultdict(list)
        for p in per:
            key = (p["exp"], p["mode"], p["eta"] if not p["mode"].startswith("oat") else None)
            groups[key].append(p)
        # C interaction
        Lmap = {(p["exp"], p["mode"], p["eta"], p["r"]): p["L"] for p in per}
        for (exp, mode, eta), ps in sorted(groups.items(), key=lambda kv: str(kv[0])):
            Ds = [p["D"] for p in ps]
            g = dict(flow=flow, exp=exp, mode=mode, eta=eta, D=summarise(Ds),
                     n_D_below_thr=int(sum(d < THR for d in Ds)),
                     n_Dlo_below_thr=int(sum(p["D_lo"] < THR for p in ps)),
                     draws=sorted(ps, key=lambda p: p["r"]))
            if nom is not None:
                g["L"] = summarise([p["L"] for p in ps])
            if exp == "C" and mode == "q" and nom is not None:
                inter = []
                for p in ps:
                    k = (eta, p["r"])
                    try:
                        inter.append(dict(r=p["r"], interaction=Lmap[("J", "joint") + k]
                                          - Lmap[("C", "q") + k] - Lmap[("C", "haz") + k]))
                    except KeyError:
                        pass
                if inter:
                    g["interaction"] = dict(summarise([x["interaction"] for x in inter]),
                                            per_draw=inter)
            summary["groups"].append(g)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    md = ["# Per-draw input-error summary (generated by analyze.py)", "",
          f"files = {summary['n_files']}, bootstrap {NB} resamples (paired within flow, seed 20260929), "
          f"threshold {THR} bp, unit = bp of $49.50", "",
          "| flow | exp | mode | eta | n | D min | D med | D max | L min | L med | L max | #D<-10 | #lo<-10 | interaction med [min,max] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in summary["groups"]:
        L = g.get("L")
        it = g.get("interaction")
        md.append("| " + " | ".join([
            g["flow"], g["exp"], g["mode"], "-" if g["eta"] is None else f"{g['eta']:g}",
            str(g["D"]["count"]),
            *(f"{g['D'][k]:+.2f}" for k in ("min", "median", "max")),
            *((f"{L[k]:+.2f}" for k in ("min", "median", "max")) if L else ("", "", "")),
            str(g["n_D_below_thr"]), str(g["n_Dlo_below_thr"]),
            f"{it['median']:+.2f} [{it['min']:+.2f},{it['max']:+.2f}] (n={it['count']})" if it else ""]) + " |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
