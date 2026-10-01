#!/usr/bin/env python3
"""Write results/generated/table_wrong_rows.tex and results/generated/wrong_cells.json.

Inputs (runs dir): wrong_E-fitted_F*_s*.json (block sums of the compiled
policies replayed in the E-fitted world), b4ref_E-fitted_F*_mid_s*.json
(in results/reference_b4; refined-grid B4 tuned in that world), settlement_E-fitted_F*_mid_s*.json
(identity check: the 'true' policy must reproduce the archived A block sums).
Per-flow 95% paired block bootstrap, 10,000 resamples, ratio of sums, in basis
points of the declared mean exposure $49.50.

Usage: python3 scripts/make_wrong_rows.py [<runs_dir> [<wrong_dir>]]
(defaults: results/fitted_runs, results/wrong_constants)
"""
import json
import sys
from pathlib import Path

import numpy as np

from paths import REFERENCE_B4, RESULTS, WRONG_CONSTANTS

GEN = RESULTS / "generated"
GEN.mkdir(parents=True, exist_ok=True)
runs = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "fitted_runs"
wrong = Path(sys.argv[2]) if len(sys.argv) > 2 else WRONG_CONSTANTS
rng = np.random.default_rng(20260929)
B, DEN, MARGIN = 10_000, 49.5, 10.0


def ci(d, c):
    n = len(d)
    bs = np.empty(B)
    for j in range(B):
        i = rng.integers(0, n, n)
        bs[j] = d[i].sum() / c[i].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(float(x) / DEN * 1e4, 1) for x in (d.sum() / c.sum(), lo, hi)]


rows = {"true": "Right constants", "declared": "Declared halt law", "eta0.1": "10\\% error", "eta0.2": "20\\% error"}
out = {}
for f, s in [("F1", 7), ("F2", 8), ("F3", 9)]:
    w = json.loads((wrong / f"wrong_E-fitted_{f}_s{s}.json").read_text())
    b4 = np.asarray(json.loads((REFERENCE_B4 / f"b4ref_E-fitted_{f}_mid_s{s}.json").read_text())["payload"]["b4_block_sums"])
    st = json.loads((runs / f"settlement_E-fitted_{f}_mid_s{s}.json").read_text())["payload"]
    c = np.asarray(st["block_counts"], float)
    gap = float(np.abs(np.asarray(w["true"]) - np.asarray(st["policies"]["A"]["block_sums"])).max())
    nb = len(w["eta0.1"])
    res = {"identity_gap": gap}
    for k in rows:
        a = np.asarray(w[k])
        m = len(a)
        res[k] = ci(a - b4[:m], c[:m])
        res[k + "_blocks"] = m
    out[f] = res
    print(f, res)
(GEN / "wrong_cells.json").write_text(json.dumps(out, indent=1))
lines = []
for k, label in rows.items():
    cells = []
    for f in ("F1", "F2", "F3"):
        pt, lo, hi = out[f][k]
        cells.append(f"${pt:+.1f}$ $[{lo:+.1f},{hi:+.1f}]$")
    lines.append(label + " & " + " & ".join(cells) + " \\\\")
(GEN / "table_wrong_rows.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
