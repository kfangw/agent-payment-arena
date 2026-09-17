#!/usr/bin/env python3
"""Regenerate the rows of the refined-grid A - B4 table (Table S14) from the
archived block sums in results/blocks/: paired difference per payment, the
nine-cell simultaneous 95% block-bootstrap interval (Bonferroni, 20,000
resamples, the cell's seed), basis points of the declared mean exposure,
and the interval's place relative to the declared margin (inside, above, below,
straddles), printed in place of a verdict.

Run from code/:  PYTHONPATH=. python3 table_b4_rows.py ../results/blocks
"""
import json, sys, glob, os
import numpy as np
from arena.experiments.statistics import bootstrap_interval
from table_rows import margin_place

DECLARED = 49.50
EPS = 0.0495
CELLS = [('E-fast','F1',1),('E-fast','F2',2),('E-fast','F3',3),('E-slow','F1',4),('E-slow','F2',5),
         ('E-slow','F3',6),('E-outage','F1',7),('E-outage','F2',8),('E-outage','F3',9)]

def main(d):
    for env, fl, s in CELLS:
        z = np.load(os.path.join(d, f'blocks_{env}_{fl}_mid_s{s}.npz'))
        meta = json.loads(str(z['meta']))
        diff = z['A2'] - z['B4_refined']; counts = z['block_counts']
        mean = float(diff.sum() / counts.sum())
        lo, hi = bootstrap_interval(diff, counts, 20000, s, 1 - 0.05 / 9)
        print(f"{env}$\\times${fl} & ${mean:+.4f}$ & $[{lo:.4f},\\,{hi:.4f}]$ & ${mean/DECLARED*1e4:.1f}$ & {meta['b4_gridN']['k']} & {margin_place(lo, hi, EPS)}  % lower endpoint {lo:.6f}")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../results/blocks')
