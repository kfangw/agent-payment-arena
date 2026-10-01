"""fitted rail diagnostic cell.

Runs the terminal-accounting machinery of
arena.experiments.settlement.recovery on the E-outage cell with its two
halt constants replaced by the values fitted from 360 days of Base batch
postings on Ethereum mainnet.
Everything else in the cell is left at its declared value, and the nine
main-run cells are untouched: this is an added diagnostic.

The only change to the machinery is that recovery.with_halt_minutes leaves
the halt start rate p01 alone; here both rates are set. The archive's own
modules are not modified.

    PYTHONPATH=. python3 fitted_rail.py --flows F1 F2 F3 --out fitted_out
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from arena.experiments.settlement import recovery as R
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import survival

# Fitted from gaps > 5 min in 359.99 days of batch postings to the Base
# batch inbox on L1: 18 halts, mean 22.8 min, 0.0792% stationary share.
FIT_P01 = 18 / 517975.2
FIT_P10 = 1 / 22.8
FIT_NOTE = ('p01, p10 fitted from L1 batch-posting gaps > 5 min over '
            '359.99 days (18 halts, mean 22.8 min); all other constants declared')


def _patched(env, halt_minutes=None, keep_halt_share=False):
    """Set both halt rates to the fitted values, ignoring the minute argument."""
    return replace(env, p01=FIT_P01, p10=FIT_P10)


def sigma_table():
    _, base, _ = envs_for('mid')['E-outage']
    fit = replace(base, p01=FIT_P01, p10=FIT_P10)
    out = {}
    for name, env in (('declared', base), ('fitted', fit)):
        out[name] = {
            'p01': env.p01, 'p10': env.p10,
            'mean_halt_minutes': 1 / env.p10,
            'stationary_halt_share': env.stationary_outage,
            'sigma_halt_arrival': {str(r): float(survival(env, recovery=r)[0, env.H, 1])
                                   for r in (0.0, 1.0)},
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flows', nargs='+', default=['F1', 'F2', 'F3'],
                   choices=['F1', 'F2', 'F3'])
    p.add_argument('--seed', type=int, default=380901)
    p.add_argument('--n-tune', type=int, default=2000)
    p.add_argument('--n-eval', type=int, default=5000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--block-size', type=int, default=50)
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--n-v', type=int, default=41)
    p.add_argument('--grid-points', type=int, default=21)
    p.add_argument('--conditions', nargs='+', default=['outage'],
                   choices=['normal', 'outage'])
    p.add_argument('--recovery', nargs='+', type=float, default=[0.0, 1.0])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=False)
    sig = sigma_table()
    (a.out / 'fitted-constants.json').write_text(json.dumps(
        {'note': FIT_NOTE, 'p01': FIT_P01, 'p10': FIT_P10, 'sigma': sig}, indent=1))
    print(json.dumps(sig, indent=1), flush=True)

    original = R.with_halt_minutes
    R.with_halt_minutes = _patched
    try:
        for flow in a.flows:
            args = SimpleNamespace(
                flow=flow, seed=a.seed, n_tune=a.n_tune, n_eval=a.n_eval,
                repeats=a.repeats, block_size=a.block_size, n_boot=a.n_boot,
                n_v=a.n_v, grid_points=a.grid_points, max_watch=None,
                conditions=list(a.conditions), recovery=list(a.recovery),
                halt_minutes=None, keep_halt_share=False,
                out=a.out / flow)
            print(f'=== {flow} ===', flush=True)
            R.run(args)
    finally:
        R.with_halt_minutes = original


if __name__ == '__main__':
    main()
