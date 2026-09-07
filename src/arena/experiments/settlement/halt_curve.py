"""Analytic settlement probability at arrival during a halt, as a function
of the mean halt duration.  No sampling: the value is read from the exact
survival recursion of the E-outage cell."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .gate import envs_for
from .outage import survival
from .recovery import with_halt_minutes


def curve(minutes, recovery=0.0, keep_halt_share=False):
    _, base, _ = envs_for('mid')['E-outage']
    rows = []
    for m in minutes:
        env = with_halt_minutes(base, m, keep_halt_share)
        sig = survival(env, recovery=recovery)
        rows.append({'halt_minutes': m, 'p01': env.p01, 'p10': env.p10,
                     'stationary_halt_share': env.stationary_outage,
                     'sigma_halt_arrival': float(sig[0, env.H, 1]),
                     'sigma_normal_arrival': float(sig[0, env.H, 0])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', nargs='+', type=float,
                        default=[5, 10, 15, 20, 30, 45, 60, 90, 120])
    parser.add_argument('--recovery', type=float, default=0.0)
    parser.add_argument('--keep-halt-share', action='store_true')
    parser.add_argument('--out', help='optional JSON path')
    args = parser.parse_args()
    rows = curve(args.minutes, args.recovery, args.keep_halt_share)
    print(f"{'halt_min':>9} {'share':>8} {'sigma_halt':>11} {'sigma_normal':>13}")
    for r in rows:
        print(f"{r['halt_minutes']:9.1f} {r['stationary_halt_share']:8.5f} "
              f"{r['sigma_halt_arrival']:11.4f} {r['sigma_normal_arrival']:13.4f}")
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'recovery': args.recovery, 'keep_halt_share': args.keep_halt_share,
                       'rows': rows}, fh, indent=1)


if __name__ == '__main__':
    main()
