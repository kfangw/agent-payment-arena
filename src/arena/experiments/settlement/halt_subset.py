"""Halted-arrival subset statistics and an additive-loss rescoring, read from
the saved outcomes of the terminal-accounting runner.  No new draws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAYOFF, RELEASED, MISUSE_RELEASED, MISUSE_EXPOSURE, UNPAID_EXPOSURE = 0, 1, 2, 3, 4


def load(run: Path, condition: str, recovery: str, repeats: int):
    outcomes, halted, theta, exposure = {}, [], [], []
    for repeat in range(repeats):
        o = np.load(run / f'{repeat}-{condition}-recovery-{recovery}-outcomes.npz')
        e = np.load(run / f'{repeat}-{condition}-eval.npz')
        halted.append(e['paths'][:, 0] == 1)
        theta.append(e['theta']); exposure.append(e['v'])
        for name in o.files:
            outcomes.setdefault(name, []).append(o[name])
    outcomes = {k: np.concatenate(v) for k, v in outcomes.items()}
    return outcomes, np.concatenate(halted), np.concatenate(theta), np.concatenate(exposure)


def block_ci(values, block, n_boot, rng):
    blocks = values.reshape(-1, block).mean(1)
    idx = rng.integers(0, len(blocks), (n_boot, len(blocks)))
    return np.quantile(blocks[idx].mean(1), [.025, .975]).tolist()


def report(run: Path, condition: str, recovery: str, repeats: int, block: int,
           n_boot: int, h: float, seed: int):
    rng = np.random.default_rng(seed)
    out, halted, theta, v = load(run, condition, recovery, repeats)
    a = out['A']; legit = theta == 0
    row = {'run': str(run), 'condition': condition, 'recovery': float(recovery),
           'payments': int(len(a)), 'halted_arrivals': int(halted.sum())}
    if halted.any():
        ah = a[halted]; lh = legit[halted]
        row['A_release_rate_halted'] = float(ah[:, RELEASED].mean())
        row['A_release_rate_halted_legitimate'] = float(ah[lh, RELEASED].mean()) if lh.any() else None
        row['A_release_rate_halted_misuse'] = float(ah[~lh, RELEASED].mean()) if (~lh).any() else None
    for name in out:
        if name == 'A':
            continue
        d = a[:, PAYOFF] - out[name][:, PAYOFF]
        entry = {'pooled_mean': float(d.mean()),
                 'pooled_block_ci95': block_ci(d, block, n_boot, rng)}
        if halted.any():
            dh = d[halted]
            entry['per_halted_mean'] = float(dh.mean())
            entry['per_halted_se'] = float(dh.std(ddof=1) / np.sqrt(len(dh))) if len(dh) > 1 else None
        row[f'A_minus_{name}'] = entry
    # Additive loss convention: a release that is both misuse and unpaid loses h*v on top.
    extra = {name: h * m[:, UNPAID_EXPOSURE] * (m[:, MISUSE_RELEASED] > 0) for name, m in out.items()}
    row['additive_rescoring'] = {
        name: {'extra_loss_mean': float(x.mean()), 'affected_payments': int((x > 0).sum())}
        for name, x in extra.items()}
    for name in out:
        if name != 'A':
            shift = (a[:, PAYOFF] - extra['A']) - (out[name][:, PAYOFF] - extra[name])
            row['additive_rescoring'][f'A_minus_{name}_shift'] = float(shift.mean() - (a[:, PAYOFF] - out[name][:, PAYOFF]).mean())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--conditions', nargs='+', default=['outage'])
    parser.add_argument('--recovery', nargs='+', default=['0', '1'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--block-size', type=int, default=50)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--h', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    rows = [report(run, c, r, args.repeats, args.block_size, args.n_boot, args.h, args.seed)
            for run in args.run for c in args.conditions for r in args.recovery]
    for row in rows:
        b5 = row.get('A_minus_B5', {})
        print(f"{Path(row['run']).name:>16} {row['condition']:>7} r={row['recovery']:.0f} "
              f"halted {row['halted_arrivals']:3d} release {row.get('A_release_rate_halted', float('nan')):.3f} "
              f"A-B5/halted {b5.get('per_halted_mean', float('nan')):+7.2f} "
              f"pooled {b5.get('pooled_mean', float('nan')):+.4f} {np.round(b5.get('pooled_block_ci95', [np.nan, np.nan]), 3)} "
              f"additive shift {row['additive_rescoring'].get('A_minus_B5_shift', float('nan')):+.5f}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
