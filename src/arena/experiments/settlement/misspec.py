"""Rescore saved terminal-accounting runs with policies fitted under another accounting.

The recovery runner fits every policy and scores it under the same accounting.
This module reads a finished run, fits nothing new, and replays the saved
evaluation paths under a different recovery probability while holding each
policy at its saved fit: A is recompiled from the saved response-rate estimate
under the policy's own accounting, B3 and B4 keep their saved thresholds and
watch length, and B5 keeps its frozen halt rule over the saved B3 thresholds.
The gap between this score and the run's own matched score is the cost of
scoring a policy in a world whose accounting differs from the one it assumed.

Two conventions for the verification exercise rule are reported. The rule is
part of the server's commitment, so by default it stays at the policy's own
accounting; the alternative recomputes it under the world's accounting.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .core import rho_hat_from_q
from .gate import envs_for
from .outage import OutageDraws, compile_outage, survival, window_AD
from .recovery import FIELDS, paired_bootstrap, replay, rule, summarize
from .report import jsonable, write_once

DRAW_KEYS = ('v', 'p_true', 'theta', 'pi0', 'u_stage', 't_ans', 'paths')


def load_eval(run: Path, repeat: int, condition: str):
    z = np.load(run / f'{repeat}-{condition}-eval.npz')
    return OutageDraws(**{k: z[k] for k in DRAW_KEYS}), z['recovery_coins']


def load_fit(run: Path, repeat: int, condition: str, recovery: float) -> dict:
    return json.loads((run / f'{repeat}-{condition}-recovery-{recovery:g}.json').read_text())


def policies_at(env, fit: dict, believed: float, n_v: int):
    """Rebuild the three policies exactly as the run fitted them at `believed`."""
    a = compile_outage(replace(env, rho=rho_hat_from_q(fit['q_hat'], env.tau)),
                       'A', n_v=n_v, recovery=believed)
    return {'A': a,
            'B4': rule(env, *fit['b4']),
            'B5': rule(env, 0, *fit['b3_for_b5'], halt_reject=True)}


def rescore(run: Path, repeat: int, condition: str, believed: float, actual: float,
            exercise_at: str, n_v: int, base):
    env = base if condition == 'outage' else replace(base, p01=0.)
    draws, coins = load_eval(run, repeat, condition)
    fit = load_fit(run, repeat, condition, believed)
    ex_recovery = believed if exercise_at == 'policy' else actual
    _, _, exercise = window_AD(env, survival(env, recovery=ex_recovery))
    outcomes = {name: replay(env, draws, policy, exercise, actual, coins)
                for name, policy in policies_at(env, fit, believed, n_v).items()}
    return draws, outcomes


def identity_check(run: Path, repeat: int, condition: str, recovery: float, n_v: int, base):
    """Replaying a run at its own accounting must reproduce its saved outcomes."""
    _, outcomes = rescore(run, repeat, condition, recovery, recovery, 'policy', n_v, base)
    saved = np.load(run / f'{repeat}-{condition}-recovery-{recovery:g}-outcomes.npz')
    return {name: float(np.max(np.abs(outcomes[name] - saved[name]))) for name in outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True,
                        help='finished recovery run directory (read only)')
    parser.add_argument('--believed', type=float, default=0.0,
                        help='recovery the policies were fitted under')
    parser.add_argument('--actual', type=float, default=1.0,
                        help='recovery the world applies when scoring')
    parser.add_argument('--conditions', nargs='+', choices=['normal', 'outage'],
                        default=['outage'])
    parser.add_argument('--exercise-at', choices=['policy', 'world', 'both'], default='both')
    parser.add_argument('--n-v', type=int, default=41)
    parser.add_argument('--block-size', type=int, default=50)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--skip-identity', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.run / 'config.json').read_text())
    repeats, seed = config['repeats'], config['seed']
    _, base, _ = envs_for('mid')['E-outage']
    args.out.mkdir(parents=True, exist_ok=False)

    identity = {}
    if not args.skip_identity:
        for condition in args.conditions:
            for recovery in (args.believed, args.actual):
                gaps = identity_check(args.run, 0, condition, recovery, args.n_v, base)
                identity[f'0-{condition}-recovery-{recovery:g}'] = gaps
                print('identity', condition, recovery, gaps, flush=True)
                if max(gaps.values()) > 1e-9:
                    raise SystemExit('replay does not reproduce the saved run; stopping')

    conventions = ['policy', 'world'] if args.exercise_at == 'both' else [args.exercise_at]
    rows = []
    for condition in args.conditions:
        for repeat in range(repeats):
            matched = load_fit(args.run, repeat, condition, args.actual)['metrics']
            for convention in conventions:
                draws, outcomes = rescore(args.run, repeat, condition, args.believed,
                                          args.actual, convention, args.n_v, base)
                metrics = {name: summarize(value, draws) for name, value in outcomes.items()}
                row = {'repeat': repeat, 'condition': condition, 'believed': args.believed,
                       'actual': args.actual, 'exercise_at': convention, 'metrics': metrics,
                       'misspecification_cost': {
                           name: matched[name]['payoff'] - metrics[name]['payoff']
                           for name in metrics},
                       'comparisons': {}}
                for name in ('B4', 'B5'):
                    delta = outcomes['A'][:, 0] - outcomes[name][:, 0]
                    row['comparisons'][name] = {
                        'mean': float(delta.mean()),
                        'conditional_ci95': paired_bootstrap(
                            delta, args.block_size,
                            np.random.default_rng([seed, repeat, 998]), args.n_boot)}
                tag = f'{repeat}-{condition}-believed-{args.believed:g}-actual-{args.actual:g}-{convention}'
                np.savez_compressed(args.out / f'{tag}-outcomes.npz', **outcomes)
                write_once(args.out / f'{tag}.json', jsonable(row))
                rows.append(row)
                print(tag, {k: round(v, 4) for k, v in row['misspecification_cost'].items()},
                      {k: round(v['mean'], 4) for k, v in row['comparisons'].items()}, flush=True)

    aggregate = []
    for condition in args.conditions:
        for convention in conventions:
            sel = [r for r in rows if r['condition'] == condition and r['exercise_at'] == convention]
            entry = {'condition': condition, 'exercise_at': convention,
                     'misspecification_cost': {}, 'a_minus': {}}
            for name in ('A', 'B4', 'B5'):
                vals = np.array([r['misspecification_cost'][name] for r in sel])
                rng = np.random.default_rng([seed, 776])
                entry['misspecification_cost'][name] = {
                    'repeat_values': vals.tolist(), 'mean': float(vals.mean()),
                    'repeat_bootstrap_ci95': None if len(vals) < 5 else np.quantile([
                        rng.choice(vals, len(vals)).mean()
                        for _ in range(args.n_boot)], [.025, .975]).tolist()}
            for name in ('B4', 'B5'):
                vals = np.array([r['comparisons'][name]['mean'] for r in sel])
                entry['a_minus'][name] = {'repeat_values': vals.tolist(), 'mean': float(vals.mean())}
            aggregate.append(entry)

    write_once(args.out / 'summary.json', jsonable({
        'source_run': str(args.run), 'believed': args.believed, 'actual': args.actual,
        'identity_check': identity, 'outcome_columns': FIELDS, 'rows': rows,
        'aggregate': aggregate,
        'inference': 'Rescoring of saved paths; no new draws, no new fitting. '
                     'Costs are per input payment in dollars. Intervals are exploratory.'}))


if __name__ == '__main__':
    main()
