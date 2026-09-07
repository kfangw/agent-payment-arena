"""Terminal-accounting sensitivity with frozen A/B4/B5 definitions.

Stage 0 means authorization not yet executed; stage >=1 means executed.
The release cutoff closes uncommitted requests. Executed transfers backing
an earlier grant may complete later. No additional post-release delay cost
is charged. This is a declared alternative model, not a chain calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, rho_hat_from_q
from .flows import make_flows
from .gate import envs_for
from .grid import _best_ab
from .outage import compile_outage, draw_outage_batch, survival, window_AD
from .policies import suspicion_grid
from .report import envelope, jsonable, write_once

FIELDS = ('payoff', 'released', 'misuse_released', 'misuse_exposure',
          'unpaid_exposure', 'legitimate_refused', 'queried', 'release_seconds',
          'late_settled', 'unexecuted_expired')


def settlement(env, draws, k, stage, tick, recovery, coin):
    """Return (paid, late, unexecuted-expired) using shared stage uniforms."""
    while stage <= env.N and tick < env.H:
        if draws.paths[k, tick] == 0:
            if draws.u_stage[k, stage] < env.f[stage]:
                return False, False, False
            stage += 1
        tick += 1
    if stage > env.N:
        return True, False, False
    if stage == 0:
        return False, False, True
    # Eventual recovery of the public regime is assumed. Reusing the remaining
    # stage uniforms integrates its duration out without inventing a tail cap.
    paid = coin < recovery and np.all(draws.u_stage[k, stage:] >= env.f[stage:])
    return bool(paid), bool(paid), False


def replay(env, draws, policy, exercise, recovery, coins):
    """Replay every payment and retain security outcomes, not only payoff."""
    out = np.zeros((len(draws), len(FIELDS)))
    for k in range(len(draws)):
        stage = tick = 0
        v, pi = float(draws.v[k]), float(draws.pi0[k])
        legitimate = draws.theta[k] == 0

        def release():
            paid, late, expired = settlement(env, draws, k, stage, tick, recovery, coins[k])
            out[k, 0] += v * ((env.m if legitimate else -env.h) if paid else -1)
            out[k, 1:5] = (1, not legitimate, v * (not legitimate), v * (not paid))
            out[k, 7:] = (tick * env.tick_seconds, late, expired)

        while tick < env.H or stage > env.N:
            action = policy(stage, max(env.H - tick, 0), int(draws.paths[k, tick]), v, pi)
            if action == GRANT:
                release()
                break
            if action == REJECT:
                break
            if action == VERIFY:
                out[k, 0] -= env.C
                out[k, 6] = 1
                for elapsed in range(1, min(env.tau, env.H - tick) + 1):
                    out[k, 0] -= env.cw * v
                    if draws.t_ans[k] == elapsed:
                        if legitimate and exercise[stage, env.H - tick, draws.paths[k, tick]] > 0:
                            release()
                        break
                    if draws.paths[k, tick] == 0 and stage <= env.N:
                        if draws.u_stage[k, stage] < env.f[stage]:
                            break
                        stage += 1
                    tick += 1
                break
            if action != WAIT or stage > env.N:
                raise ValueError('invalid action')
            out[k, 0] -= env.cw * v
            if draws.paths[k, tick] == 0:
                if draws.u_stage[k, stage] < env.f[stage]:
                    break
                stage += 1
            tick += 1
        out[k, 5] = legitimate and not out[k, 1]
    return out


def rule(env, watch, lower=0., upper=1., force=None, halt_reject=False):
    """B4 watches first; B5 rejects a halt at arrival then uses B3."""
    def choose(stage, remaining, regime, value, pi):
        if halt_reject and remaining == env.H and regime == 1:
            return REJECT
        if env.H - remaining < watch and stage <= env.N:
            return WAIT
        if force is not None:
            return force
        return GRANT if pi < lower else REJECT if pi > upper else VERIFY
    return choose


def tune_rules(env, draws, exercise, recovery, coins, grid_points, max_watch):
    """Retune B4 and B3 on each tuning split under each accounting model."""
    grid = suspicion_grid(grid_points)
    best = (-np.inf, None)
    b3 = None
    for watch in range(max_watch + 1):
        legs = [replay(env, draws, rule(env, watch, force=a), exercise, recovery, coins)[:, 0]
                for a in (GRANT, REJECT, VERIFY)]
        edges, score = _best_ab(*legs, draws.pi0, grid)
        if watch == 0:
            b3 = edges
        if score > best[0]:
            best = (score, (watch, *edges))
    return best[1], b3


def summarize(rows, draws):
    """Rates use all payments except the explicitly conditional outcomes."""
    totals = rows.sum(axis=0)
    result = dict(zip(FIELDS, rows.mean(axis=0).tolist(), strict=True))
    result['misuse_grant_rate_given_misuse'] = (
        float(totals[2] / np.sum(draws.theta == 1)) if np.any(draws.theta == 1) else None)
    result['refusal_rate_given_legitimate'] = (
        float(totals[5] / np.sum(draws.theta == 0)) if np.any(draws.theta == 0) else None)
    result['mean_release_seconds_given_released'] = (
        float(totals[7] / totals[1]) if totals[1] else None)
    return result


def paired_bootstrap(difference, block_size, rng, n_boot):
    """Resample whole independent simulation episodes, not payments."""
    sums = difference.reshape(-1, block_size).sum(axis=1)
    means = [rng.choice(sums, len(sums), replace=True).mean() / block_size
             for _ in range(n_boot)]
    return np.quantile(means, [.025, .975]).tolist()


def run(args):
    """Create one immutable experiment directory with draws and outcomes."""
    if min(args.n_tune, args.n_eval, args.repeats, args.block_size, args.n_boot) <= 0:
        raise ValueError('sample counts must be positive')
    if args.grid_points < 2 or args.n_v < 2:
        raise ValueError('grid sizes must be at least two')
    if args.n_tune % args.block_size or args.n_eval % args.block_size:
        raise ValueError('sample counts must be multiples of block size')
    if not args.recovery or any(not 0 <= x <= 1 for x in args.recovery):
        raise ValueError('recovery probabilities must be in [0,1]')
    if len(set(args.recovery)) != len(args.recovery):
        raise ValueError('duplicate recovery probabilities')
    _, base, _ = envs_for('mid')['E-outage']
    max_watch = base.N + 1 if args.max_watch is None else args.max_watch
    if not 0 <= max_watch <= base.H:
        raise ValueError('watch grid outside horizon')
    args.out.mkdir(parents=True, exist_ok=False)
    modules = sorted(Path(__file__).parent.glob('*.py'))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}
    config = vars(args) | {'out': str(args.out), 'max_watch': max_watch,
                           'python': platform.python_version(), 'numpy': np.__version__,
                           'source_sha256': manifest, 'base_environment': asdict(base)}
    write_once(args.out / 'config.json', jsonable(config))
    # Preserve the actual uncommitted experiment sources as well as their hashes.
    snapshot = args.out / 'source'; snapshot.mkdir()
    for p in modules:
        (snapshot / p.name).write_bytes(p.read_bytes())
    results = []
    for repeat in range(args.repeats):
        split_seeds = np.random.SeedSequence([args.seed, repeat]).spawn(4)
        coins_t = np.random.default_rng(split_seeds[2]).random(args.n_tune)
        coins_e = np.random.default_rng(split_seeds[3]).random(args.n_eval)
        for condition in args.conditions:
            env = base if condition == 'outage' else replace(base, p01=0.)
            # Same physical stage endpoint and 60-second resolution in both cases.
            draws = [draw_outage_batch(env, make_flows()[args.flow], n,
                     np.random.default_rng(seed), payments_per_episode=args.block_size)
                     for n, seed in zip((args.n_tune, args.n_eval), split_seeds[:2], strict=True)]
            for name, d, coins in zip(('tune', 'eval'), draws, (coins_t, coins_e), strict=True):
                np.savez_compressed(args.out / f'{repeat}-{condition}-{name}.npz',
                                    **asdict(d), recovery_coins=coins)
            tune, evaluation = draws
            q_hat = float(np.mean(tune.t_ans <= env.tau))
            for recovery in args.recovery:
                tag = f'{repeat}-{condition}-recovery-{recovery:g}'
                sig = survival(env, recovery=recovery)
                _, _, exercise = window_AD(env, sig)
                a = compile_outage(replace(env, rho=rho_hat_from_q(q_hat, env.tau)),
                                   'A', n_v=args.n_v, recovery=recovery)
                b4, b3 = tune_rules(env, tune, exercise, recovery, coins_t,
                                    args.grid_points, max_watch)
                policies = {'A': a, 'B4': rule(env, *b4),
                            'B5': rule(env, 0, *b3, halt_reject=True)}
                outcomes = {name: replay(env, evaluation, policy, exercise, recovery, coins_e)
                            for name, policy in policies.items()}
                np.savez_compressed(args.out / f'{tag}-outcomes.npz', **outcomes)
                row = {'repeat': repeat, 'condition': condition, 'recovery': recovery,
                       'q_hat': q_hat, 'b4': b4, 'b3_for_b5': b3,
                       'metrics': {name: summarize(value, evaluation)
                                   for name, value in outcomes.items()}, 'comparisons': {}}
                for name in ('B4', 'B5'):
                    delta = outcomes['A'][:, 0] - outcomes[name][:, 0]
                    row['comparisons'][name] = {
                        'mean': float(delta.mean()),
                        'conditional_ci95': paired_bootstrap(delta, args.block_size,
                            np.random.default_rng([args.seed, repeat, 999]), args.n_boot)}
                write_once(args.out / f'{tag}.json', jsonable(row))
                results.append(row)
                print(tag, row['comparisons'], flush=True)
    aggregate = []
    for condition in args.conditions:
        for recovery in args.recovery:
            selected = [r for r in results if r['condition'] == condition
                        and r['recovery'] == recovery]
            for comparator in ('B4', 'B5'):
                values = np.array([r['comparisons'][comparator]['mean'] for r in selected])
                rng = np.random.default_rng([args.seed, 777])
                ci = None if len(values) < 5 else np.quantile([
                    rng.choice(values, len(values), replace=True).mean()
                    for _ in range(args.n_boot)], [.025, .975]).tolist()
                aggregate.append({'condition': condition, 'recovery': recovery,
                    'comparator': comparator, 'repeat_differences': values.tolist(),
                    'mean': float(values.mean()), 'repeat_bootstrap_ci95': ci})
    payload = {'rows': results, 'aggregate': aggregate, 'outcome_columns': FIELDS,
               'inference': 'Exploratory unadjusted intervals. Conditional intervals hold tuning '
               'fixed; repeat bootstrap resamples independent full train/evaluate runs. '
               'No automatic equivalence verdict; no CI with fewer than five repeats.'}
    write_once(args.out / 'summary.json', envelope('terminal-recovery', args.flow,
        args.seed, args.n_eval, args.n_tune, jsonable(config), payload))
    files = sorted(p for p in args.out.rglob('*') if p.is_file())
    write_once(args.out / 'sha256.json', {str(p.relative_to(args.out)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow', choices=['F1', 'F2', 'F3'], default='F2')
    parser.add_argument('--seed', type=int, default=380901)
    parser.add_argument('--n-tune', type=int, default=2000)
    parser.add_argument('--n-eval', type=int, default=5000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--block-size', type=int, default=50)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--n-v', type=int, default=41)
    parser.add_argument('--grid-points', type=int, default=21)
    parser.add_argument('--max-watch', type=int)
    parser.add_argument('--conditions', nargs='+', choices=['normal', 'outage'],
                        default=['normal', 'outage'])
    parser.add_argument('--recovery', nargs='+', type=float, default=[0., .5, 1.])
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
