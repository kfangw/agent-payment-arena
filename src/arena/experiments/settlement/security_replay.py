"""Security outcomes of the main cells, replayed from declared seeds.

The main run stored per-block payoff sums only.  This runner regenerates a
cell's draws from its seed, rebuilds the policies from the archived fitted
parameters (no retuning), replays them with per-payment instrumentation, and
checks that the replayed mean payoffs equal the archived ones before
reporting release, misuse, unpaid, refusal, and query outcomes.

With --h the same draws are scored under a different misuse weight.  The
archived fits belong to h = 1, so they are not reused: A is re-solved and
B3, B4, B5 are re-tuned on the same tuning draws at the new weight, which
is what a comparison at another h has to do if the tuned rules are to move
with the objective too.  The payoff identity check does not apply there;
the tuning draws are still checked against the archived response rate, so
a failure to reproduce the draws is still caught.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .b5 import StateMatchedOutagePolicy
from .core import GRANT, REJECT, VERIFY, WAIT, rho_hat_from_q, sigma_list
from .flows import make_flows
from .gate import envs_for
from .outage import _settle_from, compile_outage, draw_outage_batch, survival
from .outage import replay_outage
from .outage import window_AD as outage_window_AD
from .policies import B3, compile_A, suspicion_grid
from .run import CHAIN_BLOCK, OB
from .simulate import draw_batch, replay
from .watch import (PAYMENTS_PER_EPISODE, FixedActionOutageWatchPolicy as OB4Force,
                    FixedActionWatchPolicy as B4Force, OutageWatchBandPolicy,
                    WatchBandPolicy, horizon_grid as k_grid,
                    tune_watch_policy as tune_b4)

FIELDS = ('payoff', 'released', 'misuse_released', 'misuse_exposure',
          'unpaid_exposure', 'legitimate_refused', 'queried')
DECLARED_GRID_N = 21   # the main run's B3/B4 threshold grid
REFINED_GRID_N = 161   # the diagnostic grid B5 inherits
SEEDS = {('E-fast', 'F1'): 1, ('E-fast', 'F2'): 2, ('E-fast', 'F3'): 3,
         ('E-slow', 'F1'): 4, ('E-slow', 'F2'): 5, ('E-slow', 'F3'): 6,
         ('E-outage', 'F1'): 7, ('E-outage', 'F2'): 8, ('E-outage', 'F3'): 9}


def _record(out, k, v, legitimate, released, paid, queried):
    out[k, 1] = released
    out[k, 2] = released and not legitimate
    out[k, 3] = v if (released and not legitimate) else 0.0
    out[k, 4] = v if (released and not paid) else 0.0
    out[k, 5] = legitimate and not released
    out[k, 6] = queried


def replay_chain(ch, d, policy, ex_expected):
    n = len(d)
    out = np.zeros((n, len(FIELDS)))
    N, FIN = ch.N, ch.N + 1
    for k in range(n):
        v, pi = float(d.v[k]), float(d.pi0[k])
        legitimate = d.theta[k] == 0
        payoff, stage, released, paid, queried = 0.0, 0, False, False, False

        def leg(stage):
            won = not (d.fail_at[k] >= stage and d.fail_at[k] <= N)
            return won, (v * (ch.m if legitimate else -ch.h) if won else -v)

        while True:
            a = policy(stage, v, pi)
            if a == WAIT and stage == FIN:
                a = REJECT
            if a == GRANT:
                paid, gain = leg(stage); payoff += gain; released = True
                break
            if a == REJECT:
                break
            if a == VERIFY:
                queried = True
                payoff -= ch.C
                j = stage
                for s in range(1, ch.tau + 1):
                    payoff -= ch.cw * v
                    if d.t_ans[k] == s:
                        if legitimate and ex_expected[min(j, FIN)] > 0.0:
                            paid, gain = leg(j); payoff += gain; released = True
                        break
                    if j <= N and d.fail_at[k] == j:
                        break
                    j = min(j + 1, FIN)
                break
            payoff -= ch.cw * v
            if d.fail_at[k] == stage:
                break
            stage += 1
            if stage > FIN:
                break
        out[k, 0] = payoff
        _record(out, k, v, legitimate, released, paid, queried)
    return out


def replay_outage_fields(env, d, policy, ex_pos):
    n = len(d)
    out = np.zeros((n, len(FIELDS)))
    N, H, FIN = env.N, env.H, env.N + 1
    for k in range(n):
        v, pi = float(d.v[k]), float(d.pi0[k])
        legitimate = d.theta[k] == 0
        payoff, released, paid, queried = 0.0, False, False, False
        i, t = 0, 0
        while True:
            if i > N:
                i = FIN
            l = H - t
            r = int(d.paths[k, t])
            if l <= 0 and i < FIN:
                break
            a = policy(i, max(l, 0), r, v, pi)
            if a == GRANT:
                won = 1 if i == FIN else _settle_from(env, d, k, i, t)
                payoff += v * ((env.m if legitimate else -env.h) if won else -1.0)
                released, paid = True, bool(won)
                break
            if a == REJECT:
                break
            if a == VERIFY:
                queried = True
                payoff -= env.C
                w = min(env.tau, max(l, 0))
                s = 0
                while s < w:
                    payoff -= env.cw * v
                    s += 1
                    if d.t_ans[k] == s:
                        li, lr = H - t, int(d.paths[k, t])
                        if legitimate and ex_pos[i, li, lr] > 0:
                            won = 1 if i == FIN else _settle_from(env, d, k, i, t)
                            payoff += v * (env.m if won else -1.0)
                            released, paid = True, bool(won)
                        break
                    if int(d.paths[k, t]) == 0 and i <= N:
                        if d.u_stage[k, i] < env.f[i]:
                            break
                        i += 1
                        if i > N:
                            i = FIN
                    t += 1
                    if H - t <= 0:
                        break
                break
            if a == WAIT:
                payoff -= env.cw * v
                if r == 0 and i <= N:
                    if d.u_stage[k, i] < env.f[i]:
                        break
                    i += 1
                t += 1
                continue
            break
        out[k, 0] = payoff
        _record(out, k, v, legitimate, released, paid, queried)
    return out


def archived(archive: Path, env_name: str, flow: str, seed: int):
    tag = f'{env_name}_{flow}_mid_s{seed}'
    duel = json.loads((archive / 'results' / f'duel_{tag}.json').read_text())
    b4 = json.loads((archive / 'results' / f'b4_{tag}.json').read_text())
    b5 = None
    if env_name == 'E-outage':
        b5 = json.loads((archive / 'b5' / f'b5_{tag}.json').read_text())
    return duel, b4, b5


def summarize(out, theta):
    legit = theta == 0
    n, n_legit, n_misuse = len(theta), int(legit.sum()), int((~legit).sum())
    return {'release_rate': float(out[:, 1].mean()),
            'misuse_grant_rate': float(out[:, 2].sum() / n_misuse) if n_misuse else None,
            'misuse_exposure': float(out[:, 3].mean()),
            'unpaid_exposure': float(out[:, 4].mean()),
            'legitimate_refusal_rate': float(out[:, 5].sum() / n_legit) if n_legit else None,
            'query_frequency': float(out[:, 6].mean()),
            'payoff': float(out[:, 0].mean()), 'n': n}


def block_ci(values, episodes, n_boot, rng):
    sums = np.bincount(episodes, weights=values)
    counts = np.bincount(episodes).astype(float)
    idx = rng.integers(0, len(sums), (n_boot, len(sums)))
    return np.quantile(sums[idx].sum(1) / counts[idx].sum(1), [.025, .975]).tolist()


def paired(out_a, out_b, theta, episodes, n_boot, rng):
    legit = theta == 0
    res = {}
    for name, col, denom in (('misuse_grant_rate', 2, ~legit), ('legitimate_refusal_rate', 5, legit),
                             ('misuse_exposure', 3, None), ('unpaid_exposure', 4, None),
                             ('query_frequency', 6, None)):
        d = out_a[:, col] - out_b[:, col]
        if denom is not None:
            d = d[denom]; ep = episodes[denom]
        else:
            ep = episodes
        res[name] = {'mean': float(d.mean()), 'block_ci95': block_ci(d, ep, n_boot, rng)}
    return res


def retune(force, pi0, horizons, declared, refined):
    """Re-fit the family-B rules on the tuning draws under the world the
    forced-payoff callable scores in.  Returns (B3 pair, B4 triple, B5 pair);
    B3 and B4 use the declared grid of the main run, B5 the refined one;
    pass refined=None on a rail that has no B5."""
    (_, b3a, b3b), _, _ = tune_b4(force, declared, [0], pi0)
    (k4, a4, b4), _, _ = tune_b4(force, declared, horizons, pi0)
    b5ab = None
    if refined is not None:
        (_, b5a, b5b), _, _ = tune_b4(force, refined, [0], pi0)
        b5ab = (b5a, b5b)
    return (b3a, b3b), (k4, a4, b4), b5ab


def run_cell(env_name, flow_name, archive, n_tune, n_eval, n_boot, seed_boot, check, h=None):
    seed = SEEDS[(env_name, flow_name)]
    kind, env, _ = envs_for('mid')[env_name]
    flow = make_flows()[flow_name]
    duel, b4j, b5j = archived(archive, env_name, flow_name, seed)
    n_tune = duel['n_tune'] if n_tune is None else n_tune
    n_eval = duel['n_eval'] if n_eval is None else n_eval
    rng = np.random.default_rng(seed)
    b3a, b3b = duel['payload']['calib']['b_params']['B3']
    k4, a4, b4 = b4j['payload']['b4']['k'], b4j['payload']['b4']['a'], b4j['payload']['b4']['b']
    b5ab = (b5j['b5']['a'], b5j['b5']['b']) if b5j is not None else None
    # The draws are laws, not payoffs: they do not move with h, so the same
    # seed gives the same paths and the comparison across h stays paired.
    world = env if h is None else replace(env, h=float(h))
    declared, refined = suspicion_grid(DECLARED_GRID_N), suspicion_grid(REFINED_GRID_N)
    if kind == 'chain':
        tune_d = draw_batch(env, flow, n_tune, rng)
        eval_d = draw_batch(env, flow, n_eval, rng)
        ex = np.maximum(sigma_list(env.f) * (1 + env.m) - 1.0, 0.0)
        q_hat = float((tune_d.t_ans <= env.tau).mean())
        rho_hat = rho_hat_from_q(q_hat, env.tau)
        if h is not None:
            (b3a, b3b), (k4, a4, b4), _ = retune(
                lambda k, act: replay(world, tune_d, B4Force(k, act), ex),
                tune_d.pi0, k_grid(env.N + 1), declared, None)
        policies = {'A': compile_A(world, 'A', rho=rho_hat), 'B3': B3(b3a, b3b),
                    'B4': WatchBandPolicy(k4, a4, b4)}
        outs = {name: replay_chain(world, eval_d, pol, ex) for name, pol in policies.items()}
        episodes = np.arange(n_eval) // CHAIN_BLOCK
    else:
        tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=PAYMENTS_PER_EPISODE)
        eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=PAYMENTS_PER_EPISODE)
        _, _, ex = outage_window_AD(env, survival(env))
        q_hat = float((tune_d.t_ans <= env.tau).mean())
        rho_hat = rho_hat_from_q(q_hat, env.tau)
        if h is not None:
            (b3a, b3b), (k4, a4, b4), b5ab = retune(
                lambda k, act: replay_outage(world, tune_d, OB4Force(k, act, env.H, env.N), ex),
                tune_d.pi0, k_grid(env.H), declared, refined)
        policies = {'A': compile_outage(replace(world, rho=rho_hat), 'A'), 'B3': OB(B3(b3a, b3b)),
                    'B4': OutageWatchBandPolicy(k4, a4, b4, env.H, env.N),
                    'B5': StateMatchedOutagePolicy(*b5ab)}
        outs = {name: replay_outage_fields(world, eval_d, pol, ex) for name, pol in policies.items()}
        episodes = np.arange(n_eval) // PAYMENTS_PER_EPISODE
    theta = eval_d.theta
    expected = {'A': duel['payload']['means']['A2'], 'B3': duel['payload']['means']['B3'],
                'B4': b4j['payload']['b4_mean']}
    if b5j is not None:
        expected['B5'] = b5j['b5_mean']
    identity = None if h is not None else {
        name: {'replayed': float(outs[name][:, 0].mean()), 'archived': expected[name],
               'gap': float(outs[name][:, 0].mean() - expected[name])} for name in outs}
    if check:
        # The response rate is a property of the draws, not of the payoff
        # weights, so this check holds at every h and proves the draws
        # reproduce the archived run.
        if abs(q_hat - duel['payload']['calib']['q_hat']) > 1e-12:
            raise RuntimeError(f'{env_name} {flow_name}: tuning draws differ from the archived run')
        if identity is not None:
            bad = {k: v for k, v in identity.items() if abs(v['gap']) > 1e-9}
            if bad:
                raise RuntimeError(
                    f'{env_name} {flow_name}: replayed payoffs differ from the archive: {bad}')
    rng_b = np.random.default_rng([seed_boot, seed])
    row = {'environment': env_name, 'flow': flow_name, 'seed': seed, 'n_tune': n_tune, 'n_eval': n_eval,
           'h': float(world.h), 'refitted': h is not None,
           'q_hat': q_hat, 'rho_hat': rho_hat, 'mean_exposure': float(eval_d.v.mean()),
           'policies': {'B3': [b3a, b3b], 'B4': [k4, a4, b4]},
           'identity_check': identity,
           'outcomes': {name: summarize(o, theta) for name, o in outs.items()},
           'paired': {f'A_minus_{name}': paired(outs['A'], outs[name], theta, episodes, n_boot, rng_b)
                      for name in outs if name != 'A'}}
    if b5ab is not None:
        row['policies']['B5'] = list(b5ab)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True,
                        help='directory holding the archived results/ and b5/ folders')
    parser.add_argument('--envs', nargs='+', default=['E-fast', 'E-slow', 'E-outage'])
    parser.add_argument('--flows', nargs='+', default=['F1', 'F2', 'F3'])
    parser.add_argument('--n-tune', type=int, help='override (smoke only; disables the identity check)')
    parser.add_argument('--n-eval', type=int, help='override (smoke only; disables the identity check)')
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed-boot', type=int, default=11)
    parser.add_argument('--h', type=float,
                        help='misuse weight; re-solves A and re-tunes B3, B4, B5 at it. '
                             'Omit to reuse the archived h = 1 fits.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    check = args.n_tune is None and args.n_eval is None
    suffix = '' if args.h is None else f'_h{args.h:g}'
    for env_name in args.envs:
        for flow in args.flows:
            path = args.out / f'security_{env_name}_{flow}{suffix}.json'
            if path.exists():
                print('skip', path, flush=True); continue
            row = run_cell(env_name, flow, args.archive, args.n_tune, args.n_eval,
                           args.n_boot, args.seed_boot, check, args.h)
            path.write_text(json.dumps(row, indent=1))
            o = row['outcomes']
            print(f"{env_name:>9} {flow}  " + '  '.join(
                f"{p}: mg {o[p]['misuse_grant_rate']*100:5.2f}% me {o[p]['misuse_exposure']:.3f} "
                f"ue {o[p]['unpaid_exposure']:.4f} lr {o[p]['legitimate_refusal_rate']*100:5.2f}%"
                for p in o), flush=True)


if __name__ == '__main__':
    main()
