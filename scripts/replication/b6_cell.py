#!/usr/bin/env python3
"""B6: a regime-conditional two-threshold rule on the E-outage rail.

B6 reads the rail regime at arrival and applies (a_run, b_run) when the
rail is running and (a_halt, b_halt) when it is halted.  The four
thresholds are tuned on the same tuning split as B3, B4 and B5, by
payoff, with no knowledge of the model: running and halted arrivals are
disjoint payments, so the two pairs are tuned separately, each on its own
subset.  Two grids are reported: the declared 21-point grid of B3/B4 and
the refined 161-point grid of B5.  The halted pair is searched over a <= b,
so that a rule with no verify interval (including "reject every halted
arrival", a = b = 0) is available to the tuner; the running pair uses the
a < b grid of B3 so that B6's running half is exactly a B3 fit.

Evaluation is the main run's evaluation split.  The archived policies
A, B3, B4 and B5 are replayed beside B6 and the run is refused unless
their mean payoffs equal the archived ones (the identity check of the
security replay).
"""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import replace
from pathlib import Path
import numpy as np

from arena.experiments.settlement.b5 import StateMatchedOutagePolicy
from arena.experiments.settlement.core import GRANT, REJECT, VERIFY, rho_hat_from_q
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (compile_outage, draw_outage_batch, survival,
                                                 replay_outage, window_AD as outage_window_AD)
from arena.experiments.settlement.policies import B3, suspicion_grid
from arena.experiments.settlement.run import OB
from arena.experiments.settlement.watch import (PAYMENTS_PER_EPISODE,
                                                FixedActionOutageWatchPolicy as OB4Force,
                                                OutageWatchBandPolicy)
from arena.experiments.settlement.security_replay import (SEEDS, archived, block_ci, summarize,
                                                          replay_outage_fields, _now,
                                                          DECLARED_GRID_N, REFINED_GRID_N)


def _runner_id():
    with open(__file__, 'rb') as fh:
        return {'base_commit': 'feb22194ee1adff4be03a199b9f53d222569c0b6',
                'module': 'b6_cell.py',
                'module_sha256': hashlib.sha256(fh.read()).hexdigest()}


RUNNER_ID = _runner_id()


def grid_le(n):
    axis = np.linspace(0.0, 1.0, n)
    return [(a, b) for a in axis for b in axis if a <= b]


class RegimeBandPolicy:
    """(a_run, b_run) when the arrival regime is running, (a_halt, b_halt) when halted."""

    def __init__(self, run_ab, halt_ab):
        self.run_ab, self.halt_ab = run_ab, halt_ab

    def __call__(self, i, l, r, v, pi):
        a, b = self.halt_ab if r == 1 else self.run_ab
        if pi < a:
            return GRANT
        if pi > b:
            return REJECT
        return VERIFY


def tune_pair(grant, reject, verify, pi, grid):
    """Best (a, b) on `grid` for the subset whose forced payoffs are given."""
    best, best_v = None, -np.inf
    for a, b in grid:
        sel = np.where(pi < a, grant, np.where(pi > b, reject, verify))
        val = float(sel.mean())
        if val > best_v:
            best, best_v = (float(a), float(b)), val
    return best, best_v


def run_cell(flow_name, archive, n_boot, seed_boot, out_dir):
    env_name = 'E-outage'
    seed = SEEDS[(env_name, flow_name)]
    kind, env, _ = envs_for('mid')[env_name]
    assert kind == 'outage'
    flow = make_flows()[flow_name]
    duel, b4j, b5j = archived(archive, env_name, flow_name, seed)
    n_tune, n_eval = duel['n_tune'], duel['n_eval']
    rng = np.random.default_rng(seed)
    b3a, b3b = duel['payload']['calib']['b_params']['B3']
    k4, a4, b4 = b4j['payload']['b4']['k'], b4j['payload']['b4']['a'], b4j['payload']['b4']['b']
    b5ab = (b5j['b5']['a'], b5j['b5']['b'])
    tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=PAYMENTS_PER_EPISODE)
    eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=PAYMENTS_PER_EPISODE)
    _, _, ex = outage_window_AD(env, survival(env))
    q_hat = float((tune_d.t_ans <= env.tau).mean())
    if abs(q_hat - duel['payload']['calib']['q_hat']) > 1e-12:
        raise RuntimeError('tuning draws differ from the archived run')
    rho_hat = rho_hat_from_q(q_hat, env.tau)

    # forced payoffs at arrival (k = 0) on the tuning split, then split by regime at arrival
    forced = {act: replay_outage(env, tune_d, OB4Force(0, act, env.H, env.N), ex)
              for act in (GRANT, REJECT, VERIFY)}
    r0 = np.asarray(tune_d.paths[:, 0]).astype(int)
    pi = np.asarray(tune_d.pi0, dtype=float)
    fits = {}
    for grid_name, n_grid in (('declared', DECLARED_GRID_N), ('refined', REFINED_GRID_N)):
        g_run = suspicion_grid(n_grid)
        g_halt = grid_le(n_grid)
        sub = {}
        for reg_name, reg, grid in (('run', 0, g_run), ('halt', 1, g_halt)):
            m = r0 == reg
            ab, val = tune_pair(forced[GRANT][m], forced[REJECT][m], forced[VERIFY][m], pi[m], grid)
            # the value of the three pure actions on the same subset, for the record
            sub[reg_name] = {'a': ab[0], 'b': ab[1], 'tune_mean': val, 'n': int(m.sum()),
                             'pure': {'grant': float(forced[GRANT][m].mean()),
                                      'reject': float(forced[REJECT][m].mean()),
                                      'verify': float(forced[VERIFY][m].mean())}}
        fits[grid_name] = sub

    policies = {'A': compile_outage(replace(env, rho=rho_hat), 'A'), 'B3': OB(B3(b3a, b3b)),
                'B4': OutageWatchBandPolicy(k4, a4, b4, env.H, env.N),
                'B5': StateMatchedOutagePolicy(*b5ab)}
    for grid_name in ('declared', 'refined'):
        f = fits[grid_name]
        policies[f'B6_{grid_name}'] = RegimeBandPolicy((f['run']['a'], f['run']['b']),
                                                       (f['halt']['a'], f['halt']['b']))
    outs = {name: replay_outage_fields(env, eval_d, pol, ex) for name, pol in policies.items()}
    episodes = np.arange(n_eval) // PAYMENTS_PER_EPISODE
    theta = eval_d.theta
    expected = {'A': duel['payload']['means']['A2'], 'B3': duel['payload']['means']['B3'],
                'B4': b4j['payload']['b4_mean'], 'B5': b5j['b5_mean']}
    identity = {name: {'replayed': float(outs[name][:, 0].mean()), 'archived': expected[name],
                       'gap': float(outs[name][:, 0].mean() - expected[name])} for name in expected}
    bad = {k: v for k, v in identity.items() if abs(v['gap']) > 1e-9}
    if bad:
        raise RuntimeError(f'replayed payoffs differ from the archive: {bad}')

    rng_b = np.random.default_rng([seed_boot, seed])
    v_mean = float(eval_d.v.mean())
    r0e = np.asarray(eval_d.paths[:, 0]).astype(int)
    halted = r0e == 1
    paired = {}
    for name in ('B6_declared', 'B6_refined'):
        for other in ('A', 'B5', 'B3', 'B4'):
            d = outs[other][:, 0] - outs[name][:, 0]
            ci = block_ci(d, episodes, n_boot, rng_b)
            paired[f'{other}_minus_{name}'] = {'mean': float(d.mean()), 'bp': float(d.mean() / 49.50 * 1e4),
                                                'block_ci95': ci, 'bp_ci95': [c / 49.50 * 1e4 for c in ci]}
    subset = {name: {'halted_mean_payoff': float(outs[name][halted, 0].mean()),
                     'halted_release_rate': float(outs[name][halted, 1].mean()),
                     'halted_query_rate': float(outs[name][halted, 6].mean()),
                     'n_halted': int(halted.sum())} for name in outs}
    row = {'environment': env_name, 'flow': flow_name, 'seed': seed, 'n_tune': n_tune, 'n_eval': n_eval,
           'q_hat': q_hat, 'rho_hat': rho_hat, 'mean_exposure': v_mean, 'declared_mean_exposure': 49.50,
           'bootstrap': {'n_boot': n_boot, 'seed_boot': seed_boot, 'block': PAYMENTS_PER_EPISODE,
                         'n_blocks': int(episodes.max()) + 1},
           'runner': RUNNER_ID, 'created_utc': _now(),
           'source': {'code': duel.get('code'), 'params_hash': duel.get('params_hash'),
                      'b4_params_hash': b4j.get('params_hash'), 'b5_params_hash': b5j.get('params_hash')},
           'policies': {'B3': [b3a, b3b], 'B4': [k4, a4, b4], 'B5': list(b5ab)},
           'b6_fits': fits, 'identity_check': identity,
           'payoff': {name: float(o[:, 0].mean()) for name, o in outs.items()},
           'outcomes': {name: summarize(o, theta, 60.0) for name, o in outs.items()},
           'halted_subset': subset, 'paired': paired}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f'b6_{env_name}_{flow_name}.json'
    p.write_text(json.dumps(row, indent=1))
    f = fits['declared']; g = fits['refined']
    print(f"{flow_name}: declared run=({f['run']['a']:.2f},{f['run']['b']:.2f}) halt=({f['halt']['a']:.2f},{f['halt']['b']:.2f}) n_halt_tune={f['halt']['n']}"
          f" | refined run=({g['run']['a']:.3f},{g['run']['b']:.3f}) halt=({g['halt']['a']:.3f},{g['halt']['b']:.3f})"
          f" | A-B6d {paired['A_minus_B6_declared']['bp']:+.1f}bp A-B6r {paired['A_minus_B6_refined']['bp']:+.1f}bp"
          f" B5-B6r {paired['B5_minus_B6_refined']['bp']:+.1f}bp", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--flows', nargs='+', default=['F1', 'F2', 'F3'])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed-boot', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    for fl in args.flows:
        run_cell(fl, args.archive, args.n_boot, args.seed_boot, args.out)


if __name__ == '__main__':
    main()
