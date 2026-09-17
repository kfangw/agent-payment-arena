"""Cross-rail transfer: carry a rail's fitted rules to a rail it was not tuned on.

The article's motivating claim is that a rule tuned on outcomes has answers for
the rail it has seen and none for a rail it has not, while the compiled policy
computes its numbers from that rail's declared constants.  Nothing in the main
run tests it: every cell tunes and evaluates on the same rail.  This runner
evaluates a source rail's archived B3 and B4 fits, without retuning, on a
target rail's draws, beside the target's own fits and the policy compiled from
the target's constants.  The draws are the target cell's archived draws, so
every comparison is paired.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from arena.experiments.settlement.core import rho_hat_from_q, sigma_list
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (compile_outage, draw_outage_batch,
                                                 survival, window_AD as outage_window_AD)
from arena.experiments.settlement.policies import B3, compile_A
from arena.experiments.settlement.run import CHAIN_BLOCK, OB
from arena.experiments.settlement.simulate import draw_batch
from arena.experiments.settlement.watch import (PAYMENTS_PER_EPISODE, OutageWatchBandPolicy,
                                                WatchBandPolicy, horizon_grid as k_grid)
from arena.experiments.settlement.security_replay import (SEEDS, archived, block_ci,
                                                          replay_chain, replay_outage_fields,
                                                          TICK_SECONDS, _now)
import hashlib


def _runner_id():
    # record this runner's own content hash beside the confirmatory commit
    with open(__file__, 'rb') as fh:
        return {'base_commit': 'feb22194ee1adff4be03a199b9f53d222569c0b6',
                'module': 'transfer_cell.py',
                'module_sha256': hashlib.sha256(fh.read()).hexdigest()}


RUNNER_ID = _runner_id()

def fits(archive, env, flow):
    duel, b4j, _ = archived(Path(archive), env, flow, SEEDS[(env, flow)])
    a, b = duel['payload']['calib']['b_params']['B3']
    return (a, b), (b4j['payload']['b4']['k'], b4j['payload']['b4']['a'], b4j['payload']['b4']['b'])

def run(archive, source, target, flow, n_boot, seed_boot):
    seed = SEEDS[(target, flow)]
    kind, env, _ = envs_for('mid')[target]
    duel, b4j, b5j = archived(Path(archive), target, flow, seed)
    n_tune, n_eval = duel['n_tune'], duel['n_eval']
    rng = np.random.default_rng(seed)
    (ta, tb), (tk, t4a, t4b) = fits(archive, target, flow)
    (sa, sb), (sk, s4a, s4b) = fits(archive, source, flow)

    if kind == 'chain':
        tune_d = draw_batch(env, make_flows()[flow], n_tune, rng)
        eval_d = draw_batch(env, make_flows()[flow], n_eval, rng)
        ex = np.maximum(sigma_list(env.f) * (1 + env.m) - 1.0, 0.0)
        q_hat = float((tune_d.t_ans <= env.tau).mean())
        grid = k_grid(env.N + 1)
        kk_t, kk_s = min(tk, max(grid)), min(sk, max(grid))
        pols = {'A': compile_A(env, 'A', rho=rho_hat_from_q(q_hat, env.tau)),
                'B3_tuned': B3(ta, tb), 'B4_tuned': WatchBandPolicy(kk_t, t4a, t4b),
                'B3_xfer': B3(sa, sb), 'B4_xfer': WatchBandPolicy(kk_s, s4a, s4b)}
        outs = {n: replay_chain(env, eval_d, p, ex) for n, p in pols.items()}
        episodes = np.arange(n_eval) // CHAIN_BLOCK
    else:
        tune_d = draw_outage_batch(env, make_flows()[flow], n_tune, rng,
                                   payments_per_episode=PAYMENTS_PER_EPISODE)
        eval_d = draw_outage_batch(env, make_flows()[flow], n_eval, rng,
                                   payments_per_episode=PAYMENTS_PER_EPISODE)
        _, _, ex = outage_window_AD(env, survival(env))
        q_hat = float((tune_d.t_ans <= env.tau).mean())
        grid = k_grid(env.H)
        kk_t, kk_s = min(tk, max(grid)), min(sk, max(grid))
        from dataclasses import replace as _replace
        pols = {'A': compile_outage(_replace(env, rho=rho_hat_from_q(q_hat, env.tau)), 'A'),
                'B3_tuned': OB(B3(ta, tb)),
                'B4_tuned': OutageWatchBandPolicy(kk_t, t4a, t4b, env.H, env.N),
                'B3_xfer': OB(B3(sa, sb)),
                'B4_xfer': OutageWatchBandPolicy(kk_s, s4a, s4b, env.H, env.N)}
        outs = {n: replay_outage_fields(env, eval_d, p, ex) for n, p in pols.items()}
        episodes = np.arange(n_eval) // PAYMENTS_PER_EPISODE

    # the archived identity still has to hold for the target's own fits
    gap_a = float(outs['A'][:, 0].mean()) - duel['payload']['means']['A2']
    gap_b3 = float(outs['B3_tuned'][:, 0].mean()) - duel['payload']['means']['B3']
    gap_b4 = float(outs['B4_tuned'][:, 0].mean()) - b4j['payload']['b4_mean']
    for nm, g in (('A', gap_a), ('B3_tuned', gap_b3), ('B4_tuned', gap_b4)):
        if abs(g) > 1e-9:
            raise RuntimeError(f'{target} {flow}: {nm} does not reproduce the archive ({g})')

    identity = {'A': {'replayed': float(outs['A'][:, 0].mean()),
                      'archived': duel['payload']['means']['A2'], 'gap': gap_a},
                'B3_tuned': {'replayed': float(outs['B3_tuned'][:, 0].mean()),
                             'archived': duel['payload']['means']['B3'], 'gap': gap_b3},
                'B4_tuned': {'replayed': float(outs['B4_tuned'][:, 0].mean()),
                             'archived': b4j['payload']['b4_mean'], 'gap': gap_b4}}
    src_duel, src_b4, _ = archived(Path(archive), source, flow, SEEDS[(source, flow)])
    rng_b = np.random.default_rng([seed_boot, seed])
    mean_v = float(eval_d.v.mean())
    def pair(x, y):
        d = outs[x][:, 0] - outs[y][:, 0]
        ci = block_ci(d, episodes, n_boot, rng_b)
        return {'mean': float(d.mean()), 'ci95': ci,
                'bp': float(d.mean()) / mean_v * 1e4,
                'bp_ci95': [c / mean_v * 1e4 for c in ci]}
    return {'source': source, 'target': target, 'flow': flow, 'seed': seed,
            'n_eval': n_eval, 'n_tune': n_tune, 'mean_exposure': mean_v,
            'tick_seconds': TICK_SECONDS[target],
            'tick_seconds_source': TICK_SECONDS[source],
            'bootstrap': {'n_boot': n_boot, 'seed_boot': seed_boot,
                          'block': CHAIN_BLOCK if kind == 'chain' else PAYMENTS_PER_EPISODE,
                          'n_blocks': int(episodes.max()) + 1},
            'provenance': {
                'target_fit': {'code': duel.get('code'), 'params_hash': duel.get('params_hash'),
                               'created_utc': duel.get('created_utc'),
                               'b4_params_hash': b4j.get('params_hash')},
                'source_fit': {'code': src_duel.get('code'), 'params_hash': src_duel.get('params_hash'),
                               'created_utc': src_duel.get('created_utc'),
                               'b4_params_hash': src_b4.get('params_hash')},
                'runner': RUNNER_ID, 'created_utc': _now()},
            'identity_check': identity,
            'fits': {'target_B3': [ta, tb], 'target_B4': [kk_t, t4a, t4b],
                     'source_B3': [sa, sb], 'source_B4': [kk_s, s4a, s4b],
                     'source_B4_k_declared': sk, 'target_B4_k_declared': tk},
            'payoff': {n: float(o[:, 0].mean()) for n, o in outs.items()},
            'paired': {'A_minus_B3xfer': pair('A', 'B3_xfer'),
                       'A_minus_B4xfer': pair('A', 'B4_xfer'),
                       'A_minus_B3tuned': pair('A', 'B3_tuned'),
                       'A_minus_B4tuned': pair('A', 'B4_tuned'),
                       'B3tuned_minus_B3xfer': pair('B3_tuned', 'B3_xfer'),
                       'B4tuned_minus_B4xfer': pair('B4_tuned', 'B4_xfer')}}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--source', default='E-fast')
    ap.add_argument('--targets', nargs='+', default=['E-slow', 'E-outage'])
    ap.add_argument('--flows', nargs='+', default=['F2'])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed-boot', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for t in a.targets:
        for f in a.flows:
            p = a.out / f'transfer_{a.source}_to_{t}_{f}.json'
            if p.exists():
                print('skip', p, flush=True); continue
            row = run(a.archive, a.source, t, f, a.n_boot, a.seed_boot)
            p.write_text(json.dumps(row, indent=1))
            pr = row['paired']
            print(f"{a.source}->{t} {f}  "
                  f"A-B3x {pr['A_minus_B3xfer']['bp']:+8.1f}bp  A-B4x {pr['A_minus_B4xfer']['bp']:+8.1f}bp  "
                  f"B3t-B3x {pr['B3tuned_minus_B3xfer']['bp']:+8.1f}bp  "
                  f"B4t-B4x {pr['B4tuned_minus_B4xfer']['bp']:+8.1f}bp", flush=True)

if __name__ == '__main__':
    main()
