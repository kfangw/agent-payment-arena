"""Single entry point: one command runs one cell of the comparison.

    python -m duel.run --env E-outage --flow F1 --cw mid \
        --n-eval 200000 --n-tune 50000 --seed 1 --out results

Draws the tuning and evaluation batches, compiles family A from
calibration statistics, tunes family B on realized tuning profit,
replays every policy on the same evaluation payments, and writes one
json with per-policy means, the paired A2 - B1 difference with a
bootstrap interval (episode blocks in the regime-switching cell), and
the run's metadata.  The sample size is whatever the caller passes; the
equivalence margin is decided elsewhere, from a pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, sigma_list, rho_hat_from_q
from .flows import make_flows
from .gate import envs_for
from .outage import (OutageEnv, compile_outage, draw_outage_batch,
                     replay_outage, survival, window_AD as outage_window_AD)
from .policies import (B1, B2, B3, compile_A, default_grids, make_family_C,
                       tune)
from .simulate import Channel, draw_batch, replay


# ---------------- adapters: outage signature for B and C families ----
class OB:
    """Wrap a (stage, v, pi) rule for the (i, l, r, v, pi) signature.
    Terminal rules ignore the extra state; the B3 verify fires once at
    the arrival tick, matching its chain-env semantics."""
    def __init__(self, pol):
        self.pol = pol
    def __call__(self, i, l, r, v, pi):
        return self.pol(0 if (i == 0 and l > 0) else i, v, pi)


class OWaitGrant:
    def __init__(self, FIN):
        self.FIN = FIN
    def __call__(self, i, l, r, v, pi):
        return GRANT if i >= self.FIN else WAIT


# ---------------- per-kind machinery ----------------
def run_chain(ch: Channel, rho_true, flow, n_tune, n_eval, seed):
    rng = np.random.default_rng(seed)
    tune_d = draw_batch(ch, flow, n_tune, rng)
    eval_d = draw_batch(ch, flow, n_eval, rng)
    sig = sigma_list(ch.f)
    ex_unit = np.maximum(sig * (1 + ch.m) - 1.0, 0.0)

    q_hat = float((tune_d.t_ans <= ch.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, ch.tau)
    A = {
        'A1': compile_A(ch, 'A1', pmf=ch.pmf_h),
        'A2': compile_A(ch, 'A2', rho=rho_hat),
        'A2v': compile_A(ch, 'A2v', rho=rho_hat, drop=(VERIFY,)),
        'A2w': compile_A(ch, 'A2w', rho=rho_hat, drop=(WAIT,)),
    }
    grids = default_grids()
    tuned, params = {}, {}
    for name, make, grid in (
            ('B1', lambda t: B1(t, ch.h), grids['B1']),
            ('B2', lambda t: B2(t), grids['B2']),
            ('B3', lambda ab: B3(*ab), grids['B3'])):
        best, pol, _ = tune(make, grid, ch, tune_d, ex_unit)
        tuned[name] = pol
        params[name] = best
    pols = dict(A, **tuned, **make_family_C(ch))
    out = {k: replay(ch, eval_d, p, ex_unit) for k, p in pols.items()}
    meta = dict(q_hat=q_hat, rho_hat=rho_hat, b_params=_jsonable(params))
    return out, eval_d, meta, None


def run_outage(env: OutageEnv, flow, n_tune, n_eval, seed,
               payments_per_episode=50):
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, flow, n_tune, rng,
                               payments_per_episode=payments_per_episode)
    eval_d = draw_outage_batch(env, flow, n_eval, rng,
                               payments_per_episode=payments_per_episode)
    sig = survival(env)
    _, _, ex = outage_window_AD(env, sig)

    q_hat = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, env.tau)
    env_hat = OutageEnv(f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw,
                        tau=env.tau, H=env.H, rho=rho_hat,
                        p01=env.p01, p10=env.p10,
                        tick_seconds=env.tick_seconds)
    A = {
        'A1': compile_outage(env, 'A1'),
        'A2': compile_outage(env_hat, 'A2'),
        'A2v': compile_outage(env_hat, 'A2v', drop=(VERIFY,)),
        'A2w': compile_outage(env_hat, 'A2w', drop=(WAIT,)),
    }
    grids = default_grids()
    tuned, params = {}, {}
    for name, make, grid in (
            ('B1', lambda t: OB(B1(t, env.h)), grids['B1']),
            ('B2', lambda t: OB(B2(t)), grids['B2']),
            ('B3', lambda ab: OB(B3(*ab)), grids['B3'])):
        best, best_pol = None, None
        best_val = -np.inf
        for g in grid:
            pol = make(g)
            val = float(replay_outage(env, tune_d, pol, ex).mean())
            if val > best_val:
                best, best_pol, best_val = g, pol, val
        tuned[name] = best_pol
        params[name] = best
    C = {'C1': OB(lambda s, v, pi: GRANT), 'C2': OB(lambda s, v, pi: REJECT),
         'C3': OB(lambda s, v, pi: VERIFY), 'C4': OWaitGrant(env.N + 1)}
    pols = dict(A, **tuned, **C)
    out = {k: replay_outage(env, eval_d, p, ex) for k, p in pols.items()}
    meta = dict(q_hat=q_hat, rho_hat=rho_hat, b_params=_jsonable(params))
    episodes = np.arange(len(eval_d)) // payments_per_episode
    return out, eval_d, meta, episodes


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def paired_ci(diff, episodes=None, n_boot=10_000, seed=7, level=0.95):
    """Paired bootstrap CI of the mean difference.  With episodes given,
    resampling is by episode block (regime cells)."""
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    if episodes is None:
        n = len(diff)
        idx = rng.integers(0, n, size=(n_boot, n))
        means = diff[idx].mean(axis=1)
    else:
        uniq = np.unique(episodes)
        by_ep = [diff[episodes == e] for e in uniq]
        sums = np.array([b.sum() for b in by_ep])
        cnts = np.array([len(b) for b in by_ep])
        pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True,
                    choices=['E-fast', 'E-outage', 'E-slow'])
    ap.add_argument('--flow', required=True, choices=['F1', 'F2', 'F3'])
    ap.add_argument('--cw', default='mid', choices=['high', 'mid', 'low'])
    ap.add_argument('--n-eval', type=int, default=100_000)
    ap.add_argument('--n-tune', type=int, default=50_000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='results')
    args = ap.parse_args(argv)

    kind, env, rho = envs_for(args.cw)[args.env]
    flow = make_flows()[args.flow]
    if kind == 'chain':
        out, d, meta, episodes = run_chain(env, rho, flow,
                                           args.n_tune, args.n_eval, args.seed)
    else:
        out, d, meta, episodes = run_outage(env, flow,
                                            args.n_tune, args.n_eval, args.seed)

    diff = out['A2'] - out['B1']
    lo, hi = paired_ci(diff, episodes)
    per_1000 = 1000.0
    bp = 1e4 / float(np.mean(d.v))
    res = dict(
        cell=f"{args.env}x{args.flow}", cw=args.cw, seed=args.seed,
        n_eval=args.n_eval, n_tune=args.n_tune,
        means={k: float(v.mean()) for k, v in out.items()},
        a2_minus_b1=dict(mean=float(diff.mean()),
                         per_1000=float(diff.mean() * per_1000),
                         bp_of_exposure=float(diff.mean() * bp),
                         ci95=[lo, hi],
                         ci95_per_1000=[lo * per_1000, hi * per_1000]),
        calib=meta,
        code=_git_rev(),
    )
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.env}_{args.flow}_{args.cw}_s{args.seed}"
    path = os.path.join(args.out, f"duel_{tag}.json")
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(dict(cell=res['cell'], means=res['means'],
                          a2_minus_b1=res['a2_minus_b1']), indent=1))
    print(f"wrote {path}")


def _git_rev():
    try:
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return rev or None
    except Exception:
        return None


if __name__ == '__main__':
    main()
