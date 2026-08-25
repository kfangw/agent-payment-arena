"""B4: watch settlement for k ticks, then decide on suspicion.

B4(k, a, b) waits k ticks at the arrival state, paying the same c_w * v
per tick as the wait action, and watching settlement advance.  If
settlement fails or the authorization expires during those ticks the
resource is never released and the payment ends on the wait costs alone.
Otherwise, at tick k, the two-threshold band decides on the unchanged
suspicion: grant below a, reject above b, verify in between.  At k = 0
this is B3.  Unlike the model's wait, the k ticks carry no belief update,
so B4 isolates the value of watching settlement from the value of asking.

The watch horizon is bounded by the settlement chain, not by the answer
deadline: a chain cell reaches FINAL in N+1 ticks and expiry caps the
outage cell at H, so the k grid is clipped to that horizon per cell.

This is an incremental run: it reuses the base cell's draws (same seed,
tune then eval) and reads A2 and the tuned B3 threshold from the base
result file, so A2, B1, B2, B3 are not recomputed.  B4(0, B3*) is
replayed on the reproduced eval draws and must match the base file's B3
block sums exactly, which proves the draws reproduce bit for bit and, with
them, every base policy.  Only B4 is evaluated fresh.

Tuning is over the joint grid (k, a, b); (a, b) reuses B3's grid.  For a
fixed k the three terminal actions are replayed once each on the tuning
draws, and every (a, b) is scored by selecting per payment among those
three arrays on the suspicion band.  On a payment that failed or expired
during the watch the three arrays agree, so the selection equals replaying
B4 itself, at a fraction of the cost.

    python -m duel.b4 --env E-slow --flow F2 --seed 5 --out results
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, rho_hat_from_q, sigma_list
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .outage import (compile_outage, draw_outage_batch, replay_outage,
                     survival, window_AD as outage_window_AD)
from .policies import default_grids
from .report import envelope, jsonable, write_once
from .run import CHAIN_BLOCK
from .simulate import Channel, draw_batch, replay
from .stats import boot_ci, perm_p, ratio_mean

K_BASE = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
PPE = 50


def k_grid(horizon: int) -> list[int]:
    """The declared k ladder clipped to the cell's watch horizon, with the
    horizon itself always included as the top rung."""
    return sorted({k for k in K_BASE if k < horizon} | {0, horizon})


# ---------------- B4 policies ----------------
@dataclass
class B4:
    """Chain B4: wait k ticks, then the two-threshold band."""
    k: int
    a: float
    b: float
    def __call__(self, stage, v, pi):
        if stage < self.k:
            return WAIT
        if pi < self.a:
            return GRANT
        if pi > self.b:
            return REJECT
        return VERIFY


@dataclass
class B4Force:
    """Chain: wait k ticks, then a fixed terminal action (tuning helper)."""
    k: int
    act: int
    def __call__(self, stage, v, pi):
        return WAIT if stage < self.k else self.act


class OB4:
    """Outage B4.  Ticks elapsed since arrival is H - l, so a stateless
    rule knows how long it has watched; it waits while that is below k and
    the payment is still live (not settled, not expired)."""
    def __init__(self, k, a, b, H, N):
        self.k, self.a, self.b, self.H, self.FIN = k, a, b, H, N + 1
    def __call__(self, i, l, r, v, pi):
        if (self.H - l) < self.k and i < self.FIN and l > 0:
            return WAIT
        if pi < self.a:
            return GRANT
        if pi > self.b:
            return REJECT
        return VERIFY


class OB4Force:
    def __init__(self, k, act, H, N):
        self.k, self.act, self.H, self.FIN = k, act, H, N + 1
    def __call__(self, i, l, r, v, pi):
        if (self.H - l) < self.k and i < self.FIN and l > 0:
            return WAIT
        return self.act


# ---------------- joint (k, a, b) tuning ----------------
def _tune_b4(force_fn, ab_grid, k_list, pi0):
    """Score every (k, a, b) on the tuning draws and return the best.

    force_fn(k, act) -> per-payment tuning payoff under 'wait k then act'.
    For each k the three terminal payoffs are computed once; every (a, b)
    selects among them on the suspicion band.
    """
    best, best_val, rows = None, -np.inf, []
    for k in k_list:
        g = force_fn(k, GRANT)
        r = force_fn(k, REJECT)
        w = force_fn(k, VERIFY)
        k_best, k_val = None, -np.inf
        for a, b in ab_grid:
            sel = np.where(pi0 < a, g, np.where(pi0 > b, r, w))
            val = float(sel.mean())
            if val > k_val:
                k_best, k_val = (a, b), val
        rows.append(dict(k=int(k), a=float(k_best[0]), b=float(k_best[1]),
                         tune_mean=k_val))
        if k_val > best_val:
            best, best_val = (int(k), float(k_best[0]), float(k_best[1])), k_val
    return best, best_val, rows


def _block_sums(arr, episodes, uniq):
    return np.array([float(arr[episodes == e].sum()) for e in uniq])


# ---------------- per-kind runners ----------------
def run_chain_b4(ch: Channel, flow, n_tune, n_eval, seed, base):
    rng = np.random.default_rng(seed)
    tune_d = draw_batch(ch, flow, n_tune, rng)
    eval_d = draw_batch(ch, flow, n_eval, rng)
    ex = np.maximum(sigma_list(ch.f) * (1 + ch.m) - 1.0, 0.0)
    episodes = np.arange(len(eval_d)) // CHAIN_BLOCK
    uniq = np.unique(episodes)

    horizon = ch.N + 1
    ks = k_grid(horizon)
    best, best_val, rows = _tune_b4(
        lambda k, act: replay(ch, tune_d, B4Force(k, act), ex),
        default_grids()['B3'], ks, tune_d.pi0)
    k_star, a_star, b_star = best

    b4_pay = replay(ch, eval_d, B4(k_star, a_star, b_star), ex)
    b3a, b3b = base['b3_params']
    id0 = replay(ch, eval_d, B4(0, b3a, b3b), ex)
    return _finish(b4_pay, id0, episodes, uniq, base,
                   dict(k=k_star, a=a_star, b=b_star, tune_mean=best_val),
                   rows, horizon, ks)


def run_outage_b4(env, flow, n_tune, n_eval, seed, base, ppe=PPE):
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=ppe)
    eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=ppe)
    _, _, ex = outage_window_AD(env, survival(env))
    episodes = np.arange(len(eval_d)) // ppe
    uniq = np.unique(episodes)

    horizon = env.H
    ks = k_grid(horizon)
    best, best_val, rows = _tune_b4(
        lambda k, act: replay_outage(env, tune_d,
                                     OB4Force(k, act, env.H, env.N), ex),
        default_grids()['B3'], ks, tune_d.pi0)
    k_star, a_star, b_star = best

    b4_pay = replay_outage(env, eval_d, OB4(k_star, a_star, b_star, env.H, env.N), ex)
    b3a, b3b = base['b3_params']
    id0 = replay_outage(env, eval_d, OB4(0, b3a, b3b, env.H, env.N), ex)
    return _finish(b4_pay, id0, episodes, uniq, base,
                   dict(k=k_star, a=a_star, b=b_star, tune_mean=best_val),
                   rows, horizon, ks)


def _finish(b4_pay, id0, episodes, uniq, base, b4meta, rows, horizon, ks):
    """Shared tail: block sums, the k=0 bit-identity check against the base
    B3 block sums, and the A2 - B4 paired statistic from base A2."""
    counts = np.asarray(base['block_counts'], dtype=float)
    b4_sums = _block_sums(b4_pay, episodes, uniq)
    id0_sums = _block_sums(id0, episodes, uniq)
    a2_sums = np.asarray(base['a2_block_sums'], dtype=float)
    b3_sums = np.asarray(base['b3_block_sums'], dtype=float)

    # k=0 identity, at the block-sum level: B4(0, B3*) must equal the base
    # B3 block sums exactly.  Equality here proves the eval draws reproduce.
    id_gap = float(np.abs(id0_sums - b3_sums).max())

    diff_sums = a2_sums - b4_sums
    mean = ratio_mean(diff_sums, counts)
    lo, hi = boot_ci(diff_sums, counts)
    p = perm_p(diff_sums, counts)
    payload = dict(
        b4_mean=ratio_mean(b4_sums, counts),
        b4_block_sums=b4_sums, block_counts=counts,
        n_episodes=int(len(counts)),
        a2_minus_b4=dict(mean=mean, ci95=[lo, hi], perm_p=p),
        b4=b4meta, b4_k_rows=rows, horizon=int(horizon), k_grid=ks,
        k0_identity_max_gap=id_gap,
    )
    return payload


def _base_facts(env_name, flow, seed, results_dir):
    """Pull A2 and B3 block sums, block counts, and the tuned B3 threshold
    from the base cell's result file."""
    path = Path(results_dir) / f"duel_{env_name}_{flow}_mid_s{seed}.json"
    d = json.load(open(path))
    p = d["payload"]
    return dict(
        a2_block_sums=p["policies"]["A2"]["block_sums"],
        b3_block_sums=p["policies"]["B3"]["block_sums"],
        block_counts=p["block_counts"],
        b3_params=p["calib"]["b_params"]["B3"],
        mean_exposure=p["mean_exposure"],
        base_means={k: p["means"][k] for k in ("A2", "B1", "B3")},
        base_code=d["code"], base_hash=d["params_hash"], base_path=str(path),
    )


def _resolved(args, kind, env, rho):
    if kind == 'chain':
        env_d = dict(kind=kind, f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw,
                     tau=env.tau, rho=rho)
        horizon = env.N + 1
    else:
        env_d = dict(kind=kind, f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw,
                     tau=env.tau, H=env.H, rho=env.rho, p01=env.p01,
                     p10=env.p10, tick_seconds=env.tick_seconds)
        horizon = env.H
    return dict(env_name=args.env, cell=f"{args.env}x{args.flow}",
                flow=args.flow, cw_key=args.cw, cw_per_s=CW_PER_S[args.cw],
                env=env_d, b3_grid=default_grids()['B3'],
                k_grid=k_grid(horizon))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True,
                    choices=['E-fast', 'E-outage', 'E-slow'])
    ap.add_argument('--flow', required=True, choices=['F1', 'F2', 'F3'])
    ap.add_argument('--cw', default='mid', choices=['high', 'mid', 'low'])
    ap.add_argument('--n-eval', type=int, default=5_663_400)
    ap.add_argument('--n-tune', type=int, default=200_000)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='results')
    args = ap.parse_args(argv)

    kind, env, rho = envs_for(args.cw)[args.env]
    flow = make_flows()[args.flow]
    base = _base_facts(args.env, args.flow, args.seed, args.results)
    if kind == 'chain':
        payload = run_chain_b4(env, flow, args.n_tune, args.n_eval,
                               args.seed, base)
    else:
        payload = run_outage_b4(env, flow, args.n_tune, args.n_eval,
                                args.seed, base)

    if payload['k0_identity_max_gap'] > 0.0:
        raise SystemExit(f"k=0 identity failed against base B3 block sums: "
                         f"max gap {payload['k0_identity_max_gap']:.3e} "
                         f"(draws did not reproduce)")

    payload.update(cw=args.cw, mean_exposure=base['mean_exposure'],
                   base_means=base['base_means'], base_hash=base['base_hash'],
                   base_code=base['base_code'])
    cell = f"{args.env} x {args.flow}"
    obj = envelope('b4', cell, args.seed, args.n_eval, args.n_tune,
                   _resolved(args, kind, env, rho), payload)
    tag = f"{args.env}_{args.flow}_{args.cw}_s{args.seed}"
    path = str(Path(args.out) / f"b4_{tag}.json")
    write_once(path, obj)
    amb = payload['a2_minus_b4']
    print(json.dumps(dict(
        cell=cell, k_star=payload['b4']['k'], a=payload['b4']['a'],
        b=payload['b4']['b'], k0_gap=payload['k0_identity_max_gap'],
        A2=round(base['base_means']['A2'], 6), B3=round(base['base_means']['B3'], 6),
        B4=round(payload['b4_mean'], 6),
        a2_minus_b4=round(amb['mean'], 6),
        ci95=[round(amb['ci95'][0], 6), round(amb['ci95'][1], 6)],
        perm_p=amb['perm_p']), indent=1))
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
