"""T5/T6: sensitivity of the surviving margins to unsourced constants.

T5 wiggles the reward constants (C, h, m) one at a time on the three
E-outage cells and reports A minus B4, since those are the cells that carry
the empirical claim.  T6 wiggles the E-slow hazard shape (f0, gamma) on the
three E-slow cells and reports A minus B1, since E-slow still sits in the
confirmatory family even though it no longer holds the margin.  Both retune
the competitor at every point, at the reduced sample, and both report the
margin recomputed at the point's own mean exposure alongside the closed-form
floor and v* so the manuscript table can be checked.

    python -m duel.sweep --env E-outage --flow F2 --seed 8 --C 0.25 --competitor B4
    python -m duel.sweep --env E-slow --flow F2 --seed 5 --f0 0.015 --gamma 0.3 --competitor B1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .b4 import OB4, OB4Force, _block_sums, _tune_b4, k_grid
from .core import derived, rho_hat_from_q, sigma_list
from .design import eps_for
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .gitcheck import require_clean_tree
from .outage import (compile_outage, draw_outage_batch, replay_outage,
                     survival, window_AD)
from .policies import B1, compile_A, default_grids, tune
from .report import envelope, write_once
from .run import CHAIN_BLOCK, OB
from .simulate import draw_batch, replay
from .stats import boot_ci, ratio_mean, units

BASE = dict(C=0.5, h=1.0, m=0.35, f0=0.06, gamma=0.5)


def _closed_form(env, rho, tau):
    d = derived(dict(m=env.m, h=env.h, C=env.C, cw=env.cw, rho=rho, tau=tau))
    return dict(floor=d["floor"], vstar=d["vstar"], q=d["q"])


def run_outage(env, flow, seed, n_tune, n_eval):
    """A - B4 on E-outage with B4 retuned at the reduced sample."""
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=50)
    eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=50)
    q = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q, env.tau)
    env_hat = replace(env, rho=rho_hat)
    a = compile_outage(env_hat, "A")
    _, _, ex = window_AD(env, survival(env))
    best, _, _ = _tune_b4(
        lambda k, act: replay_outage(env, tune_d, OB4Force(k, act, env.H, env.N), ex),
        default_grids()["B3"], k_grid(env.H), tune_d.pi0)
    b4 = OB4(best[0], best[1], best[2], env.H, env.N)
    a_pay = replay_outage(env, eval_d, a, ex)
    b4_pay = replay_outage(env, eval_d, b4, ex)
    episodes = np.arange(len(eval_d)) // 50
    diff = _block_sums(a_pay - b4_pay, episodes)
    counts = np.bincount(episodes).astype(float)
    return diff, counts, float(np.mean(eval_d.v)), dict(k=best[0], a=best[1], b=best[2]), \
        _closed_form(env_hat, rho_hat, env.tau)


def run_chain(env, rho, flow, seed, n_tune, n_eval):
    """A - B1 on a chain cell with B1 retuned at the reduced sample."""
    rng = np.random.default_rng(seed)
    tune_d = draw_batch(env, flow, n_tune, rng)
    eval_d = draw_batch(env, flow, n_eval, rng)
    ex = np.maximum(sigma_list(env.f) * (1 + env.m) - 1.0, 0.0)
    q = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q, env.tau)
    a = compile_A(env, "A", rho=rho_hat)
    _, b1, _ = tune(lambda t: B1(t, env.h), default_grids()["B1"], env, tune_d, ex)
    a_pay = replay(env, eval_d, a, ex)
    b1_pay = replay(env, eval_d, b1, ex)
    episodes = np.arange(len(eval_d)) // CHAIN_BLOCK
    diff = _block_sums(a_pay - b1_pay, episodes)
    counts = np.bincount(episodes).astype(float)
    return diff, counts, float(np.mean(eval_d.v)), dict(theta=b1.theta), \
        _closed_form(env, rho_hat, env.tau)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True,
                    choices=["E-outage", "E-slow"])
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--C", type=float, default=BASE["C"])
    ap.add_argument("--h", type=float, default=BASE["h"])
    ap.add_argument("--m", type=float, default=BASE["m"])
    ap.add_argument("--f0", type=float, default=BASE["f0"])
    ap.add_argument("--gamma", type=float, default=BASE["gamma"])
    ap.add_argument("--competitor", required=True, choices=["B1", "B4"])
    ap.add_argument("--n-eval", type=int, default=100_000)
    ap.add_argument("--n-tune", type=int, default=50_000)
    ap.add_argument("--out", default="results_t5")
    args = ap.parse_args(argv)
    require_clean_tree()

    kind, env, rho = envs_for("mid", m=args.m, h=args.h, C=args.C)[args.env]
    flow = make_flows()[args.flow]
    if args.env == "E-slow":
        env = replace(env, f=args.f0 * args.gamma ** np.arange(8))
    if kind == "outage":
        diff, counts, mexp, comp, cf = run_outage(env, flow, args.seed,
                                                   args.n_tune, args.n_eval)
    else:
        diff, counts, mexp, comp, cf = run_chain(env, rho, flow, args.seed,
                                                 args.n_tune, args.n_eval)

    mean = ratio_mean(diff, counts)
    lo, hi = boot_ci(diff, counts, n_boot=20_000, seed=args.seed, level=0.95)
    eps = eps_for(mexp)
    payload = dict(
        cell=f"{args.env} x {args.flow}", competitor=args.competitor,
        constants=dict(C=args.C, h=args.h, m=args.m, f0=args.f0, gamma=args.gamma),
        adv=dict(mean=mean, ci95=[lo, hi], bp=units(mean, mexp)["bp"]),
        eps=eps, above_margin=bool(lo > eps), mean_exposure=mexp,
        competitor_params=comp, closed_form=cf, block_counts=counts,
        diff_block_sums=diff,
    )
    resolved = dict(cell=f"{args.env}x{args.flow}", flow=args.flow,
                    competitor=args.competitor, cw_per_s=CW_PER_S["mid"],
                    constants=payload["constants"])
    obj = envelope("sweep", f"{args.env} x {args.flow}", args.seed, args.n_eval,
                   args.n_tune, resolved, payload)
    tag = (f"{args.env}_{args.flow}_C{args.C}_h{args.h}_m{args.m}"
           f"_f{args.f0}_g{args.gamma}_s{args.seed}")
    path = str(Path(args.out) / f"sweep_{tag}.json")
    write_once(path, obj)
    print(json.dumps(dict(cell=payload["cell"], comp=args.competitor,
                          const=payload["constants"], adv=round(mean, 6),
                          ci=[round(lo, 6), round(hi, 6)],
                          bp=round(payload["adv"]["bp"], 2),
                          above_margin=payload["above_margin"],
                          floor=round(cf["floor"], 4),
                          vstar=round(cf["vstar"], 2)), indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
