"""T7: the arrival-shape axis (experiment 2's fifth axis).

Every cell draws the answer arrival geometric and the compiler calibrates
geometric, so the identification interval of Theorem 3 collapses to a point
and partial identification is never tested.  This axis changes the arrival
SHAPE while holding (q, tau) fixed; the compiler still calibrates geometric,
which is exactly the situation Theorem 3 covers.  Three shapes on the two
main-narrative cells, at the reduced sample: geometric (baseline), uniform
over 1..tau, and bimodal (most mass early, a tail near the deadline).

    python -m duel.shape --cell "E-slow x F2" --shape uniform --seed 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .watch import (
    FixedActionOutageWatchPolicy as OB4Force,
    OutageWatchBandPolicy as OB4,
    block_sums,
    horizon_grid as k_grid,
    tune_watch_policy as tune_b4,
)
from .core import rho_hat_from_q, sigma_list
from .design import eps_for
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .gitcheck import require_clean_tree
from .inject import parse_cell, tune_outage_baseline
from .outage import compile_outage, draw_outage_batch_crn, replay_outage, survival, window_AD
from .policies import B1, compile_A, default_grids, tune
from .report import envelope, write_once
from .run import CHAIN_BLOCK
from .simulate import draw_batch_crn, replay, retime_answers
from .stats import boot_ci, ratio_mean, units

SHAPES = ["geometric", "uniform", "bimodal"]


def shaped_pmf(shape, rho, tau):
    """Answer-arrival pmf over s=1..tau with the same q = 1-(1-rho)^tau as
    the geometric baseline, reshaped.  Mass 1-q stays on 'never answers'."""
    q = 1.0 - (1.0 - rho) ** tau
    if shape == "geometric":
        return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])
    if shape == "uniform":
        return np.full(tau, q / tau)
    if shape == "bimodal":
        w = max(1, round(0.1 * tau))
        pmf = np.zeros(tau)
        pmf[:w] = 0.7 * q / w  # early mass
        pmf[-w:] += 0.3 * q / w  # deadline tail
        return pmf
    raise ValueError(shape)


def _retimed(d, u_ans, pmf, tau):
    return replace(d, t_ans=retime_answers(d.theta, u_ans, pmf, pmf, tau))


def run_chain(env, rho, flow, seed, shape, n_tune, n_eval):
    rng = np.random.default_rng(seed)
    tune_d, u_t = draw_batch_crn(env, flow, n_tune, rng)
    eval_d, u_e = draw_batch_crn(env, flow, n_eval, rng)
    pmf = shaped_pmf(shape, rho, env.tau)
    tu = _retimed(tune_d, u_t, pmf, env.tau)
    ev = _retimed(eval_d, u_e, pmf, env.tau)
    ex = np.maximum(sigma_list(env.f) * (1 + env.m) - 1.0, 0.0)
    q = float((tu.t_ans <= env.tau).mean())  # compiler stays geometric
    a = compile_A(env, "A", rho=rho_hat_from_q(q, env.tau))
    _, b1, _ = tune(lambda t: B1(t, env.h), default_grids()["B1"], env, tu, ex)
    best, _, _ = tune_b4(
        lambda k, act: replay(env, tu, _B4Force(k, act), ex),
        default_grids()["B3"],
        k_grid(env.N + 1),
        tu.pi0,
    )
    from .b4 import B4

    b4 = B4(*best)
    a_pay = replay(env, ev, a, ex)
    episodes = np.arange(len(ev)) // CHAIN_BLOCK
    return _pack(
        a_pay,
        replay(env, ev, b1, ex),
        replay(env, ev, b4, ex),
        episodes,
        float(np.mean(ev.v)),
        best,
    )


def _B4Force(k, act):
    from .b4 import B4Force

    return B4Force(k, act)


def run_outage(env, flow, seed, shape, n_tune, n_eval):
    rng = np.random.default_rng(seed)
    tune_d, u_t = draw_outage_batch_crn(env, flow, n_tune, rng, payments_per_episode=50)
    eval_d, u_e = draw_outage_batch_crn(env, flow, n_eval, rng, payments_per_episode=50)
    pmf = shaped_pmf(shape, env.rho, env.tau)
    tu = _retimed(tune_d, u_t, pmf, env.tau)
    ev = _retimed(eval_d, u_e, pmf, env.tau)
    _, _, ex = window_AD(env, survival(env))
    q = float((tu.t_ans <= env.tau).mean())
    a = compile_outage(replace(env, rho=rho_hat_from_q(q, env.tau)), "A")
    b1 = tune_outage_baseline(env, tu, ex, default_grids()["B1"])
    best, _, _ = tune_b4(
        lambda k, act: replay_outage(env, tu, OB4Force(k, act, env.H, env.N), ex),
        default_grids()["B3"],
        k_grid(env.H),
        tu.pi0,
    )
    b4 = OB4(best[0], best[1], best[2], env.H, env.N)
    a_pay = replay_outage(env, ev, a, ex)
    episodes = np.arange(len(ev)) // 50
    return _pack(
        a_pay,
        replay_outage(env, ev, b1, ex),
        replay_outage(env, ev, b4, ex),
        episodes,
        float(np.mean(ev.v)),
        best,
    )


def _pack(a_pay, b1_pay, b4_pay, episodes, mexp, best):
    counts = np.bincount(episodes).astype(float)
    return dict(
        ab1=block_sums(a_pay - b1_pay, episodes),
        ab4=block_sums(a_pay - b4_pay, episodes),
        counts=counts,
        mexp=mexp,
        b4=dict(k=best[0], a=best[1], b=best[2]),
    )


def _adv(diff, counts, mexp, seed):
    mean = ratio_mean(diff, counts)
    lo, hi = boot_ci(diff, counts, n_boot=20_000, seed=seed, level=0.95)
    return dict(mean=mean, ci95=[lo, hi], bp=units(mean, mexp)["bp"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--shape", required=True, choices=SHAPES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-eval", type=int, default=100_000)
    ap.add_argument("--n-tune", type=int, default=50_000)
    ap.add_argument("--out", default="results_t7")
    args = ap.parse_args(argv)
    require_clean_tree()

    env_name, flow_name = parse_cell(args.cell)
    kind, env, rho = envs_for("mid")[env_name]
    flow = make_flows()[flow_name]
    if kind == "chain":
        r = run_chain(env, rho, flow, args.seed, args.shape, args.n_tune, args.n_eval)
    else:
        r = run_outage(env, flow, args.seed, args.shape, args.n_tune, args.n_eval)

    eps = eps_for(r["mexp"])
    payload = dict(
        cell=args.cell,
        shape=args.shape,
        mean_exposure=r["mexp"],
        eps=eps,
        a_minus_b1=_adv(r["ab1"], r["counts"], r["mexp"], args.seed),
        a_minus_b4=_adv(r["ab4"], r["counts"], r["mexp"], args.seed + 1),
        b4=r["b4"],
        ab1_block_sums=r["ab1"],
        ab4_block_sums=r["ab4"],
        block_counts=r["counts"],
    )
    resolved = dict(cell=args.cell, shape=args.shape, flow=flow_name, cw_per_s=CW_PER_S["mid"])
    obj = envelope("shape", args.cell, args.seed, args.n_eval, args.n_tune, resolved, payload)
    e, f = env_name, flow_name
    path = str(Path(args.out) / f"shape_{e}_{f}_{args.shape}_s{args.seed}.json")
    write_once(path, obj)
    print(
        json.dumps(
            dict(
                cell=args.cell,
                shape=args.shape,
                a_b1=round(payload["a_minus_b1"]["mean"], 6),
                a_b4=round(payload["a_minus_b4"]["mean"], 6),
                b1_bp=round(payload["a_minus_b1"]["bp"], 2),
                b4_bp=round(payload["a_minus_b4"]["bp"], 2),
            ),
            indent=1,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
