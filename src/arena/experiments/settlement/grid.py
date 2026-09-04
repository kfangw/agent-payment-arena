"""S1: B3 suspicion-grid refinement.

The declared B3 grid steps suspicion by 0.05 (n = 21).  This sweep re-tunes
B3 only, at n in {21, 41, 81, 161}, on the reproduced tuning draws and
re-scores it on the evaluation draws, to read how much of the A2 - B3 edge
is grid coarseness.  A2, B1, B2 are not recomputed: the sweep reuses the
base cell's draws (same seed) and reads A2 and the base B3 block sums from
the base result file.

For a fixed cell the three terminal legs (grant, reject, verify at the
arrival tick) are replayed once on each split; every (a, b) then selects
among them on the suspicion band, so all four resolutions share the same
six replays.  B3 at n = 21 must reproduce the base B3 block sums exactly,
which proves the draws reproduce and, with them, A2.

    python -m arena.experiments.settlement.grid --env E-slow --flow F2 --seed 5 --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b4 import load_base_facts
from .core import GRANT, REJECT, VERIFY, sigma_list
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .outage import draw_outage_batch, replay_outage, survival, window_AD as outage_window_AD
from .policies import suspicion_grid
from .report import envelope, write_once
from .run import CHAIN_BLOCK
from .simulate import draw_batch, replay
from .stats import ratio_mean
from .watch import (
    PAYMENTS_PER_EPISODE as PPE,
    FixedActionOutageWatchPolicy as OB4Force,
    FixedActionWatchPolicy as B4Force,
    block_sums,
)

NS = [21, 41, 81, 161]


def _terminals_chain(ch, d, ex):
    return (
        replay(ch, d, B4Force(0, GRANT), ex),
        replay(ch, d, B4Force(0, REJECT), ex),
        replay(ch, d, B4Force(0, VERIFY), ex),
    )


def _terminals_outage(env, d, ex):
    return (
        replay_outage(env, d, OB4Force(0, GRANT, env.H, env.N), ex),
        replay_outage(env, d, OB4Force(0, REJECT, env.H, env.N), ex),
        replay_outage(env, d, OB4Force(0, VERIFY, env.H, env.N), ex),
    )


def _select(g, r, w, pi0, a, b):
    """B3(a, b) payoff by choosing among the three terminal legs per
    payment: grant below a, reject above b, verify in the middle band."""
    return np.where(pi0 < a, g, np.where(pi0 > b, r, w))


def _best_ab(g, r, w, pi0, grid):
    best, best_val = None, -np.inf
    for a, b in grid:
        val = float(_select(g, r, w, pi0, a, b).mean())
        if val > best_val:
            best, best_val = (float(a), float(b)), val
    return best, best_val


def run_cell(env_name, flow_name, seed, n_tune, n_eval, results, ns):
    base = load_base_facts(env_name, flow_name, seed, results)
    kind, env, rho = envs_for("mid")[env_name]
    flow = make_flows()[flow_name]
    rng = np.random.default_rng(seed)

    if kind == "chain":
        tune_d = draw_batch(env, flow, n_tune, rng)
        eval_d = draw_batch(env, flow, n_eval, rng)
        ex = np.maximum(sigma_list(env.f) * (1 + env.m) - 1.0, 0.0)
        gt, rt, wt = _terminals_chain(env, tune_d, ex)
        ge, re_, we = _terminals_chain(env, eval_d, ex)
        episodes = np.arange(len(eval_d)) // CHAIN_BLOCK
    else:
        tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=PPE)
        eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=PPE)
        _, _, ex = outage_window_AD(env, survival(env))
        gt, rt, wt = _terminals_outage(env, tune_d, ex)
        ge, re_, we = _terminals_outage(env, eval_d, ex)
        episodes = np.arange(len(eval_d)) // PPE

    pi_tune, pi_eval = tune_d.pi0, eval_d.pi0
    uniq = np.unique(episodes)
    counts = np.asarray(base["block_counts"], dtype=float)
    a2_sums = np.asarray(base["a2_block_sums"], dtype=float)
    b3_base_sums = np.asarray(base["b3_block_sums"], dtype=float)

    curve, id_gap, matched = [], None, None
    for n in ns:
        grid = suspicion_grid(n)
        (a, b), tune_mean = _best_ab(gt, rt, wt, pi_tune, grid)
        b3_sums = block_sums(_select(ge, re_, we, pi_eval, a, b), episodes)
        if n == 21:
            id_gap = float(np.abs(b3_sums - b3_base_sums).max())
            ba, bb = base["b3_params"]
            matched = bool(a == ba and b == bb)
        diff = a2_sums - b3_sums
        curve.append(
            dict(
                n=int(n),
                n_grid=len(grid),
                a=a,
                b=b,
                tune_mean=tune_mean,
                b3_mean=ratio_mean(b3_sums, counts),
                a2_minus_b3=ratio_mean(diff, counts),
            )
        )
    coarse = curve[0]["a2_minus_b3"]
    for row in curve:
        row["coarseness_cost"] = coarse - row["a2_minus_b3"]

    payload = dict(
        curve=curve,
        ns=list(ns),
        k0_identity_max_gap=id_gap,
        n21_ab_matches_base=matched,
        mean_exposure=base["mean_exposure"],
        base_hash=base["base_hash"],
        base_code=base["base_code"],
        base_path=base["base_path"],
    )
    return payload, kind, env, rho


def _resolved(env_name, flow_name, kind, env, rho, ns):
    if kind == "chain":
        env_d = dict(kind=kind, f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw, tau=env.tau, rho=rho)
    else:
        env_d = dict(
            kind=kind,
            f=env.f,
            m=env.m,
            h=env.h,
            C=env.C,
            cw=env.cw,
            tau=env.tau,
            H=env.H,
            rho=env.rho,
            p01=env.p01,
            p10=env.p10,
            tick_seconds=env.tick_seconds,
        )
    return dict(
        env_name=env_name,
        cell=f"{env_name}x{flow_name}",
        flow=flow_name,
        cw_key="mid",
        cw_per_s=CW_PER_S["mid"],
        env=env_d,
        ns=list(ns),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["E-fast", "E-outage", "E-slow"])
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ns", default=",".join(str(n) for n in NS))
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    ns = [int(x) for x in args.ns.split(",")]
    payload, kind, env, rho = run_cell(
        args.env, args.flow, args.seed, args.n_tune, args.n_eval, args.results, ns
    )
    if payload["k0_identity_max_gap"] > 1e-9:
        raise SystemExit(
            f"n=21 identity failed: A2-B3 grid does not reproduce base B3 "
            f"(max gap {payload['k0_identity_max_gap']:.3e})"
        )

    resolved = _resolved(args.env, args.flow, kind, env, rho, ns)
    cell = f"{args.env} x {args.flow}"
    obj = envelope("grid", cell, args.seed, args.n_eval, args.n_tune, resolved, payload)
    tag = f"{args.env}_{args.flow}_mid_s{args.seed}"
    path = str(Path(args.out) / f"settlement_gridN_{tag}.json")
    write_once(path, obj)
    print(
        json.dumps(
            dict(
                cell=cell,
                id_gap=payload["k0_identity_max_gap"],
                n21_matches_base=payload["n21_ab_matches_base"],
                curve=[
                    dict(
                        n=r["n"],
                        ab=[r["a"], r["b"]],
                        a2_minus_b3=round(r["a2_minus_b3"], 6),
                        cost=round(r["coarseness_cost"], 6),
                    )
                    for r in payload["curve"]
                ],
            ),
            indent=1,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
