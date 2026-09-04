"""S3: experiment 2 with B4 as the competitor.

The four injection axes of inject.py, but the advantage is A2 - B4 with B4
retuned at every grid point (as B1 is), on the two main-narrative cells.
The A2 side is computed exactly as in the B1 sweep, so the two experiments
differ only in the competitor; retention is A2 - B4 at a level over A2 - B4
with no injection.  The pre-injection edge is small, so the absolute dollars
and the interval are reported next to the ratio, not the ratio alone.

B4 is retuned wherever the environment or the draws it is tuned on move
(prior scale kappa, regime shift delta, response bias lambda); under
measurement noise the true environment is unchanged, so the level-zero B4
is reused, matching the B1 convention.

    python -m duel.inject_b4 --cell "E-slow x F2" --axis delta --seed 5
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .core import rho_hat_from_q
from .design import eps_for
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .inject import (
    AXIS_GRIDS,
    IDENTITY,
    NOISE_REPLICATES,
    chain_context,
    _geom,
    outage_context,
    parse_cell,
)
from .outage import compile_outage, replay_outage, retime_answers_geom
from .policies import compile_A, default_grids
from .report import envelope, write_once
from .simulate import replay, retime_answers
from .stats import boot_ci, ratio_mean
from .watch import (
    FixedActionOutageWatchPolicy as OB4Force,
    FixedActionWatchPolicy as B4Force,
    OutageWatchBandPolicy as OB4,
    WatchBandPolicy as B4,
    horizon_grid as k_grid,
    tune_watch_policy as tune_b4,
)


def _tune_chain_b4(env, tune_d, ex):
    best, _, _ = tune_b4(
        lambda k, act: replay(env, tune_d, B4Force(k, act), ex),
        default_grids()["B3"],
        k_grid(env.N + 1),
        tune_d.pi0,
    )
    return B4(*best)


def _tune_outage_b4(env, tune_d, ex):
    best, _, _ = tune_b4(
        lambda k, act: replay_outage(env, tune_d, OB4Force(k, act, env.H, env.N), ex),
        default_grids()["B3"],
        k_grid(env.H),
        tune_d.pi0,
    )
    return OB4(best[0], best[1], best[2], env.H, env.N)


# ------------------------------------------------------------ chain cell
def _chain_diff_b4(ctx, axis, level):
    """Per-payment A2 - B4 on the evaluation draws at one axis level."""
    env, ex = ctx["env"], ctx["ex"]
    eval_d, tune_d = ctx["eval_d"], ctx["tune_d"]
    if axis == "kappa":
        ev = replace(eval_d, pi0=np.clip(level * eval_d.pi0, 0.0, 1.0))
        tu = replace(tune_d, pi0=np.clip(level * tune_d.pi0, 0.0, 1.0))
        b4 = _tune_chain_b4(env, tu, ex)
        return replay(env, ev, ctx["a2"], ex) - replay(env, ev, b4, ex)
    if axis == "delta":
        d_bar = abs(float(np.mean(env.f)) - env.cw)
        env2 = replace(env, cw=env.cw + level * d_bar)
        a2 = compile_A(env2, "A2", rho=ctx["rho_hat"])
        b4 = _tune_chain_b4(env2, tune_d, ex)
        return replay(env2, eval_d, a2, ex) - replay(env2, eval_d, b4, ex)
    if axis == "lambda":
        rho_m = min(ctx["rho"] * level, 1.0)
        pmf_m = _geom(rho_m, env.tau)
        ev = replace(
            eval_d, t_ans=retime_answers(eval_d.theta, ctx["u_eval"], env.pmf_h, pmf_m, env.tau)
        )
        tu = replace(
            tune_d, t_ans=retime_answers(tune_d.theta, ctx["u_tune"], env.pmf_h, pmf_m, env.tau)
        )
        q_hat = float((tu.t_ans <= env.tau).mean())
        a2 = compile_A(env, "A2", rho=rho_hat_from_q(q_hat, env.tau))
        b4 = _tune_chain_b4(env, tu, ex)
        return replay(env, ev, a2, ex) - replay(env, ev, b4, ex)
    if axis == "noise":
        return _chain_noise_diff_b4(ctx, level)
    raise ValueError(f"unknown axis {axis}")


def _chain_noise_diff_b4(ctx, sigma):
    env, ex, eval_d = ctx["env"], ctx["ex"], ctx["eval_d"]
    b4_pay = replay(env, eval_d, ctx["b4"], ex)
    if sigma == 0.0:
        return replay(env, eval_d, ctx["a2"], ex) - b4_pay
    acc = np.zeros(len(eval_d))
    for r in range(NOISE_REPLICATES):
        rng = np.random.default_rng([ctx.get("seed", 0), 700, r, int(sigma * 1000)])
        z = rng.standard_normal(len(env.f))
        f_hat = np.clip(env.f * np.exp(sigma * z), 0.0, 1.0)
        zq = rng.standard_normal()
        q_hat = float(np.clip(ctx["q_hat"] * np.exp(sigma * zq), 0.0, 1.0))
        a2 = compile_A(replace(env, f=f_hat), "A2", rho=rho_hat_from_q(q_hat, env.tau))
        acc += replay(env, eval_d, a2, ex) - b4_pay
    return acc / NOISE_REPLICATES


# ------------------------------------------------------------ outage cell
def _outage_diff_b4(ctx, axis, level):
    env, ex = ctx["env"], ctx["ex"]
    eval_d, tune_d = ctx["eval_d"], ctx["tune_d"]
    if axis == "kappa":
        ev = replace(eval_d, pi0=np.clip(level * eval_d.pi0, 0.0, 1.0))
        tu = replace(tune_d, pi0=np.clip(level * tune_d.pi0, 0.0, 1.0))
        b4 = _tune_outage_b4(env, tu, ex)
        return replay_outage(env, ev, ctx["a2"], ex) - replay_outage(env, ev, b4, ex)
    if axis == "delta":
        d_bar = abs(float(np.mean(env.f)) - env.cw)
        env2 = replace(env, cw=env.cw + level * d_bar)
        a2 = compile_outage(replace(env2, rho=ctx["rho_hat"]), "A2")
        b4 = _tune_outage_b4(env2, tune_d, ex)
        return replay_outage(env2, eval_d, a2, ex) - replay_outage(env2, eval_d, b4, ex)
    if axis == "lambda":
        rho_m = min(env.rho * level, 1.0 - 1e-12)
        ev = replace(eval_d, t_ans=retime_answers_geom(eval_d.theta, ctx["u_eval"], env.rho, rho_m))
        tu = replace(tune_d, t_ans=retime_answers_geom(tune_d.theta, ctx["u_tune"], env.rho, rho_m))
        q_hat = float((tu.t_ans <= env.tau).mean())
        a2 = compile_outage(replace(env, rho=rho_hat_from_q(q_hat, env.tau)), "A2")
        b4 = _tune_outage_b4(env, tu, ex)
        return replay_outage(env, ev, a2, ex) - replay_outage(env, ev, b4, ex)
    if axis == "noise":
        return _outage_noise_diff_b4(ctx, level)
    raise ValueError(f"unknown axis {axis}")


def _outage_noise_diff_b4(ctx, sigma):
    env, ex, eval_d = ctx["env"], ctx["ex"], ctx["eval_d"]
    b4_pay = replay_outage(env, eval_d, ctx["b4"], ex)
    if sigma == 0.0:
        return replay_outage(env, eval_d, ctx["a2"], ex) - b4_pay
    acc = np.zeros(len(eval_d))
    for r in range(NOISE_REPLICATES):
        rng = np.random.default_rng([ctx.get("seed", 0), 700, r, int(sigma * 1000)])
        z = rng.standard_normal(len(env.f))
        f_hat = np.clip(env.f * np.exp(sigma * z), 0.0, 1.0)
        zq = rng.standard_normal()
        q_hat = float(np.clip(ctx["q_hat"] * np.exp(sigma * zq), 0.0, 1.0))
        a2 = compile_outage(replace(env, f=f_hat, rho=rho_hat_from_q(q_hat, env.tau)), "A2")
        acc += replay_outage(env, eval_d, a2, ex) - b4_pay
    return acc / NOISE_REPLICATES


# ------------------------------------------------------------ driver
def run_axis(cell, axis, seed, n_eval, n_tune, out_dir, cw="mid", levels=None, n_boot=2000):
    """Run one axis on one cell against B4 and write b4_{cell}_{axis}_s{seed}."""
    env_name, flow_name = parse_cell(cell)
    kind, env, rho = envs_for(cw)[env_name]
    flow = make_flows()[flow_name]
    levels = list(AXIS_GRIDS[axis] if levels is None else levels)

    if kind == "chain":
        ctx = chain_context(env, rho, flow, seed, n_tune, n_eval)
        ctx["b4"] = _tune_chain_b4(env, ctx["tune_d"], ctx["ex"])
        diff_fn = _chain_diff_b4
    else:
        ctx = outage_context(env, flow, seed, n_tune, n_eval)
        ctx["b4"] = _tune_outage_b4(env, ctx["tune_d"], ctx["ex"])
        diff_fn = _outage_diff_b4
    ctx["seed"] = seed
    episodes = ctx["episodes"]
    counts = np.bincount(episodes).astype(float)
    mean_exposure = float(np.mean(ctx["eval_d"].v))

    block_sums = {}
    for lv in levels:
        diff = diff_fn(ctx, axis, lv)
        block_sums[lv] = np.bincount(episodes, weights=diff)

    curve = {lv: ratio_mean(block_sums[lv], counts) for lv in levels}
    ident = IDENTITY[axis]
    g0 = curve[ident]
    advantage = []
    for lv in levels:
        lo, hi = boot_ci(block_sums[lv], counts, n_boot=n_boot, seed=seed + 1)
        advantage.append(
            dict(
                level=lv,
                mean=curve[lv],
                ci95=[lo, hi],
                n_ep=int(len(counts)),
                retention=(curve[lv] / g0 if g0 != 0 else None),
            )
        )
    lo0, hi0 = boot_ci(block_sums[ident], counts, n_boot=n_boot, seed=seed + 2)

    payload = dict(
        axis=axis,
        cell=cell,
        levels=levels,
        baseline="B4",
        advantage=advantage,
        no_injection=dict(mean=g0, ci95=[lo0, hi0]),
        eps=eps_for(mean_exposure),
        mean_exposure=mean_exposure,
        replicates=NOISE_REPLICATES if axis == "noise" else 1,
        b4_level0=dict(k=int(ctx["b4"].k), a=float(ctx["b4"].a), b=float(ctx["b4"].b)),
    )
    resolved = dict(
        cell=cell,
        axis=axis,
        cw=cw,
        cw_per_s=CW_PER_S[cw],
        levels=levels,
        flow=flow_name,
        baseline="B4",
        replicates=payload["replicates"],
    )
    obj = envelope("inject_b4", cell, seed, n_eval, n_tune, resolved, payload)
    path = str(Path(out_dir) / f"b4_{env_name}_{flow_name}_{axis}_s{seed}.json")
    write_once(path, obj)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--axis", required=True, choices=list(AXIS_GRIDS))
    ap.add_argument("--cw", default="mid", choices=["high", "mid", "low"])
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results_inject")
    args = ap.parse_args(argv)
    path = run_axis(args.cell, args.axis, args.seed, args.n_eval, args.n_tune, args.out, cw=args.cw)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
