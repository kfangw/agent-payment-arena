"""Experiment 2: the injection axes.

In a main-narrative cell A2 and B1 face off while one assumption at a
time is violated, and the advantage curve says how far the lead holds.
Four axes: prior miscalibration kappa (a symmetric score distortion),
measurement noise sigma (A2's compile inputs only), regime shift delta (a
waiting-cost move that changes the true environment), and response bias
lambda (a misuse-only answer hazard the operator cannot see).

Every level reuses the same base draw so the curve is not shaken by fresh
randomness; only lambda re-times answers, through the fixed answer
uniforms.  Level zero (kappa=1, sigma=0, delta=0, lambda=1) reproduces
the experiment-1 result on the same seed.

Run:  python -m duel.inject --cell "E-outage x F1" --axis lambda --seed 2001
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .core import rho_hat_from_q, sigma_list
from .design import eps_for
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .outage import (
    compile_outage,
    draw_outage_batch_crn,
    replay_outage,
    retime_answers_geom,
    survival,
    window_AD,
)
from .policies import B1, compile_A, default_grids, tune
from .report import envelope, write_once
from .run import OB, CHAIN_BLOCK
from .simulate import draw_batch_crn, replay, retime_answers
from .stats import boot_ci, ratio_mean

# Axis level grids (spec 2.2).  Signed axes carry both branches; the
# identity level is the no-injection point.
AXIS_GRIDS = {
    "kappa": [0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0],
    "noise": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    "delta": [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "lambda": [0.25, 0.5, 2 / 3, 1.0, 1.5, 2.0, 4.0],
}
IDENTITY = {"kappa": 1.0, "noise": 0.0, "delta": 0.0, "lambda": 1.0}
NOISE_REPLICATES = 20


def parse_cell(cell: str) -> tuple[str, str]:
    """'E-outage x F1' -> ('E-outage', 'F1')."""
    env_name, flow_name = (s.strip() for s in cell.split("x"))
    return env_name, flow_name


def _geom(rho: float, tau: int) -> np.ndarray:
    return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])


# ------------------------------------------------------------ chain cell
def chain_context(env, rho, flow, seed, n_tune, n_eval):
    """Base draws, exercise values, and the level-0 A2 and B1 for a chain
    cell, plus everything the axes reshape."""
    rng = np.random.default_rng(seed)
    tune_d, u_tune = draw_batch_crn(env, flow, n_tune, rng)
    eval_d, u_eval = draw_batch_crn(env, flow, n_eval, rng)
    ex = np.maximum(sigma_list(env.f) * (1 + env.m) - 1.0, 0.0)
    q_hat = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, env.tau)
    a2 = compile_A(env, "A2", rho=rho_hat)
    grid = default_grids()["B1"]
    _, b1, _ = tune(lambda t: B1(t, env.h), grid, env, tune_d, ex)
    return dict(
        env=env,
        rho=rho,
        flow=flow,
        ex=ex,
        q_hat=q_hat,
        rho_hat=rho_hat,
        a2=a2,
        b1=b1,
        grid=grid,
        tune_d=tune_d,
        u_tune=u_tune,
        eval_d=eval_d,
        u_eval=u_eval,
        episodes=np.arange(n_eval) // CHAIN_BLOCK,
    )


def chain_difference(ctx, axis, level):
    """Per-payment A2 - B1 on the evaluation draws at one axis level."""
    env, ex = ctx["env"], ctx["ex"]
    eval_d, tune_d = ctx["eval_d"], ctx["tune_d"]
    if axis == "kappa":
        ev = replace(eval_d, pi0=np.clip(level * eval_d.pi0, 0.0, 1.0))
        tu = replace(tune_d, pi0=np.clip(level * tune_d.pi0, 0.0, 1.0))
        _, b1, _ = tune(lambda t: B1(t, env.h), ctx["grid"], env, tu, ex)
        return replay(env, ev, ctx["a2"], ex) - replay(env, ev, b1, ex)
    if axis == "delta":
        d_bar = abs(float(np.mean(env.f)) - env.cw)
        env2 = replace(env, cw=env.cw + level * d_bar)
        a2 = compile_A(env2, "A2", rho=ctx["rho_hat"])
        _, b1, _ = tune(lambda t: B1(t, env2.h), ctx["grid"], env2, tune_d, ex)
        return replay(env2, eval_d, a2, ex) - replay(env2, eval_d, b1, ex)
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
        return replay(env, ev, a2, ex) - replay(env, ev, ctx["b1"], ex)
    if axis == "noise":
        return _chain_noise_diff(ctx, level)
    raise ValueError(f"unknown axis {axis}")


def _chain_noise_diff(ctx, sigma):
    """Mean A2 - B1 over noise replicates; A2's compile inputs carry the
    noise, the true environment and B1 do not."""
    env, ex, eval_d = ctx["env"], ctx["ex"], ctx["eval_d"]
    b1_pay = replay(env, eval_d, ctx["b1"], ex)
    if sigma == 0.0:
        return replay(env, eval_d, ctx["a2"], ex) - b1_pay
    acc = np.zeros(len(eval_d))
    for r in range(NOISE_REPLICATES):
        rng = np.random.default_rng([ctx.get("seed", 0), 700, r, int(sigma * 1000)])
        z = rng.standard_normal(len(env.f))
        f_hat = np.clip(env.f * np.exp(sigma * z), 0.0, 1.0)
        zq = rng.standard_normal()
        q_hat = float(np.clip(ctx["q_hat"] * np.exp(sigma * zq), 0.0, 1.0))
        a2 = compile_A(replace(env, f=f_hat), "A2", rho=rho_hat_from_q(q_hat, env.tau))
        acc += replay(env, eval_d, a2, ex) - b1_pay
    return acc / NOISE_REPLICATES


# ------------------------------------------------------------ outage cell
def tune_outage_baseline(env, tune_d, ex, grid):
    best_val, best = -np.inf, None
    for g in grid:
        pol = OB(B1(g, env.h))
        val = float(replay_outage(env, tune_d, pol, ex).mean())
        if val > best_val:
            best_val, best = val, pol
    return best


def outage_context(env, flow, seed, n_tune, n_eval, ppe=50):
    rng = np.random.default_rng(seed)
    tune_d, u_tune = draw_outage_batch_crn(env, flow, n_tune, rng, payments_per_episode=ppe)
    eval_d, u_eval = draw_outage_batch_crn(env, flow, n_eval, rng, payments_per_episode=ppe)
    _, _, ex = window_AD(env, survival(env))
    q_hat = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, env.tau)
    a2 = compile_outage(replace(env, rho=rho_hat), "A2")
    grid = default_grids()["B1"]
    b1 = tune_outage_baseline(env, tune_d, ex, grid)
    return dict(
        env=env,
        rho=env.rho,
        flow=flow,
        ex=ex,
        q_hat=q_hat,
        rho_hat=rho_hat,
        a2=a2,
        b1=b1,
        grid=grid,
        tune_d=tune_d,
        u_tune=u_tune,
        eval_d=eval_d,
        u_eval=u_eval,
        episodes=np.arange(n_eval) // ppe,
    )


def outage_difference(ctx, axis, level):
    env, ex = ctx["env"], ctx["ex"]
    eval_d, tune_d = ctx["eval_d"], ctx["tune_d"]
    if axis == "kappa":
        ev = replace(eval_d, pi0=np.clip(level * eval_d.pi0, 0.0, 1.0))
        tu = replace(tune_d, pi0=np.clip(level * tune_d.pi0, 0.0, 1.0))
        b1 = tune_outage_baseline(env, tu, ex, ctx["grid"])
        return replay_outage(env, ev, ctx["a2"], ex) - replay_outage(env, ev, b1, ex)
    if axis == "delta":
        d_bar = abs(float(np.mean(env.f)) - env.cw)
        env2 = replace(env, cw=env.cw + level * d_bar)
        a2 = compile_outage(replace(env2, rho=ctx["rho_hat"]), "A2")
        b1 = tune_outage_baseline(env2, tune_d, ex, ctx["grid"])
        return replay_outage(env2, eval_d, a2, ex) - replay_outage(env2, eval_d, b1, ex)
    if axis == "lambda":
        rho_m = min(env.rho * level, 1.0 - 1e-12)
        ev = replace(eval_d, t_ans=retime_answers_geom(eval_d.theta, ctx["u_eval"], env.rho, rho_m))
        tu = replace(tune_d, t_ans=retime_answers_geom(tune_d.theta, ctx["u_tune"], env.rho, rho_m))
        q_hat = float((tu.t_ans <= env.tau).mean())
        a2 = compile_outage(replace(env, rho=rho_hat_from_q(q_hat, env.tau)), "A2")
        return replay_outage(env, ev, a2, ex) - replay_outage(env, ev, ctx["b1"], ex)
    if axis == "noise":
        return _outage_noise_diff(ctx, level)
    raise ValueError(f"unknown axis {axis}")


def _outage_noise_diff(ctx, sigma):
    env, ex, eval_d = ctx["env"], ctx["ex"], ctx["eval_d"]
    b1_pay = replay_outage(env, eval_d, ctx["b1"], ex)
    if sigma == 0.0:
        return replay_outage(env, eval_d, ctx["a2"], ex) - b1_pay
    acc = np.zeros(len(eval_d))
    for r in range(NOISE_REPLICATES):
        rng = np.random.default_rng([ctx.get("seed", 0), 700, r, int(sigma * 1000)])
        z = rng.standard_normal(len(env.f))
        f_hat = np.clip(env.f * np.exp(sigma * z), 0.0, 1.0)
        zq = rng.standard_normal()
        q_hat = float(np.clip(ctx["q_hat"] * np.exp(sigma * zq), 0.0, 1.0))
        a2 = compile_outage(replace(env, f=f_hat, rho=rho_hat_from_q(q_hat, env.tau)), "A2")
        acc += replay_outage(env, eval_d, a2, ex) - b1_pay
    return acc / NOISE_REPLICATES


# ------------------------------------------------------------ halving
def halving(levels, curve, eps: float) -> dict:
    """Read the halving point off one branch (spec 2.4).  levels[0] is the
    identity level and curve[0] = G(0); the branch runs outward from it.
    Returns the crossing, its status, whether the branch is monotone, and
    the residual advantage at the far end."""
    levels = list(levels)
    curve = list(curve)
    g0 = curve[0]
    monotone = all(b <= a + 1e-12 for a, b in zip(curve, curve[1:]))
    if g0 <= eps:
        return dict(point=None, status="undefined", monotone=monotone, residual=None, target=None)
    target = g0 / 2.0
    for i in range(len(levels) - 1):
        g_a, g_b = curve[i], curve[i + 1]
        if (g_a - target) * (g_b - target) <= 0 and g_a != g_b:
            frac = (target - g_a) / (g_b - g_a)
            x = levels[i] + frac * (levels[i + 1] - levels[i])
            return dict(
                point=float(x), status="ok", monotone=monotone, residual=None, target=float(target)
            )
        if g_a == target:
            return dict(
                point=float(levels[i]),
                status="ok",
                monotone=monotone,
                residual=None,
                target=float(target),
            )
    return dict(
        point=None,
        status="beyond_range",
        monotone=monotone,
        residual=float(curve[-1] / g0),
        target=float(target),
    )


def _branches(axis, levels):
    """Split the level grid at the identity into an outward-running pair
    (positive branch, negative branch); unsigned axes have no negative."""
    ident = IDENTITY[axis]
    pos = sorted((x for x in levels if x >= ident))
    neg = sorted((x for x in levels if x <= ident), reverse=True)
    out = {"pos": pos}
    if len(neg) > 1:
        out["neg"] = neg
    return out


# ------------------------------------------------------------ driver
def run_axis(
    cell: str,
    axis: str,
    seed: int,
    n_eval: int,
    n_tune: int,
    out_dir: str,
    cw: str = "mid",
    levels=None,
    n_boot: int = 2000,
) -> str:
    """Run one axis on one cell and write inject_{cell}_{axis}_s{seed}.json."""
    env_name, flow_name = parse_cell(cell)
    kind, env, rho = envs_for(cw)[env_name]
    flow = make_flows()[flow_name]
    levels = list(AXIS_GRIDS[axis] if levels is None else levels)

    if kind == "chain":
        ctx = chain_context(env, rho, flow, seed, n_tune, n_eval)
        diff_fn = chain_difference
    else:
        ctx = outage_context(env, flow, seed, n_tune, n_eval)
        diff_fn = outage_difference
    ctx["seed"] = seed
    episodes = ctx["episodes"]
    uniq = np.unique(episodes)
    counts = np.array([int((episodes == e).sum()) for e in uniq], dtype=float)
    mean_exposure = float(np.mean(ctx["eval_d"].v))

    # per-level episode block sums of the paired difference
    block_sums = {}
    for lv in levels:
        diff = diff_fn(ctx, axis, lv)
        block_sums[lv] = np.array([float(diff[episodes == e].sum()) for e in uniq])

    curve = {lv: ratio_mean(block_sums[lv], counts) for lv in levels}
    advantage = []
    for lv in levels:
        lo, hi = boot_ci(block_sums[lv], counts, n_boot=n_boot, seed=seed + 1)
        advantage.append(dict(level=lv, mean=curve[lv], ci95=[lo, hi], n_ep=int(len(counts))))

    ident = IDENTITY[axis]
    g0 = curve[ident]
    lo0, hi0 = boot_ci(block_sums[ident], counts, n_boot=n_boot, seed=seed + 2)
    eps = eps_for(mean_exposure)  # frozen anchor (duel.design)

    half = {}
    for name, br in _branches(axis, levels).items():
        pt = halving(br, [curve[x] for x in br], eps)
        pt["ci95"] = _halving_ci(br, block_sums, counts, eps, n_boot, seed)
        half[name] = pt

    payload = dict(
        axis=axis,
        cell=cell,
        levels=levels,
        advantage=advantage,
        no_injection=dict(mean=g0, ci95=[lo0, hi0]),
        halving=half,
        eps=eps,
        mean_exposure=mean_exposure,
        replicates=NOISE_REPLICATES if axis == "noise" else 1,
    )
    resolved = dict(
        cell=cell,
        axis=axis,
        cw=cw,
        cw_per_s=CW_PER_S[cw],
        levels=levels,
        flow=flow_name,
        replicates=payload["replicates"],
    )
    obj = envelope("inject", cell, seed, n_eval, n_tune, resolved, payload)
    e, f = env_name, flow_name
    path = str(Path(out_dir) / f"inject_{e}_{f}_{axis}_s{seed}.json")
    write_once(path, obj)
    return path


def _halving_ci(branch, block_sums, counts, eps, n_boot, seed):
    """Bootstrap CI of the halving point: resample episodes once per draw
    (indices shared across levels) and reread the crossing."""
    rng = np.random.default_rng(seed + 3)
    n = len(counts)
    pts = []
    for _ in range(min(n_boot, 400)):
        pick = rng.integers(0, n, size=n)
        denom = counts[pick].sum()
        cur = [float(block_sums[x][pick].sum()) / denom for x in branch]
        h = halving(branch, cur, eps)
        if h["status"] == "ok":
            pts.append(h["point"])
    if len(pts) < 10:
        return None
    return [float(np.quantile(pts, 0.025)), float(np.quantile(pts, 0.975))]


def main(argv: list[str] | None = None) -> None:
    """CLI entry: run one axis on one cell."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--axis", required=True, choices=list(AXIS_GRIDS))
    ap.add_argument("--cw", default="mid", choices=["high", "mid", "low"])
    ap.add_argument("--n-eval", type=int, default=100_000)
    ap.add_argument("--n-tune", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--out", default="results_inject")
    args = ap.parse_args(argv)
    path = run_axis(args.cell, args.axis, args.seed, args.n_eval, args.n_tune, args.out, cw=args.cw)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
