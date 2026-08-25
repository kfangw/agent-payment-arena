"""Regime-shift channel decomposition (R2).

Experiment 2's delta axis moves the waiting-cost rate and reports how far
A2's lead over B1 holds.  This module adds the attribution: at every delta
level it evaluates A2, A2 with verify removed (A\\V), and A2 with wait
removed (A\\W) on the same shifted environment, so the lead can be split
into the part that comes from asking (A - A\\V) and the part that comes
from waiting (A - A\\W).  A2 - B1 is re-derived here too and must match
the injection run's advantage curve on the same seed.

Only the two thick-middle main-narrative cells carry a verify band, so
the decomposition is run on E-slow x F2 and E-outage x F2.

    python -m duel.rshift --cell "E-slow x F2" --seed 2004
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .core import VERIFY, WAIT
from .design import eps_for
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .inject import _chain_ctx, _outage_ctx, parse_cell
from .outage import compile_outage, replay_outage
from .policies import B1, compile_A, tune
from .report import envelope, write_once
from .simulate import replay
from .stats import boot_ci, ratio_mean

DELTA_LEVELS = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


def _blocks(diff, episodes, uniq):
    return np.array([float(diff[episodes == e].sum()) for e in uniq])


def _chain_channels(ctx, level):
    """A2, A\\V, A\\W, B1 per-payment payoffs at one delta level (chain)."""
    env, ex = ctx["env"], ctx["ex"]
    eval_d, tune_d = ctx["eval_d"], ctx["tune_d"]
    d_bar = abs(float(np.mean(env.f)) - env.cw)
    env2 = replace(env, cw=env.cw + level * d_bar)
    a2 = compile_A(env2, "A2", rho=ctx["rho_hat"])
    av = compile_A(env2, "A2v", rho=ctx["rho_hat"], drop=(VERIFY,))
    aw = compile_A(env2, "A2w", rho=ctx["rho_hat"], drop=(WAIT,))
    _, b1, _ = tune(lambda t: B1(t, env2.h), ctx["grid"], env2, tune_d, ex)
    return dict(A=replay(env2, eval_d, a2, ex),
                AV=replay(env2, eval_d, av, ex),
                AW=replay(env2, eval_d, aw, ex),
                B1=replay(env2, eval_d, b1, ex), cw=env2.cw)


def _outage_channels(ctx, level):
    env, ex = ctx["env"], ctx["ex"]
    eval_d, tune_d = ctx["eval_d"], ctx["tune_d"]
    from .inject import _tune_outage_b1
    d_bar = abs(float(np.mean(env.f)) - env.cw)
    env2 = replace(env, cw=env.cw + level * d_bar)
    a2 = compile_outage(replace(env2, rho=ctx["rho_hat"]), "A2")
    av = compile_outage(replace(env2, rho=ctx["rho_hat"]), "A2v", drop=(VERIFY,))
    aw = compile_outage(replace(env2, rho=ctx["rho_hat"]), "A2w", drop=(WAIT,))
    b1 = _tune_outage_b1(env2, tune_d, ex, ctx["grid"])
    return dict(A=replay_outage(env2, eval_d, a2, ex),
                AV=replay_outage(env2, eval_d, av, ex),
                AW=replay_outage(env2, eval_d, aw, ex),
                B1=replay_outage(env2, eval_d, b1, ex), cw=env2.cw)


def run_channels(cell, seed, n_eval, n_tune, out_dir, cw="mid",
                 levels=None, n_boot=2000):
    env_name, flow_name = parse_cell(cell)
    kind, env, rho = envs_for(cw)[env_name]
    flow = make_flows()[flow_name]
    levels = list(DELTA_LEVELS if levels is None else levels)

    if kind == "chain":
        ctx = _chain_ctx(env, rho, flow, seed, n_tune, n_eval)
        chan_fn = _chain_channels
    else:
        ctx = _outage_ctx(env, flow, seed, n_tune, n_eval)
        chan_fn = _outage_channels
    ctx["seed"] = seed
    episodes = ctx["episodes"]
    uniq = np.unique(episodes)
    counts = np.array([int((episodes == e).sum()) for e in uniq], dtype=float)
    mean_exposure = float(np.mean(ctx["eval_d"].v))
    eps = eps_for(mean_exposure)

    rows = []
    for lv in levels:
        c = chan_fn(ctx, lv)
        # channel differences, per-payment, on shared draws
        d_amv = c["A"] - c["AV"]        # A - A\V : asking / verify channel
        d_amw = c["A"] - c["AW"]        # A - A\W : waiting channel
        d_amb1 = c["A"] - c["B1"]       # advantage curve (cross-check)
        s_a = _blocks(c["A"], episodes, uniq)
        s_amv = _blocks(d_amv, episodes, uniq)
        s_amw = _blocks(d_amw, episodes, uniq)
        s_amb1 = _blocks(d_amb1, episodes, uniq)
        row = dict(
            level=lv, cw=float(c["cw"]),
            A=ratio_mean(s_a, counts),
            A_no_V=float(c["AV"].mean()), A_no_W=float(c["AW"].mean()),
            B1=float(c["B1"].mean()),
            asking=ratio_mean(s_amv, counts),      # A - A\V
            waiting=ratio_mean(s_amw, counts),      # A - A\W
            advantage=ratio_mean(s_amb1, counts),   # A - B1
        )
        row["asking_ci95"] = list(boot_ci(s_amv, counts, n_boot=n_boot,
                                           seed=seed + 1))
        row["waiting_ci95"] = list(boot_ci(s_amw, counts, n_boot=n_boot,
                                            seed=seed + 2))
        row["advantage_ci95"] = list(boot_ci(s_amb1, counts, n_boot=n_boot,
                                              seed=seed + 3))
        rows.append(row)

    payload = dict(axis="delta", cell=cell, levels=levels, rows=rows,
                   eps=eps, mean_exposure=mean_exposure)
    resolved = dict(cell=cell, axis="delta_channels", cw=cw,
                    cw_per_s=CW_PER_S[cw], levels=levels, flow=flow_name)
    obj = envelope("rshift", cell, seed, n_eval, n_tune, resolved, payload)
    e, f = env_name, flow_name
    path = str(Path(out_dir) / f"rshift_{e}_{f}_delta_s{seed}.json")
    write_once(path, obj)
    return path, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--cw", default="mid", choices=["high", "mid", "low"])
    ap.add_argument("--n-eval", type=int, default=566_340)
    ap.add_argument("--n-tune", type=int, default=100_000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results_inject")
    args = ap.parse_args(argv)
    path, rows = run_channels(args.cell, args.seed, args.n_eval, args.n_tune,
                              args.out, cw=args.cw)
    for r in rows:
        print(f"  delta={r['level']:+.2f}  A={r['A']:.4f}  "
              f"asking(A-A\\V)={r['asking']:.4f}  "
              f"waiting(A-A\\W)={r['waiting']:.4f}  A-B1={r['advantage']:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
