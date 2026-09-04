"""Single entry point: one command runs one cell of the comparison.

    python -m arena.experiments.settlement.run --env E-outage --flow F1 --cw mid \
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
import json
from pathlib import Path

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, sigma_list, rho_hat_from_q
from .flows import FLOW_SPEC, make_flows
from .gate import CW_PER_S, envs_for
from .outage import (
    OutageEnv,
    compile_outage,
    draw_outage_batch,
    replay_outage,
    survival,
    window_AD as outage_window_AD,
)
from .policies import B1, B2, B3, compile_A, default_grids, make_family_C, tune
from .report import envelope, jsonable, write_once
from .simulate import Channel, draw_batch, replay

CHAIN_BLOCK = 1000  # payments per episode for the independent chain cells


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
        "A_full": compile_A(ch, "A_full", pmf=ch.pmf_h),
        "A": compile_A(ch, "A", rho=rho_hat),
        "A_noV": compile_A(ch, "A_noV", rho=rho_hat, drop=(VERIFY,)),
        "A_noW": compile_A(ch, "A_noW", rho=rho_hat, drop=(WAIT,)),
    }
    grids = default_grids()
    tuned, params = {}, {}
    for name, make, grid in (
        ("B1", lambda t: B1(t, ch.h), grids["B1"]),
        ("B2", lambda t: B2(t), grids["B2"]),
        ("B3", lambda ab: B3(*ab), grids["B3"]),
    ):
        best, pol, _ = tune(make, grid, ch, tune_d, ex_unit)
        tuned[name] = pol
        params[name] = best
    pols = dict(A, **tuned, **make_family_C(ch))
    out = {k: replay(ch, eval_d, p, ex_unit) for k, p in pols.items()}
    meta = dict(q_hat=q_hat, rho_hat=rho_hat, b_params=jsonable(params))
    episodes = np.arange(len(eval_d)) // CHAIN_BLOCK
    return out, eval_d, meta, episodes


def run_outage(env: OutageEnv, flow, n_tune, n_eval, seed, payments_per_episode=50):
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=payments_per_episode)
    eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=payments_per_episode)
    sig = survival(env)
    _, _, ex = outage_window_AD(env, sig)

    q_hat = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, env.tau)
    env_hat = OutageEnv(
        f=env.f,
        m=env.m,
        h=env.h,
        C=env.C,
        cw=env.cw,
        tau=env.tau,
        H=env.H,
        rho=rho_hat,
        p01=env.p01,
        p10=env.p10,
        tick_seconds=env.tick_seconds,
    )
    A = {
        "A_full": compile_outage(env, "A_full"),
        "A": compile_outage(env_hat, "A"),
        "A_noV": compile_outage(env_hat, "A_noV", drop=(VERIFY,)),
        "A_noW": compile_outage(env_hat, "A_noW", drop=(WAIT,)),
    }
    grids = default_grids()
    tuned, params = {}, {}
    for name, make, grid in (
        ("B1", lambda t: OB(B1(t, env.h)), grids["B1"]),
        ("B2", lambda t: OB(B2(t)), grids["B2"]),
        ("B3", lambda ab: OB(B3(*ab)), grids["B3"]),
    ):
        best, best_pol = None, None
        best_val = -np.inf
        for g in grid:
            pol = make(g)
            val = float(replay_outage(env, tune_d, pol, ex).mean())
            if val > best_val:
                best, best_pol, best_val = g, pol, val
        tuned[name] = best_pol
        params[name] = best
    C = {
        "C1": OB(lambda s, v, pi: GRANT),
        "C2": OB(lambda s, v, pi: REJECT),
        "C3": OB(lambda s, v, pi: VERIFY),
        "C4": OWaitGrant(env.N + 1),
    }
    pols = dict(A, **tuned, **C)
    out = {k: replay_outage(env, eval_d, p, ex) for k, p in pols.items()}
    meta = dict(q_hat=q_hat, rho_hat=rho_hat, b_params=jsonable(params))
    episodes = np.arange(len(eval_d)) // payments_per_episode
    return out, eval_d, meta, episodes


def block_stats(arr, episodes):
    """Per-episode sums of `arr` and the shared per-episode counts.  Every
    policy replays the same evaluation draws, so counts are shared and any
    paired difference is formed downstream by subtracting block sums."""
    uniq = np.unique(episodes)
    sums = np.array([float(arr[episodes == e].sum()) for e in uniq])
    counts = np.array([int((episodes == e).sum()) for e in uniq])
    return sums, counts


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


def _resolved(args, kind, env, rho):
    """The resolved parameter dict that params_hash digests: environment
    constants, the flow spec, and the tuning grids."""
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
        env_name=args.env,
        cell=f"{args.env}x{args.flow}",
        flow=args.flow,
        flow_spec=FLOW_SPEC,
        cw_key=args.cw,
        cw_per_s=CW_PER_S[args.cw],
        env=env_d,
        grids=default_grids(),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["E-fast", "E-outage", "E-slow", "E-slow-deep"])
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--cw", default="mid", choices=["high", "mid", "low"])
    ap.add_argument("--n-eval", type=int, default=100_000)
    ap.add_argument("--n-tune", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    kind, env, rho = envs_for(args.cw)[args.env]
    flow = make_flows()[args.flow]
    if kind == "chain":
        out, d, meta, episodes = run_chain(env, rho, flow, args.n_tune, args.n_eval, args.seed)
    else:
        out, d, meta, episodes = run_outage(env, flow, args.n_tune, args.n_eval, args.seed)

    # Per-policy episode block sums with shared counts; downstream forms any
    # paired difference by subtracting sums (spec 0: the interchange format).
    counts = None
    policies = {}
    for name, arr in out.items():
        sums, cnts = block_stats(arr, episodes)
        policies[name] = dict(block_sums=sums)
        counts = cnts
    payload = dict(
        cw=args.cw,
        policies=policies,
        block_counts=counts,
        n_episodes=int(len(counts)),
        mean_exposure=float(np.mean(d.v)),
        means={k: float(v.mean()) for k, v in out.items()},
        calib=meta,
    )
    cell = f"{args.env} x {args.flow}"
    env_obj = envelope(
        "settlement",
        cell,
        args.seed,
        args.n_eval,
        args.n_tune,
        _resolved(args, kind, env, rho),
        payload,
    )

    tag = f"{args.env}_{args.flow}_{args.cw}_s{args.seed}"
    path = str(Path(args.out) / f"settlement_{tag}.json")
    write_once(path, env_obj)
    diff = out["A"] - out["B1"]
    print(
        json.dumps(
            dict(
                cell=cell,
                means={k: round(float(v.mean()), 6) for k, v in out.items()},
                a2_minus_b1_mean=round(float(diff.mean()), 6),
            ),
            indent=1,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
