"""T1: value of information under a hidden coordinate.

R4's ablation projected the optimal table by a majority vote, which is not
the value of a coordinate (it is not monotone in what is hidden).  Here the
optimal policy is re-derived under partial observation: the four action
values from the exact outage DP are averaged over the hidden coordinate's
distribution, and the argmax is taken on the averaged values.  Averaging the
value, not the action, is the difference.

Only r (regime) and v (exposure) are re-derived: the compiled policy is
queried only at the arrival state (i = 0, l = H), which this module asserts
on the evaluation split before anything else, so i and l have nothing to
hide.  The regime carries no observable signal, so belief stays at the
stationary outage probability; exposure is averaged over its declared
distribution.

    python -m duel.voi --flow F2 --seed 8 --out results_t1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b4 import OB4, _block_sums
from .core import GRANT, REJECT, VERIFY, WAIT, rho_hat_from_q
from .gate import CW_PER_S, envs_for
from .gitcheck import require_clean_tree
from .flows import make_flows
from .naming import canon_keys
from .outage import (
    CompiledOutagePolicy,
    OutageEnv,
    PI_GRID_OUTAGE,
    draw_outage_batch,
    replay_outage,
    survival,
    window_AD,
)
from .report import envelope, git_rev, write_once

PPE = 50


def _env_hat(env, tune_d):
    q = float((tune_d.t_ans <= env.tau).mean())
    return OutageEnv(
        f=env.f,
        m=env.m,
        h=env.h,
        C=env.C,
        cw=env.cw,
        tau=env.tau,
        H=env.H,
        rho=rho_hat_from_q(q, env.tau),
        p01=env.p01,
        p10=env.p10,
        tick_seconds=env.tick_seconds,
    )


def _action_values(env, v, pi_grid):
    """The four action values Q[a][i, l, r, pi] from the exact outage DP,
    before the argmax (mirrors outage.value_labels, keeping the stack)."""
    N, H, tau = env.N, env.H, env.tau
    FIN = N + 1
    P = env.trans()
    sig = survival(env)
    A, D, _ = window_AD(env, sig)
    pi = np.asarray(pi_grid)
    kappa = 1 + (1 - pi) * env.m - pi * env.h
    npi = len(pi)
    V = np.zeros((FIN + 1, H + 1, 2, npi))
    Q = {a: np.zeros((FIN + 1, H + 1, 2, npi)) for a in (GRANT, REJECT, VERIFY, WAIT)}
    minf = np.full(npi, -np.inf)
    for l in range(0, H + 1):
        for i in range(FIN, -1, -1):
            for r in (0, 1):
                G = v * (sig[i, l, r] * kappa - 1.0)
                R = np.zeros(npi)
                w = min(tau, l)
                W = (
                    -env.C - v * D[w, i, l, r] + (1 - pi) * v * A[w, i, l, r]
                    if w > 0
                    else np.full(npi, -env.C)
                )
                if l == 0 or i == FIN:
                    Wait = minf
                elif r == 0:
                    cont = np.tensordot(P[0], V[i + 1, l - 1, :, :], axes=(0, 0))
                    Wait = -env.cw * v + (1 - env.f[i]) * cont
                else:
                    cont = np.tensordot(P[1], V[i, l - 1, :, :], axes=(0, 0))
                    Wait = -env.cw * v + cont
                Q[GRANT][i, l, r], Q[REJECT][i, l, r] = G, R
                Q[VERIFY][i, l, r], Q[WAIT][i, l, r] = W, Wait
                V[i, l, r] = np.maximum.reduce([G, R, W, Wait])
    return Q


def _argmax_labels(stacks):
    """Tie order grant<reject<verify<wait over a list of value arrays."""
    order = [GRANT, REJECT, VERIFY, WAIT]
    st = np.stack([stacks[a] for a in order], axis=0)
    return np.asarray(order, dtype=np.int8)[st.argmax(axis=0)]


def _v_weights(v_eval, v_grid):
    """Declared exposure weight per v-grid bin: the eval exposure mass
    assigned to its nearest log grid point."""
    lg = np.log(v_grid)
    lv = np.log(np.maximum(v_eval, 1e-12))
    iv = np.clip(np.searchsorted(lg, lv), 0, len(lg) - 1)
    lo = np.maximum(iv - 1, 0)
    iv = np.where(np.abs(lg[lo] - lv) < np.abs(lg[iv] - lv), lo, iv)
    w = np.bincount(iv, minlength=len(v_grid)).astype(float)
    return w / w.sum()


def rederive(env_hat, v_grid, pi_grid, v_eval, hide):
    """Compiled policy that re-derives the argmax after averaging the action
    values over the hidden coordinates in `hide` (a subset of {'r','v'})."""
    p_out = env_hat.stationary_outage
    rw = np.array([1 - p_out, p_out])  # regime belief
    vw = _v_weights(v_eval, v_grid)
    Qs = [(_action_values(env_hat, v, pi_grid), v) for v in v_grid]

    def marg(Q):
        acts = {}
        for a in (GRANT, REJECT, VERIFY, WAIT):
            q = Q[a]  # (i,l,r,pi)
            if "r" in hide:
                q = rw[0] * q[:, :, 0, :] + rw[1] * q[:, :, 1, :]  # (i,l,pi)
                q = q[:, :, None, :]  # keep r axis
                q = np.repeat(q, 2, axis=2)
            acts[a] = q
        return acts

    if "v" not in hide:
        tabs = [_argmax_labels(marg(Q)) for Q, _ in Qs]
    else:
        # average the (already r-margined) values over the exposure weights
        acc = {a: 0.0 for a in (GRANT, REJECT, VERIFY, WAIT)}
        for (Q, _), wv in zip(Qs, vw):
            if wv == 0.0:  # empty bin: avoid 0 * -inf
                continue
            m = marg(Q)
            for a in acc:
                acc[a] = acc[a] + wv * m[a]
        lab = _argmax_labels(acc)
        tabs = [lab for _ in v_grid]
    name = "A_hide_" + "".join(hide) if hide else "A"
    return CompiledOutagePolicy(name, v_grid, [t for t in tabs])


class _Arrival:
    """Wrap a compiled policy to count queries off the arrival state."""

    def __init__(self, pol, H):
        self.p = pol
        self.H = H
        self.bad = 0
        self.total = 0

    def __call__(self, i, l, r, v, pi):
        self.total += 1
        if not (i == 0 and l == self.H):
            self.bad += 1
        return self.p(i, l, r, v, pi)


def run_cell(flow_name, seed, n_tune, n_eval, results, results_grid):
    kind, env, rho = envs_for("mid")["E-outage"]
    flow = make_flows()[flow_name]
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=PPE)
    eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=PPE)
    env_hat = _env_hat(env, tune_d)
    _, _, ex = window_AD(env, survival(env))
    ex_pos = ex > 0.0

    from .outage import compile_outage

    A = compile_outage(env_hat, "A")

    # acceptance 0: A is queried only at the arrival state (i=0, l=H)
    probe = _Arrival(A, env.H)
    replay_outage(env, eval_d, probe, ex_pos)
    arrival_violations = int(probe.bad)

    episodes = np.arange(len(eval_d)) // PPE
    counts = np.bincount(episodes).astype(float)
    base = json.load(open(Path(results) / f"duel_E-outage_{flow_name}_mid_s{seed}.json"))
    a_base = np.asarray(canon_keys(base["payload"]["policies"])["A"]["block_sums"], dtype=float)
    gridf = json.load(open(Path(results_grid) / f"b4_gridN_E-outage_{flow_name}_mid_s{seed}.json"))
    b3a, b3b = gridf["payload"]["b4"]["a"], gridf["payload"]["b4"]["b"]

    block = {"A": _block_sums(replay_outage(env, eval_d, A, ex_pos), episodes)}
    for hide in (("r",), ("v",), ("r", "v")):
        pol = rederive(env_hat, A.v_grid, PI_GRID_OUTAGE, eval_d.v, hide)
        block["A_hide_" + "".join(hide)] = _block_sums(
            replay_outage(env, eval_d, pol, ex_pos), episodes
        )
    block["B3"] = _block_sums(
        replay_outage(env, eval_d, OB4(0, b3a, b3b, env.H, env.N), ex_pos), episodes
    )

    a_repro_gap = float(np.abs(block["A"] - a_base).max())
    payload = dict(
        cell=f"E-outage x {flow_name}",
        block_sums=block,
        block_counts=counts,
        mean_exposure=float(np.mean(eval_d.v)),
        a2_base_repro_gap=a_repro_gap,
        arrival_violations=arrival_violations,
        arrival_total=int(probe.total),
        b3_params=[b3a, b3b],
        stationary_outage=float(env.stationary_outage),
    )
    return payload, env


def _resolved(flow_name, env):
    env_d = dict(
        kind="outage",
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
        env_name="E-outage",
        cell=f"E-outage x {flow_name}",
        flow=flow_name,
        cw_key="mid",
        cw_per_s=CW_PER_S["mid"],
        env=env_d,
        hides=["r", "v", "rv"],
        b3_grid_n=161,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--results-grid", default="results")
    ap.add_argument("--out", default="results_t1")
    args = ap.parse_args(argv)
    require_clean_tree()

    payload, env = run_cell(
        args.flow, args.seed, args.n_tune, args.n_eval, args.results, args.results_grid
    )
    if payload["arrival_violations"] != 0:
        raise SystemExit(
            f"acceptance 0 failed: {payload['arrival_violations']} "
            f"of {payload['arrival_total']} policy calls off the "
            f"arrival state (i=0,l=H); section 7.5 narrative wrong"
        )
    if payload["a2_base_repro_gap"] > 1e-9:
        raise SystemExit(f"A reproduction failed: gap {payload['a2_base_repro_gap']:.3e}")

    obj = envelope(
        "voi",
        f"E-outage x {args.flow}",
        args.seed,
        args.n_eval,
        args.n_tune,
        _resolved(args.flow, env),
        payload,
    )
    tag = f"E-outage_{args.flow}_mid_s{args.seed}"
    path = str(Path(args.out) / f"t1_{tag}.json")
    write_once(path, obj)
    b, cnt = payload["block_sums"], np.sum(payload["block_counts"])
    print(
        json.dumps(
            dict(
                cell=payload["cell"],
                code=git_rev()[:7],
                arrival_violations=payload["arrival_violations"],
                a_repro_gap=payload["a2_base_repro_gap"],
                A_minus_B3=round(float((b["A"] - b["B3"]).sum() / cnt), 6),
                loss={
                    k: round(float((b["A"] - b[k]).sum() / cnt), 6)
                    for k in ("A_hide_r", "A_hide_v", "A_hide_rv")
                },
            ),
            indent=1,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
