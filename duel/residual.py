"""R4: E-outage residual diagnosis.

The surviving empirical margin is E-outage A - B3 at the refined grid, and
k* = 0 there, so B4 = B3: the margin is the amount the manuscript calls the
undiagnosed residual.  B3 reads only suspicion pi; the compiled family-A
policy reads (i, l, r, v, pi).  This module measures what the four extra
coordinates buy.

R4a (coordinate ablation).  The compiled A table is marginalized over one
coordinate at a time: at a hidden coordinate the policy emits the majority
action over that coordinate, weighted by how often the tuning split visits
each value there.  The collapsed table is the same shape, so it replays at
full speed.  A-ilrv reads only pi (B3's information set); A - A-ilrv is what
the extra coordinates earn and A-ilrv - B3 is what a different compilation
of the same information earns, and their sum is the residual (an identity).

R4b (disagreement map).  Replay records, per payment, the arrival state and
binned exposure/suspicion, the arrival actions of A and B3, and each policy's
realized payoff, aggregated by cell so the residual can be attributed to a
region of the state space rather than stored per payment.

    python -m duel.residual --env E-outage --flow F2 --seed 8 --out results_r4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b4 import OB4, _block_sums
from .core import rho_hat_from_q
from .gate import CW_PER_S, envs_for
from .flows import make_flows
from .naming import canon_keys
from .outage import (CompiledOutagePolicy, OutageEnv, PI_GRID_OUTAGE,
                     compile_outage, draw_outage_batch, replay_outage,
                     survival, window_AD)
from .report import envelope, write_once

PPE = 50
# table axes of the stacked compiled labels: (iv, i, l, r, ip)
AX = dict(v=0, i=1, l=2, r=3, pi=4)
ABLATIONS = {
    "A": (),
    "A-r": (AX["r"],),
    "A-l": (AX["l"],),
    "A-i": (AX["i"],),
    "A-v": (AX["v"],),
    "A-ilr": (AX["i"], AX["l"], AX["r"]),
    "A-ilrv": (AX["v"], AX["i"], AX["l"], AX["r"]),
}


# ----------------------------------------------------------- A2 machinery
def _compile_a2(env: OutageEnv, tune_d):
    """Reproduce run.py's outage A2: calibrate rho on the tuning split, then
    compile on the geometric window at that rate.  Returns the compiled
    policy, the stacked label array, and the exercise sign map."""
    q_hat = float((tune_d.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, env.tau)
    env_hat = OutageEnv(f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw,
                        tau=env.tau, H=env.H, rho=rho_hat, p01=env.p01,
                        p10=env.p10, tick_seconds=env.tick_seconds)
    a2 = compile_outage(env_hat, "A2")
    _, _, ex = window_AD(env, survival(env))
    ex_pos = ex > 0.0
    arr = np.stack(a2.tables).astype(np.int8)     # (iv, i, l, r, ip)
    return a2, arr, ex_pos, dict(q_hat=q_hat, rho_hat=rho_hat)


class _Recorder:
    """Wrap the compiled A to count (iv, i, l, r, ip) queries into N."""
    def __init__(self, a2: CompiledOutagePolicy, N):
        self.tabs = a2.tables
        self.logv = np.log(a2.v_grid)
        self.npi = len(PI_GRID_OUTAGE)
        self.N = N

    def __call__(self, i, l, r, v, pi):
        iv = int(np.argmin(np.abs(self.logv - np.log(max(v, 1e-12)))))
        ip = int(round(pi * (self.npi - 1)))
        self.N[iv, i, l, r, ip] += 1
        return int(self.tabs[iv][i, l, r, ip])


def _visit_counts(env, tune_d, a2, ex_pos, shape):
    N = np.zeros(shape, dtype=np.int64)
    replay_outage(env, tune_d, _Recorder(a2, N), ex_pos)
    return N


def _marginalize(arr, N, hidden):
    """Collapse `arr` over the hidden axes by visit-weighted majority action
    (uniform majority where the tuning split never visited the cell), then
    broadcast back to the full shape."""
    if not hidden:
        return arr
    wsum = np.stack([(N * (arr == a)).sum(axis=hidden, keepdims=True)
                     for a in range(4)], axis=0)
    marg = wsum.argmax(axis=0)
    seen = wsum.sum(axis=0) > 0
    if not bool(seen.all()):
        usum = np.stack([(arr == a).sum(axis=hidden, keepdims=True)
                         for a in range(4)], axis=0)
        marg = np.where(seen, marg, usum.argmax(axis=0))
    return np.broadcast_to(marg, arr.shape).astype(np.int8).copy()


def _policy_from_arr(name, v_grid, arr):
    return CompiledOutagePolicy(name, v_grid, [arr[iv] for iv in range(len(v_grid))])


# --------------------------------------------------------------- binning
def _v_bins(v, edges):
    return np.clip(np.digitize(np.log(np.maximum(v, 1e-12)), edges), 0,
                   len(edges)).astype(np.int64)


def _nearest_iv(v, v_grid):
    """Nearest v-grid index in log space, matching CompiledOutagePolicy,
    without materializing an n-by-grid array."""
    lv = np.log(np.maximum(v, 1e-12))
    lg = np.log(v_grid)
    iv = np.clip(np.searchsorted(lg, lv), 0, len(lg) - 1)
    lo = np.maximum(iv - 1, 0)
    pick_lo = np.abs(lg[lo] - lv) < np.abs(lg[iv] - lv)
    return np.where(pick_lo, lo, iv).astype(np.int64)


def _arrival_actions_A(arr, v, pi0, r0, v_grid, H):
    """A's action at the arrival state (i=0, l=H) per payment, vectorized."""
    iv = _nearest_iv(v, v_grid)
    ip = np.round(pi0 * (len(PI_GRID_OUTAGE) - 1)).astype(np.int64)
    return arr[iv, 0, H, r0, ip]


def _arrival_actions_B3(pi0, a, b):
    """B3's arrival action: grant<a, reject>b, else verify."""
    out = np.full(len(pi0), 2, dtype=np.int8)      # VERIFY
    out[pi0 < a] = 0                               # GRANT
    out[pi0 > b] = 1                               # REJECT
    return out


# ----------------------------------------------------------------- driver
def run_cell(env_name, flow_name, seed, n_tune, n_eval, results, results_grid):
    kind, env, rho = envs_for("mid")[env_name]
    flow = make_flows()[flow_name]
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, flow, n_tune, rng, payments_per_episode=PPE)
    eval_d = draw_outage_batch(env, flow, n_eval, rng, payments_per_episode=PPE)

    a2, arr, ex_pos, calib = _compile_a2(env, tune_d)
    N = _visit_counts(env, tune_d, a2, ex_pos, arr.shape)

    episodes = np.arange(len(eval_d)) // PPE
    counts = np.bincount(episodes).astype(float)

    # base A2 block sums (grid-independent) and n=161 B3 from S1's b4_gridN
    base = json.load(open(Path(results) /
                          f"duel_{env_name}_{flow_name}_mid_s{seed}.json"))
    a2_base = np.asarray(canon_keys(base["payload"]["policies"])["A"]["block_sums"],
                         dtype=float)
    gridf = json.load(open(Path(results_grid) /
                           f"b4_gridN_{env_name}_{flow_name}_mid_s{seed}.json"))
    b3a, b3b = gridf["payload"]["b4"]["a"], gridf["payload"]["b4"]["b"]

    # R4a: replay every ablation on eval
    pays, block = {}, {}
    for name, hidden in ABLATIONS.items():
        tab = _marginalize(arr, N, hidden)
        pol = _policy_from_arr(name, a2.v_grid, tab)
        p = replay_outage(env, eval_d, pol, ex_pos)
        pays[name] = p
        block[name] = _block_sums(p, episodes)
    b3 = OB4(0, b3a, b3b, env.H, env.N)
    pay_b3 = replay_outage(env, eval_d, b3, ex_pos)
    block["B3"] = _block_sums(pay_b3, episodes)

    a_repro_gap = float(np.abs(block["A"] - a2_base).max())

    # R4b: arrival-state disagreement map, aggregated by cell
    v_edges = np.quantile(np.log(np.maximum(eval_d.v, 1e-12)),
                          np.linspace(0, 1, 9)[1:-1])
    pi_edges = np.quantile(eval_d.pi0, np.linspace(0, 1, 21)[1:-1])
    r0 = eval_d.paths[:, 0].astype(np.int64)
    vb = np.clip(np.digitize(np.log(np.maximum(eval_d.v, 1e-12)), v_edges),
                 0, 7).astype(np.int64)
    pb = np.clip(np.digitize(eval_d.pi0, pi_edges), 0, 19).astype(np.int64)
    aA = _arrival_actions_A(arr, eval_d.v, eval_d.pi0, r0, a2.v_grid, env.H)
    aB = _arrival_actions_B3(eval_d.pi0, b3a, b3b)
    diff = pays["A"] - pay_b3
    # composite cell key: r0, vb, pb, aA, aB
    key = (((r0 * 8 + vb) * 20 + pb) * 4 + aA.astype(np.int64)) * 4 + aB.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    cell_cnt = np.bincount(inv)
    cell_diff = np.bincount(inv, weights=diff)
    cell_pA = np.bincount(inv, weights=pays["A"])
    cell_pB = np.bincount(inv, weights=pay_b3)
    # decode keys
    kk = uniq.copy()
    d_aB = kk % 4; kk //= 4
    d_aA = kk % 4; kk //= 4
    d_pb = kk % 20; kk //= 20
    d_vb = kk % 8; kk //= 8
    d_r0 = kk
    cells = [dict(r=int(d_r0[j]), v_bin=int(d_vb[j]), pi_bin=int(d_pb[j]),
                  a_A=int(d_aA[j]), a_B3=int(d_aB[j]), n=int(cell_cnt[j]),
                  sum_diff=float(cell_diff[j]), sum_A=float(cell_pA[j]),
                  sum_B3=float(cell_pB[j])) for j in range(len(uniq))]
    disagree = aA != aB
    diag_gap = float(abs(cell_diff.sum() - diff.sum()))

    payload = dict(
        cell=f"{env_name} x {flow_name}", ablations=list(ABLATIONS),
        block_sums={k: v for k, v in block.items()},
        block_counts=counts, mean_exposure=float(np.mean(eval_d.v)),
        a2_base_repro_gap=a_repro_gap, b3_params=[b3a, b3b], calib=calib,
        r4b=dict(cells=cells, diag_gap=diag_gap,
                 disagree_mass=float(disagree.mean()),
                 disagree_resid=float(diff[disagree].sum()),
                 total_resid=float(diff.sum()),
                 v_edges=[float(x) for x in v_edges],
                 pi_edges=[float(x) for x in pi_edges]),
    )
    return payload


def _resolved(env_name, flow_name, env, ns=161):
    env_d = dict(kind="outage", f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw,
                 tau=env.tau, H=env.H, rho=env.rho, p01=env.p01, p10=env.p10,
                 tick_seconds=env.tick_seconds)
    return dict(env_name=env_name, cell=f"{env_name}x{flow_name}",
                flow=flow_name, cw_key="mid", cw_per_s=CW_PER_S["mid"],
                env=env_d, ablations=list(ABLATIONS), b3_grid_n=ns)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="E-outage",
                    choices=["E-fast", "E-outage", "E-slow"])
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--results-grid", default="results")
    ap.add_argument("--out", default="results_r4")
    args = ap.parse_args(argv)

    kind, env, rho = envs_for("mid")[args.env]
    payload = run_cell(args.env, args.flow, args.seed, args.n_tune,
                       args.n_eval, args.results, args.results_grid)
    if payload["a2_base_repro_gap"] > 1e-9:
        raise SystemExit(f"A reproduction failed: gap "
                         f"{payload['a2_base_repro_gap']:.3e} (lookup path changed)")
    if payload["r4b"]["diag_gap"] > 1e-6:
        raise SystemExit(f"R4b instrumentation gap {payload['r4b']['diag_gap']:.3e} "
                         f"(cell sums disagree with replay)")

    obj = envelope("residual", f"{args.env} x {args.flow}", args.seed,
                   args.n_eval, args.n_tune, _resolved(args.env, args.flow, env),
                   payload)
    tag = f"{args.env}_{args.flow}_mid_s{args.seed}"
    path = str(Path(args.out) / f"r4_{tag}.json")
    write_once(path, obj)
    b = payload["block_sums"]
    cnt = payload["block_counts"]
    tot = np.sum(cnt)
    amb3 = (b["A"] - b["B3"]).sum() / tot
    print(json.dumps(dict(
        cell=payload["cell"], a_repro_gap=payload["a2_base_repro_gap"],
        diag_gap=payload["r4b"]["diag_gap"],
        A_minus_B3=round(amb3, 6),
        losses={k: round(float((b["A"] - b[k]).sum() / tot), 6)
                for k in ABLATIONS if k != "A"},
        disagree_mass=round(payload["r4b"]["disagree_mass"], 4)), indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
