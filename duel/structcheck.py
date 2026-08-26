"""U1: structural acceptance check of the augmented-state-space table.

The surviving margin comes from compile_outage, which solves the full
(i, l, r) state space by backward induction.  Algorithm 1's four checks and
Theorems 1-2 are established on the single-stage chain, not on (i, l, r).
This module reads the compiled labels along the suspicion axis at every
state and tests, per state, the conclusions of Theorem 1: each action's
optimal region is an interval, grant is a prefix reaching pi=0, reject is a
non-empty suffix reaching pi=1, verify is an interval off both endpoints,
and a direct grant-reject boundary sits at the indifference point
pi_hat(state) = (1 + m - 1/sigma) / (m + h).  This is an observation on the
compiled tables, not a proof for the augmented state space.

    python -m duel.structcheck --env E-outage --flow F2 --seed 8 --out results_u1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, rho_hat_from_q, sigma_list
from .flows import make_flows
from .gate import CW_PER_S, envs_for
from .gitcheck import require_clean_tree
from .outage import (OutageEnv, PI_GRID_OUTAGE, compile_outage,
                     draw_outage_batch, survival)
from .policies import PI_GRID as PI_GRID_CHAIN, compile_A
from .report import envelope, write_once
from .simulate import draw_batch

TOL_STEPS = 3          # grant-reject boundary tolerance, in pi-grid steps


def _interval(idx):
    """True if the index set is empty or contiguous."""
    return len(idx) == 0 or (idx[-1] - idx[0] + 1 == len(idx))


def _pi_hat(sigma, m, h):
    if sigma <= 0:
        return -np.inf
    return (1.0 + m - 1.0 / sigma) / (m + h)


def check_pi(lab, pi_grid, sigma, m, h):
    """Test Theorem 1's conclusions along the suspicion axis at one state.
    Returns a dict of five booleans and the failing detail."""
    npi = len(lab)
    g = np.where(lab == GRANT)[0]
    r = np.where(lab == REJECT)[0]
    ver = np.where(lab == VERIFY)[0]
    w = np.where(lab == WAIT)[0]
    c1 = len(g) == 0 or (_interval(g) and g[0] == 0)
    c2 = len(r) > 0 and _interval(r) and r[-1] == npi - 1
    c3 = len(ver) == 0 or (_interval(ver) and ver[0] > 0 and ver[-1] < npi - 1)
    c4 = all(_interval(x) for x in (g, r, ver, w))
    # c5: direct grant->reject adjacency should sit at pi_hat
    c5 = True
    boundary = None
    dev = None
    adj = np.where((lab[:-1] == GRANT) & (lab[1:] == REJECT))[0]
    c5_fired = len(adj) > 0
    if c5_fired:
        j = int(adj[0])
        boundary = 0.5 * (pi_grid[j] + pi_grid[j + 1])
        ph = _pi_hat(sigma, m, h)
        tol = TOL_STEPS / (npi - 1)
        dev = abs(boundary - ph)
        c5 = 0.0 <= ph <= 1.0 and dev <= tol
    present = [bool(len(g)), bool(len(r)), bool(len(ver)), bool(len(w))]
    single = (len(g) == npi) or (len(r) == npi) or (len(ver) == npi) or (len(w) == npi)
    return dict(c1=bool(c1), c2=bool(c2), c3=bool(c3), c4=bool(c4), c5=bool(c5),
                boundary=boundary, c5_fired=c5_fired, dev=dev, present=present,
                single=single)


def _run_checks(state_iter, checks):
    """Aggregate checks over a state iterator yielding (key, lab, sigma).
    Also records how strong the check was: single-action rows (where the
    interval clauses pass trivially), rows where the grant-reject boundary
    actually fired, the worst boundary deviation, and which of the four
    actions appear anywhere in the table."""
    names = ["c1", "c2", "c3", "c4", "c5"]
    n_states = 0
    viol = {c: [] for c in names}
    n_single = 0
    n_c5_fired = 0
    max_dev = 0.0
    presence = [False, False, False, False]
    for key, lab, sigma in state_iter:
        n_states += 1
        res = checks(lab, sigma)
        for c in names:
            if not res[c] and len(viol[c]) < 10:
                viol[c].append(dict(state=key, sigma=round(float(sigma), 6),
                                    boundary=res["boundary"]))
        n_single += int(res["single"])
        if res["c5_fired"]:
            n_c5_fired += 1
            if res["dev"] is not None:
                max_dev = max(max_dev, res["dev"])
        presence = [p or q for p, q in zip(presence, res["present"])]
    summary = {c: dict(violations=len(v), examples=v) for c, v in viol.items()}
    density = dict(
        n_single_action=n_single, n_c5_fired=n_c5_fired,
        max_boundary_dev=max_dev, label_presence=dict(
            grant=presence[0], reject=presence[1],
            verify=presence[2], wait=presence[3]))
    return n_states, summary, density


def check_outage(env, flow_name, seed, n_tune):
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(env, make_flows()[flow_name], n_tune, rng,
                               payments_per_episode=50)
    q = float((tune_d.t_ans <= env.tau).mean())
    env_hat = OutageEnv(f=env.f, m=env.m, h=env.h, C=env.C, cw=env.cw,
                        tau=env.tau, H=env.H, rho=rho_hat_from_q(q, env.tau),
                        p01=env.p01, p10=env.p10, tick_seconds=env.tick_seconds)
    pol = compile_outage(env_hat, "A")
    sig = survival(env_hat)              # sigma(i, l, r)
    N, H = env.N, env.H

    def it():
        for iv in range(len(pol.v_grid)):
            tab = pol.tables[iv]         # (i, l, r, pi)
            for i in range(N + 2):
                for l in range(H + 1):
                    for r in (0, 1):
                        yield ((iv, i, l, r), tab[i, l, r, :], sig[i, l, r])

    n, out, density = _run_checks(
        it(), lambda lab, s: check_pi(lab, PI_GRID_OUTAGE, s, env.m, env.h))
    n_ilr = (N + 2) * (H + 1) * 2
    density["max_boundary_dev_steps"] = density["max_boundary_dev"] * (
        len(PI_GRID_OUTAGE) - 1)
    return dict(kind="outage", flow_used=flow_name, n_states_checked=n,
                n_ilr=n_ilr, n_v=len(pol.v_grid), n_pi=len(PI_GRID_OUTAGE),
                checks=out, density=density, q_hat=q, rho_hat=env_hat.rho)


def check_chain(ch, flow_name, seed, n_tune):
    rng = np.random.default_rng(seed)
    tune_d = draw_batch(ch, make_flows()[flow_name], n_tune, rng)
    q = float((tune_d.t_ans <= ch.tau).mean())
    pol = compile_A(ch, "A", rho=rho_hat_from_q(q, ch.tau))
    sig = sigma_list(ch.f)               # sigma(stage)
    n_stage = pol.tables.shape[1]

    def it():
        for iv in range(pol.tables.shape[0]):
            for st in range(n_stage):
                yield ((iv, st), pol.tables[iv, st, :], sig[st])

    n, out, density = _run_checks(
        it(), lambda lab, s: check_pi(lab, PI_GRID_CHAIN, s, ch.m, ch.h))
    density["max_boundary_dev_steps"] = density["max_boundary_dev"] * (
        len(PI_GRID_CHAIN) - 1)
    return dict(kind="chain", flow_used=flow_name, n_states_checked=n,
                n_stage=n_stage, n_v=pol.tables.shape[0],
                n_pi=len(PI_GRID_CHAIN), checks=out, density=density, q_hat=q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True,
                    choices=["E-outage", "E-fast", "E-slow"])
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--out", default="results_u1")
    args = ap.parse_args(argv)
    require_clean_tree()

    kind, env, rho = envs_for("mid")[args.env]
    payload = (check_outage(env, args.flow, args.seed, args.n_tune)
               if kind == "outage"
               else check_chain(env, args.flow, args.seed, args.n_tune))
    payload["cell"] = f"{args.env} x {args.flow}"
    assert payload["flow_used"] == args.flow
    total_viol = sum(payload["checks"][c]["violations"] for c in payload["checks"])
    payload["total_violations"] = total_viol

    resolved = dict(env_name=args.env, cell=f"{args.env}x{args.flow}",
                    flow=args.flow, cw_per_s=CW_PER_S["mid"], tol_steps=TOL_STEPS)
    obj = envelope("structcheck", f"{args.env} x {args.flow}", args.seed,
                   0, args.n_tune, resolved, payload)
    path = str(Path(args.out) / f"u1_{args.env}_{args.flow}_s{args.seed}.json")
    write_once(path, obj)
    print(json.dumps(dict(cell=payload["cell"], flow_used=payload["flow_used"],
                          kind=payload["kind"], q_hat=round(payload["q_hat"], 5),
                          states=payload["n_states_checked"],
                          violations={c: payload["checks"][c]["violations"]
                                      for c in payload["checks"]},
                          total=total_viol, density=payload["density"]),
                     indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
