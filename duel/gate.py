"""Phase-map gate: compute, per cell, the map of initial-state
optimal actions on the (v, pi) plane, and the flow mass each action
receives; then check the coverage condition of the run spec.

The pivotal channel constant is the per-tick waiting cost rate c_w:
within the measured range (sustained marginal delay cost between
0.001%/s and 0.1%/s) it moves cells between verify-active and
verify-vanishing regimes.  The gate therefore sweeps declared c_w
candidates and reports the resulting masses, so the per-cell channel
constants are declared once, with sources, before anything is run.

Run:  python -m duel.gate            (writes gate_report.json)
"""

from __future__ import annotations

import json

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, best_action
from .flows import make_flows
from .outage import OutageEnv, value_labels as outage_labels
from .simulate import Channel

PI_GRID = np.linspace(0.0, 1.0, 501)
V_GRID = np.geomspace(0.5, 2000.0, 61)

# c_w candidates: sustained marginal delay-cost rate, declared per second
CW_PER_S = dict(high=1e-3, mid=1e-4, low=1e-5)  # 0.1%/s, 0.01%/s, 0.001%/s

# answer channel (push/mobile draft): q = 0.75 within tau = 10 min
TAU_S = 600.0
Q_CHANNEL = 0.75


def rho_for(tick_s):
    n_ticks = int(round(TAU_S / tick_s))
    return n_ticks, 1.0 - (1.0 - Q_CHANNEL) ** (1.0 / n_ticks)


def envs_for(cw_key, m=0.35, h=1.0, C=0.5):
    """The three environments at one c_w declaration."""
    out = {}
    # E-fast: Base, tick 2 s, chain truncated at the unsafe boundary
    tick = 2.0
    tau, rho = rho_for(tick)
    cw = CW_PER_S[cw_key] * tick
    f_fast = np.array([0.005] + [1e-6] * 39)
    out["E-fast"] = (
        "chain",
        Channel(f=f_fast, m=m, h=h, C=C, cw=cw, tau=tau, pmf_h=_geom(rho, tau)),
        rho,
    )
    # E-slow: depth-decaying hazard chain (fallback shape)
    f_slow = 0.06 * 0.5 ** np.arange(8)
    out["E-slow"] = (
        "chain",
        Channel(f=f_slow, m=m, h=h, C=C, cw=cw, tau=tau, pmf_h=_geom(rho, tau)),
        rho,
    )
    # E-slow-deep: the same decaying-hazard chain carried to a deeper
    # settlement horizon (12 stages) so the watch value that peaks where the
    # hazard crosses c_w falls inside the chain rather than at its end.
    f_slow_deep = 0.06 * 0.5 ** np.arange(12)
    out["E-slow-deep"] = (
        "chain",
        Channel(f=f_slow_deep, m=m, h=h, C=C, cw=cw, tau=tau, pmf_h=_geom(rho, tau)),
        rho,
    )
    # E-outage: coarse tick 60 s
    tick = 60.0
    tau_o, rho_o = rho_for(tick)
    cw_o = CW_PER_S[cw_key] * tick
    out["E-outage"] = (
        "outage",
        OutageEnv(
            f=np.array([0.005] + [3e-5] * 16),
            m=m,
            h=h,
            C=C,
            cw=cw_o,
            tau=tau_o,
            H=30,
            rho=rho_o,
            p01=2.0 / (30 * 24 * 60),
            p10=1.0 / 60.0,
        ),
        rho_o,
    )
    return out


def _geom(rho, tau):
    return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])


def maps_chain(ch: Channel, rho):
    """Per (v_bin, pi_bin): initial-stage action, and whether verify /
    wait appear in the optimal policy at ANY stage.  Activation means
    the action appears somewhere, not only at the arrival tick."""
    nv, npi = len(V_GRID), len(PI_GRID)
    init = np.zeros((nv, npi), dtype=np.int8)
    v_any = np.zeros((nv, npi), dtype=bool)
    w_any = np.zeros((nv, npi), dtype=bool)
    p = ch.params(rho=rho)
    for iv, v in enumerate(V_GRID):
        lab = best_action(ch.f, v, PI_GRID, p)[0]
        init[iv] = lab[0]
        v_any[iv] = (lab == VERIFY).any(axis=0)
        w_any[iv] = (lab == WAIT).any(axis=0)
    return init, v_any, w_any


def maps_outage(env: OutageEnv):
    """Initial action per regime, plus any-state verify/wait presence."""
    nv, npi = len(V_GRID), len(PI_GRID)
    init = np.zeros((nv, 2, npi), dtype=np.int8)
    v_any = np.zeros((nv, npi), dtype=bool)
    w_any = np.zeros((nv, npi), dtype=bool)
    for iv, v in enumerate(V_GRID):
        L = outage_labels(env, v, PI_GRID)[0]
        init[iv, 0] = L[0, env.H, 0]
        init[iv, 1] = L[0, env.H, 1]
        v_any[iv] = (L == VERIFY).any(axis=(0, 1, 2))
        w_any[iv] = (L == WAIT).any(axis=(0, 1, 2))
    return init, v_any, w_any


def _v_bins(v):
    iv = np.clip(np.searchsorted(V_GRID, v), 0, len(V_GRID) - 1)
    lo = np.maximum(iv - 1, 0)
    pick_lo = np.abs(np.log(V_GRID[lo]) - np.log(v)) < np.abs(np.log(V_GRID[iv]) - np.log(v))
    return np.where(pick_lo, lo, iv)


def flow_masses(init, v_any, w_any, v, pi0, p_out=None):
    """Flow-mass shares: initial action distribution, and the mass whose
    (v, pi) has verify / wait anywhere in the optimal policy."""
    iv = _v_bins(v)
    ip = np.round(pi0 * (len(PI_GRID) - 1)).astype(int)
    if init.ndim == 3 and p_out is not None:
        s0 = np.array([(init[iv, 0, ip] == a).mean() for a in range(4)])
        s1 = np.array([(init[iv, 1, ip] == a).mean() for a in range(4)])
        shares = (1 - p_out) * s0 + p_out * s1
    else:
        a0 = init[iv, ip]
        shares = np.array([(a0 == a).mean() for a in range(4)])
    return shares, float(v_any[iv, ip].mean()), float(w_any[iv, ip].mean())


ACT_NAMES = ["grant", "reject", "verify", "wait"]

# Coverage thresholds (provisional): a cell is verify-ACTIVE when at
# least 1% of its flow mass sits where verify is optimal somewhere, and
# verify-DEAD when at most 0.1% does; between the two it counts for
# neither side.
ACTIVE_MIN, DEAD_MAX = 0.01, 0.001


def run(n_flow=200_000, seed=20260822):
    rng = np.random.default_rng(seed)
    flows = make_flows()
    samples = {fn: fl.sample(n_flow, rng) for fn, fl in flows.items()}
    report = {}
    for cw_key in ("high", "mid", "low"):
        envs = envs_for(cw_key)
        cells = {}
        for en, (kind, env, rho) in envs.items():
            if kind == "chain":
                init, v_any, w_any = maps_chain(env, rho)
                p_out = None
            else:
                init, v_any, w_any = maps_outage(env)
                p_out = env.stationary_outage
            for fn in flows:
                v, _, _, pi0 = samples[fn]
                sh, mv, mw = flow_masses(init, v_any, w_any, v, pi0, p_out)
                cells[f"{en}x{fn}"] = dict(
                    {ACT_NAMES[a]: round(float(sh[a]), 5) for a in range(4)},
                    verify_any=round(mv, 5),
                    wait_any=round(mw, 5),
                )
        act = {c: m["verify_any"] >= ACTIVE_MIN for c, m in cells.items()}
        dead = {c: m["verify_any"] <= DEAD_MAX for c, m in cells.items()}
        n_active_slow = sum(act[c] for c in act if c.startswith(("E-outage", "E-slow")))
        report[cw_key] = dict(
            cw_per_s=CW_PER_S[cw_key],
            cells=cells,
            coverage=dict(
                active=[c for c, x in act.items() if x],
                dead=[c for c, x in dead.items() if x],
                active_outage_slow=n_active_slow,
                passes=(n_active_slow >= 2 and any(dead.values()) and any(act.values())),
            ),
        )
    return report


def main():
    report = run()
    with open("gate_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    for k, r in report.items():
        cov = r["coverage"]
        print(
            f"\n== cw = {r['cw_per_s']:.0e}/s ({k}) "
            f"passes={cov['passes']} active(outage/slow)={cov['active_outage_slow']}"
        )
        for c, m in r["cells"].items():
            print(
                f"  {c:16s} "
                + "  ".join(f"{a}={m[a]:.3f}" for a in ACT_NAMES)
                + f"  | any: verify={m['verify_any']:.3f} wait={m['wait_any']:.3f}"
            )
    print("\nwrote gate_report.json")


if __name__ == "__main__":
    main()
