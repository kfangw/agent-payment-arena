"""Theoretical verify-band closing on the delta axis (R2, spec 3.3.3).

The verify band on the (v) line is [pi_gv, pi_rv] from core.verify_edges;
it closes when pi_rv drops to pi_gv.  As delta raises the waiting-cost
rate the band shrinks, so this locates the delta at which verify stops
paying and cross-checks it against the asking channel measured by
duel.rshift (asking = A - A\\V goes to zero once the band is empty).
"""
from __future__ import annotations

import numpy as np

from duel.core import verify_edges
from duel.flows import make_flows
from duel.gate import envs_for
from duel.simulate import draw_batch

DELTAS = [-0.25, 0.0, 0.05, 0.07, 0.09, 0.15, 0.25, 0.5, 1.0, 2.0]


def cw_of(env, level):
    d_bar = abs(float(np.mean(env.f)) - env.cw)
    return env.cw + level * d_bar, d_bar


def band_open_fraction(env, rho, v, level):
    """Share of payments whose verify band is non-empty at this delta."""
    cw, _ = cw_of(env, level)
    p = env.params(rho=rho)
    p = dict(p, cw=cw)
    pgv, prv = verify_edges(v, p)
    return float((prv > pgv).mean()), cw


def closing_delta(env, rho, v_ref):
    """Delta at which pi_rv = pi_gv for a single reference exposure."""
    d_bar = abs(float(np.mean(env.f)) - env.cw)
    lo, hi = -0.5, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cw = env.cw + mid * d_bar
        p = dict(env.params(rho=rho), cw=cw)
        pgv, prv = verify_edges(np.array([v_ref]), p)
        if prv[0] > pgv[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    kind, env, rho = envs_for("mid")["E-slow"]
    flow = make_flows()["F2"]
    d = draw_batch(env, flow, 566_340, np.random.default_rng(2004 + 0))
    # rshift draws tune first (100k) then eval; reproduce eval for the v dist
    rng = np.random.default_rng(2004)
    _ = draw_batch(env, flow, 100_000, rng)
    ev = draw_batch(env, flow, 566_340, rng)
    v = ev.v
    q = 1 - (1 - rho) ** env.tau
    mu = q / rho
    print(f"E-slow x F2: q={q:.4f} rho={rho:.6f} mu=q/rho={mu:.2f} "
          f"m={env.m} C={env.C} mean_v={v.mean():.3f}")
    print(f"cw0={env.cw:.6f}  d_bar={cw_of(env,1.0)[1]:.6f}")
    for vq, lab in ((np.quantile(v, 0.1), "p10"), (np.median(v), "median"),
                    (v.mean(), "mean"), (np.quantile(v, 0.9), "p90")):
        dc = closing_delta(env, rho, float(vq))
        print(f"  verify closes at delta={dc:+.4f}  (v={vq:8.3f} {lab})")
    print("  fraction of payments with verify band OPEN, by delta:")
    for lv in DELTAS:
        frac, cw = band_open_fraction(env, rho, v, lv)
        print(f"    delta={lv:+.2f}  cw={cw:.6f}  open={frac*100:6.2f}%")


if __name__ == "__main__":
    main()
