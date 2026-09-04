"""Harness validation: the simulator must reproduce the exact model
expectations, policy by policy, at Monte Carlo precision.

Run:  python -m arena.experiments.settlement.validate_harness
"""

from __future__ import annotations

import numpy as np

from .core import DEFAULT, action_values, derived, sigma_list
from .flows import Flow, MixD, LogNormalV
from .policies import compile_A, default_grids, make_family_C, tune, B1, PI_GRID
from .simulate import Channel, draw_batch, replay


def geometric_pmf(rho, tau):
    return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])


def exact_policy_values(ch: Channel, d, rho):
    """Exact expected payoff per payment for the four fixed rules and the
    optimum, evaluated at the payment's own (v, p_true)."""
    out = {k: np.zeros(len(d)) for k in ("C1", "C2", "C3", "C4", "OPT")}
    p = ch.params(rho=rho)
    for k in range(len(d)):
        v, pt = float(d.v[k]), float(d.p_true[k])
        o = action_values(ch.f, v, np.array([pt]), p)
        N = ch.N
        aw = o["G"][N + 1].copy()
        for i in range(N, -1, -1):
            aw = -ch.cw * v + (1 - ch.f[i]) * aw
        out["C1"][k] = o["G"][0][0]
        out["C2"][k] = 0.0
        out["C3"][k] = o["W"][0][0]
        out["C4"][k] = aw[0]
        out["OPT"][k] = o["V"][0][0]
    return out


def main():
    rng = np.random.default_rng(20260821)
    P = DEFAULT
    der = derived(P)
    assert abs(der["q"] - 0.96875) < 1e-12
    assert abs(der["pihat"] - 0.3 / 1.3) < 1e-12
    print(
        f"derived: q={der['q']:.5f} vstar={der['vstar']:.4f} "
        f"rhostar={der['rhostar']:.4f} floor={der['floor']:.4f}  [OK]"
    )

    pmf = geometric_pmf(P["rho"], P["tau"])
    for rail, f in (("fast", [0.005] * 3), ("slow", [0.06, 0.03, 0.015])):
        ch = Channel(
            f=np.array(f), m=P["m"], h=P["h"], C=P["C"], cw=P["cw"], tau=P["tau"], pmf_h=pmf
        )
        flow = Flow("F1cal", MixD(0.05), LogNormalV(1.0, 0.7, 0.05, 40.0))
        d = draw_batch(ch, flow, 400_000, rng)
        sig = sigma_list(ch.f)
        ex_unit = np.maximum(sig * (1 + ch.m) - 1.0, 0.0)

        pols = make_family_C(ch)
        exact = exact_policy_values(ch, d, P["rho"])
        print(f"\n[{rail}] n={len(d)}  (realized vs exact, per payment)")
        for name, pol in pols.items():
            r = replay(ch, d, pol, ex_unit)
            e = exact[name].mean()
            se = r.std(ddof=1) / np.sqrt(len(r))
            z = (r.mean() - e) / se if se > 0 else 0.0
            print(f"  {name}: realized {r.mean():+.5f}  exact {e:+.5f}  z={z:+.2f}")
            assert abs(z) < 4.0, (rail, name)

        a1 = compile_A(ch, "A1", pmf=pmf, n_v=41, v_lo=0.05, v_hi=40.0)
        r = replay(ch, d, a1, ex_unit)
        e = exact["OPT"].mean()
        se = r.std(ddof=1) / np.sqrt(len(r))
        print(
            f"  A1: realized {r.mean():+.5f}  exact-opt {e:+.5f}  "
            f"z={(r.mean() - e) / se:+.2f}  (small - allowed: grid gap)"
        )

        grids = default_grids(v_hi=40.0, n=25)
        tun = draw_batch(ch, flow, 50_000, rng)
        best, b1, _ = tune(lambda t: B1(t, ch.h), grids["B1"], ch, tun, ex_unit)
        rb = replay(ch, d, b1, ex_unit)
        print(
            f"  B1 tuned theta={best:.4f}: realized {rb.mean():+.5f}  "
            f"(A1 - B1 = {r.mean() - rb.mean():+.5f})"
        )


if __name__ == "__main__":
    main()
