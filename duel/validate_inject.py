"""Acceptance checks for the injection axes (spec 2.6).

Run:  python -m duel.validate_inject
"""
from __future__ import annotations

import tempfile

import numpy as np

from .core import rho_hat_from_q
from .flows import make_flows
from .gate import envs_for
from .inject import (_chain_ctx, _chain_diff, halving, run_axis)
from .run import run_chain
from .simulate import retime_answers

CELL = "E-slow x F1"


def a4_5_halving_interpolation() -> None:
    """A4-5: crossings on a straight, a step, and a non-monotone curve
    match hand computation."""
    eps = 0.01
    # straight line G = 1 - 0.3 x on [0,0.5,1,1.5,2]
    lv = [0.0, 0.5, 1.0, 1.5, 2.0]
    g = [1.0 - 0.3 * x for x in lv]
    h = halving(lv, g, eps)
    assert abs(h["point"] - 5.0 / 3.0) < 1e-9, h
    assert h["status"] == "ok" and h["monotone"]
    # step from 1 to 0.4 between x=1 and x=2
    h2 = halving([0, 1, 2, 3], [1.0, 1.0, 0.4, 0.4], eps)
    assert abs(h2["point"] - (1 + (0.5 - 1.0) / (0.4 - 1.0))) < 1e-9, h2
    # non-monotone: rises then falls through the target; first crossing wins
    h3 = halving([0, 1, 2, 3], [1.0, 1.2, 0.3, 0.5], eps)
    assert abs(h3["point"] - (1 + (0.5 - 1.2) / (0.3 - 1.2))) < 1e-9, h3
    assert not h3["monotone"]
    # G(0) below eps -> undefined
    assert halving([0, 1], [0.005, 0.004], eps)["status"] == "undefined"
    # never reaches half -> beyond_range with residual
    hb = halving([0, 1, 2], [1.0, 0.9, 0.8], eps)
    assert hb["status"] == "beyond_range" and abs(hb["residual"] - 0.8) < 1e-9
    print("A4-5 ok: straight, step, non-monotone, undefined, beyond-range")


def a4_1_identity_matches_exp1() -> None:
    """A4-1: the identity level reproduces the experiment-1 A2 - B1 mean on
    the same seed, exactly."""
    kind, env, rho = envs_for("mid")[CELL.split(" x ")[0]]
    flow = make_flows()["F1"]
    seed, n_tune, n_eval = 5, 15_000, 30_000
    out, _, _, _ = run_chain(env, rho, flow, n_tune, n_eval, seed)
    exp1 = float((out["A2"] - out["B1"]).mean())
    ctx = _chain_ctx(env, rho, flow, seed, n_tune, n_eval)
    for axis, ident in (("kappa", 1.0), ("lambda", 1.0), ("delta", 0.0)):
        got = float(_chain_diff(ctx, axis, ident).mean())
        assert abs(got - exp1) < 1e-9, f"{axis}: {got} != {exp1}"
    print(f"A4-1 ok: identity == exp1 mean {exp1:.6f} for kappa/lambda/delta")


def a4_3_noise_direction() -> None:
    """A4-3: growing sigma lowers A2's advantage (B1 does not move)."""
    kind, env, rho = envs_for("mid")[CELL.split(" x ")[0]]
    flow = make_flows()["F1"]
    ctx = _chain_ctx(env, rho, flow, 7, 15_000, 40_000)
    ctx["seed"] = 7
    a0 = float(_chain_diff(ctx, "noise", 0.0).mean())
    a_hi = float(_chain_diff(ctx, "noise", 0.7).mean())
    assert a_hi < a0 + 1e-6, f"noise did not degrade A2: {a0} -> {a_hi}"
    print(f"A4-3 ok: advantage {a0:.5f} (sigma 0) -> {a_hi:.5f} (sigma 0.7)")


def a4_4_lambda_identification() -> None:
    """A4-4: the misuse within-deadline rate q_m moves monotonically with
    lambda, so |q_m - q_h| grows as lambda leaves 1."""
    kind, env, rho = envs_for("mid")[CELL.split(" x ")[0]]
    flow = make_flows()["F1"]
    ctx = _chain_ctx(env, rho, flow, 9, 1, 200_000)
    d, u = ctx["eval_d"], ctx["u_eval"]
    mis = d.theta == 1
    q_h = float((d.t_ans[d.theta == 0] <= env.tau).mean())
    qs = []
    for lam in [0.5, 2 / 3, 1.0, 1.5, 2.0]:
        pmf_m = np.array([min(rho * lam, 1.0) * (1 - min(rho * lam, 1.0)) ** (s - 1)
                          for s in range(1, env.tau + 1)])
        t = retime_answers(d.theta, u, env.pmf_h, pmf_m, env.tau)
        qs.append(float((t[mis] <= env.tau).mean()))
    assert all(a <= b + 1e-9 for a, b in zip(qs, qs[1:])), qs
    # distance from the honest rate grows on both sides of lambda=1
    assert abs(qs[0] - q_h) > 1e-3 and abs(qs[-1] - q_h) > 1e-3
    print(f"A4-4 ok: q_m monotone in lambda {['%.3f' % q for q in qs]} (q_h={q_h:.3f})")


def outage_smoke() -> None:
    """The regime-cell path runs end to end and writes an envelope."""
    with tempfile.TemporaryDirectory() as dd:
        path = run_axis("E-outage x F1", "lambda", 2001, 3_000, 2_000, dd,
                        n_boot=200)
        import json
        from pathlib import Path
        env = json.loads(Path(path).read_text())
        assert env["kind"] == "inject"
        assert env["payload"]["axis"] == "lambda"
        assert len(env["payload"]["advantage"]) == 7
        print("outage smoke ok: run_axis wrote an inject envelope")


def main() -> None:
    """Run every acceptance check; raise on the first failure."""
    a4_5_halving_interpolation()
    a4_1_identity_matches_exp1()
    a4_3_noise_direction()
    a4_4_lambda_identification()
    outage_smoke()
    print("\nS4 injection: all checks passed")


if __name__ == "__main__":
    main()
