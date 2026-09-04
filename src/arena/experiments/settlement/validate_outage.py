"""E-outage validation: realized replay must reproduce the exact
regime-switching DP, policy by policy, at Monte Carlo precision.

Run:  python -m arena.experiments.settlement.validate_outage
"""

from __future__ import annotations

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT
from .flows import Flow, MixD, LogNormalV
from .outage import (
    OutageEnv,
    compile_outage,
    draw_outage_batch,
    replay_outage,
    survival,
    value_labels,
    window_AD,
)


def draft_env(**kw):
    base = dict(
        f=np.array([0.005] + [3e-5] * 16),
        m=0.35,
        h=1.0,
        C=0.5,
        cw=0.006,
        tau=10,
        H=30,
        rho=0.1294,
        p01=4.63e-5,
        p10=1.0 / 60.0,
    )
    base.update(kw)
    return OutageEnv(**base)


# ---- fixed rules in the (i, l, r) signature ----
class OGrant:
    def __call__(self, i, l, r, v, pi):
        return GRANT


class OReject:
    def __call__(self, i, l, r, v, pi):
        return REJECT


class OVerify:
    """Verify at arrival, terminal reject if somehow asked again."""

    def __init__(self):
        self.asked = False

    def __call__(self, i, l, r, v, pi):
        return VERIFY


class OWaitGrant:
    def __init__(self, FIN):
        self.FIN = FIN

    def __call__(self, i, l, r, v, pi):
        return GRANT if i >= self.FIN else WAIT


def affine_wait_chain(env: OutageEnv):
    """U(i,l,r) = v*(alpha*kappa + beta) for the wait-until-FINAL-then-
    grant rule, with expiry paying nothing further (reject)."""
    N, H, FIN = env.N, env.H, env.N + 1
    P = env.trans()
    alpha = np.zeros((FIN + 1, H + 1, 2))
    beta = np.zeros((FIN + 1, H + 1, 2))
    alpha[FIN, :, :] = 1.0
    beta[FIN, :, :] = -1.0
    for l in range(1, H + 1):
        for i in range(N, -1, -1):
            for r in (0, 1):
                if r == 0:
                    ea = alpha[i + 1, l - 1, :] @ P[0]
                    eb = beta[i + 1, l - 1, :] @ P[0]
                    alpha[i, l, r] = (1 - env.f[i]) * ea
                    beta[i, l, r] = -env.cw + (1 - env.f[i]) * eb
                else:
                    alpha[i, l, r] = alpha[i, l - 1, :] @ P[1]
                    beta[i, l, r] = -env.cw + beta[i, l - 1, :] @ P[1]
    return alpha, beta


def main():
    rng = np.random.default_rng(20260822)
    env = draft_env()
    FIN = env.N + 1
    sig = survival(env)
    A, D, ex = window_AD(env, sig)
    alpha, beta = affine_wait_chain(env)

    flow = Flow("F1cal", MixD(0.05), LogNormalV(30.0, 1.0, 0.5, 2000.0))
    n = 300_000
    d = draw_outage_batch(env, flow, n, rng)
    r0 = d.paths[:, 0].astype(int)
    kappa = 1 + (1 - d.p_true) * env.m - d.p_true * env.h
    w0 = min(env.tau, env.H)

    exact = {
        "C1": d.v * (sig[0, env.H, r0] * kappa - 1.0),
        "C2": np.zeros(n),
        "C3": (-env.C - d.v * D[w0, 0, env.H, r0] + (1 - d.p_true) * d.v * A[w0, 0, env.H, r0]),
        "C4": d.v * (alpha[0, env.H, r0] * kappa + beta[0, env.H, r0]),
    }
    pols = {"C1": OGrant(), "C2": OReject(), "C3": OVerify(), "C4": OWaitGrant(FIN)}

    print(
        f"E-outage draft env: N={env.N} H={env.H} tau={env.tau} "
        f"stationary outage={env.stationary_outage:.4%}"
    )
    print(f"[n={n}]  (realized vs exact, per payment, dollars)")
    for name in ("C1", "C2", "C3", "C4"):
        r = replay_outage(env, d, pols[name], ex)
        e = exact[name].mean()
        diff = r - exact[name]
        se = diff.std(ddof=1) / np.sqrt(n)
        z = diff.mean() / se if se > 0 else 0.0
        print(f"  {name}: realized {r.mean():+8.5f}  exact {e:+8.5f}  z={z:+.2f}")
        assert abs(z) < 4.0, name

    # optimal policy at a fixed exposure: exact DP value vs replay
    v_fix = 30.0
    flow_fix = Flow("F1fix", MixD(0.05), LogNormalV(v_fix, 1e-9, v_fix, v_fix))
    n2 = 60_000
    d2 = draw_outage_batch(env, flow_fix, n2, rng)
    r02 = d2.paths[:, 0].astype(int)
    lab, V, parts = value_labels(env, v_fix, d2.p_true)
    e_opt = V[0, env.H, r02, np.arange(n2)]

    class ExactPol:
        """Reads the label table by the payment's own column index, so
        each payment is played by its exact optimal policy."""

        idx = 0

        def __call__(self, i, l, r, v, pi):
            return int(lab[i, l, r, self.idx])

    real = np.zeros(n2)
    ep = ExactPol()
    for k in range(n2):
        ep.idx = k
        one = OutageDrawsView(d2, k)
        real[k] = replay_outage(env, one, ep, parts["ex"])[0]
    diff = real - e_opt
    se = diff.std(ddof=1) / np.sqrt(n2)
    print(
        f"  OPT(v=30): realized {real.mean():+8.5f}  exact {e_opt.mean():+8.5f}"
        f"  z={diff.mean() / se:+.2f}"
    )
    assert abs(diff.mean() / se) < 4.0


class OutageDrawsView:
    """Single-payment view for per-payment exact-policy replay."""

    def __init__(self, d, k):
        self.v = d.v[k : k + 1]
        self.p_true = d.p_true[k : k + 1]
        self.theta = d.theta[k : k + 1]
        self.pi0 = d.pi0[k : k + 1]
        self.u_stage = d.u_stage[k : k + 1]
        self.t_ans = d.t_ans[k : k + 1]
        self.paths = d.paths[k : k + 1]

    def __len__(self):
        return 1


if __name__ == "__main__":
    main()
