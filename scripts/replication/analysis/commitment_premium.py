#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the verify commitment lose value against a server free to abandon?

Flexible model: after paying C and sending the query at stage i, the server
keeps its ordinary options at every tick while no answer has arrived: abandon
to grant, abandon to reject, or continue.  The answer, when it arrives,
executes the truth-revealed optimum (legit -> state-contingent grant, misuse
-> reject).  The deadline default stays reject (design constraint of the
mechanism).  Committed model: no abandonment; the registered rule runs.

Questions:
 1. Is V_flex ever strictly above V_committed at the episode level?
 2. Settled-state closed form: premium on the verify region
      Delta(tau) = max_{1<=t<=tau} (1-rho)^(tau-t) * (max(G,0) - w(t))^+ ,
      w(t) = (1-(1-rho)^t) * ((1-pi) v m - c_w v / rho).
    Verify against brute-force DP.
 3. General stages: sweep random instances, report the premium's size and
    where it is zero.
"""
import math

import numpy as np

import model as M

P = dict(M.DEFAULT)


def flexible_U(f, v, pi, p):
    """Post-query value U(j, t) with abandonment; deadline default reject."""
    m, h, C, cw, rho, tau = p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau']
    N = len(f) - 1
    FIN = N + 1
    sig = M.sigma_list(np.array(f))
    Gleg = np.array([v * (sig[j] * (1 + m) - 1.0) for j in range(FIN + 1)])
    ex = np.maximum(Gleg, 0.0)
    G = np.array([v * (sig[j] * (1 + (1 - pi) * m - pi * h) - 1.0)
                  for j in range(FIN + 1)])
    ans = (1 - pi) * ex          # truth-revealed execution value per state
    U = np.zeros((FIN + 1, tau + 1))   # U[j, t]; t = ticks remaining; U[.,0]=0
    for t in range(1, tau + 1):
        for j in range(FIN, -1, -1):
            nxt = U[FIN, t - 1] if j == FIN else U[j + 1, t - 1]
            surv = 1.0 if j == FIN else 1.0 - f[j]
            cont = -cw * v + rho * ans[j] + (1 - rho) * surv * nxt
            U[j, t] = max(G[j], 0.0, cont)
    return U


def committed_w(f, v, pi, p):
    """Post-query value w(j, t) of the registered rule (no abandonment)."""
    m, h, C, cw, rho, tau = p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau']
    N = len(f) - 1
    FIN = N + 1
    sig = M.sigma_list(np.array(f))
    ex = np.maximum(np.array([v * (sig[j] * (1 + m) - 1.0)
                              for j in range(FIN + 1)]), 0.0)
    ans = (1 - pi) * ex
    w = np.zeros((FIN + 1, tau + 1))
    for t in range(1, tau + 1):
        for j in range(FIN, -1, -1):
            nxt = w[FIN, t - 1] if j == FIN else w[j + 1, t - 1]
            surv = 1.0 if j == FIN else 1.0 - f[j]
            w[j, t] = -cw * v + rho * ans[j] + (1 - rho) * surv * nxt
    return w


def episode_values(f, v, pi, p):
    """V_committed and V_flex at every stage (verify = -C + window value)."""
    m, h, C, cw, rho, tau = p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau']
    N = len(f) - 1
    FIN = N + 1
    sig = M.sigma_list(np.array(f))
    G = np.array([v * (sig[j] * (1 + (1 - pi) * m - pi * h) - 1.0)
                  for j in range(FIN + 1)])
    U = flexible_U(f, v, pi, p)
    w = committed_w(f, v, pi, p)

    def induct(Wcol):
        V = np.zeros(FIN + 1)
        V[FIN] = max(G[FIN], 0.0, Wcol[FIN])
        for i in range(N, -1, -1):
            wait = -cw * v + (1 - f[i]) * V[i + 1]
            V[i] = max(G[i], 0.0, Wcol[i], wait)
        return V

    Vc = induct(-C + w[:, tau])
    Vf = induct(-C + U[:, tau])
    return Vc, Vf, G, U, w


def settled_premium_formula(v, pi, p):
    m, h, C, cw, rho, tau = p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau']
    G = v * ((1 - pi) * m - pi * h)
    best = 0.0
    for t in range(1, tau + 1):
        wt = (1 - (1 - rho) ** t) * ((1 - pi) * v * m - cw * v / rho)
        best = max(best, (1 - rho) ** (tau - t) * max(max(G, 0.0) - wt, 0.0))
    return best


def main():
    rng = np.random.default_rng(20260816)

    # ---- 1+2: settled state, formula vs brute force -----------------------
    print("== settled state: brute-force premium vs closed form ==")
    p = dict(P)
    worst = 0.0
    for _ in range(4000):
        v = float(np.exp(rng.uniform(np.log(0.05), np.log(50))))
        pi = float(rng.uniform(0, 1))
        p2 = dict(p)
        p2['m'] = float(rng.uniform(0.05, 1.5)); p2['h'] = float(rng.uniform(0.1, 4))
        p2['C'] = float(rng.uniform(0.005, 1)); p2['cw'] = float(rng.uniform(0.0, 0.2))
        p2['rho'] = float(rng.uniform(0.05, 0.95)); p2['tau'] = int(rng.integers(1, 12))
        f0 = [0.0]
        Vc, Vf, G, U, w = episode_values(f0, v, pi, p2)
        # premium of flexibility inside the verify option at FINAL:
        dU = (-p2['C'] + U[-1, p2['tau']]) - (-p2['C'] + w[-1, p2['tau']])
        formula = settled_premium_formula(v, pi, p2)
        worst = max(worst, abs(dU - formula))
    print(f"  4000 random settled instances: max |bruteforce - formula| = {worst:.2e}")

    # worked-example instance from the derivation
    p3 = dict(P)
    Vc, Vf, G, U, w = episode_values([0.0], 2.0, 0.08, p3)
    print(f"  instance v=2, pi=0.08: V_committed={Vc[-1]:.4f}  V_flex={Vf[-1]:.4f}  "
          f"episode premium={Vf[-1]-Vc[-1]:.4f}  formula={settled_premium_formula(2.0,0.08,p3):.4f}")

    # ---- 3: general stages, random sweep ---------------------------------
    print("\n== general stages: episode-level premium sweep ==")
    n_pos = 0; n_tot = 0; max_prem = 0.0; max_rel = 0.0; max_inst = None
    zero_when_reject = True; zero_when_verify_off = True
    for _ in range(4000):
        Nst = int(rng.integers(1, 4))
        f = [float(rng.uniform(0, 0.15)) for _ in range(Nst)]
        v = float(np.exp(rng.uniform(np.log(0.05), np.log(50))))
        pi = float(rng.uniform(0, 1))
        p2 = dict(P)
        p2['m'] = float(rng.uniform(0.05, 1.5)); p2['h'] = float(rng.uniform(0.1, 4))
        p2['C'] = float(rng.uniform(0.005, 1)); p2['cw'] = float(rng.uniform(0.0, 0.2))
        p2['rho'] = float(rng.uniform(0.05, 0.95)); p2['tau'] = int(rng.integers(1, 12))
        Vc, Vf, G, U, w = episode_values(f, v, pi, p2)
        prem = float(Vf[0] - Vc[0])
        n_tot += 1
        if prem > 1e-12 * max(v, 1.0):
            n_pos += 1
            if prem > max_prem:
                max_prem, max_inst = prem, (Nst, v, pi, p2)
            max_rel = max(max_rel, prem / v)
            # was verify part of the committed optimum here?
            lab, _ = M.best_action(f, v, np.array([pi]), p2)
            if not (lab == 2).any():
                zero_when_verify_off = False
            pihat = p2['m'] / (p2['m'] + p2['h'])
            if pi >= pihat:
                zero_when_reject = False
    print(f"  {n_pos}/{n_tot} instances with strictly positive premium")
    print(f"  max premium {max_prem:.4f}  max premium/exposure {max_rel:.4f}")
    print(f"  premium>0 only where committed policy uses verify: {zero_when_verify_off}")
    print(f"  premium>0 only on the approval side (pi < pi-hat): {zero_when_reject}")

    # candidate bound at general stages: nu/(1-rho) * max_j (G(j,pi))^+
    print("\n== bound check: premium <= (1-rho)^(tau-1) * max_j G(j,pi)^+ ==")
    viol = 0; worst_ratio = 0.0
    rng2 = np.random.default_rng(7)
    for _ in range(4000):
        Nst = int(rng2.integers(1, 4))
        f = [float(rng2.uniform(0, 0.15)) for _ in range(Nst)]
        v = float(np.exp(rng2.uniform(np.log(0.05), np.log(50))))
        pi = float(rng2.uniform(0, 1))
        p2 = dict(P)
        p2['m'] = float(rng2.uniform(0.05, 1.5)); p2['h'] = float(rng2.uniform(0.1, 4))
        p2['C'] = float(rng2.uniform(0.005, 1)); p2['cw'] = float(rng2.uniform(0.0, 0.2))
        p2['rho'] = float(rng2.uniform(0.05, 0.95)); p2['tau'] = int(rng2.integers(1, 12))
        Vc, Vf, G, U, w = episode_values(f, v, pi, p2)
        prem = float(Vf[0] - Vc[0])
        bound = (1 - p2['rho']) ** (p2['tau'] - 1) * max(float(G.max()), 0.0)
        if prem > bound + 1e-10:
            viol += 1
        if bound > 0:
            worst_ratio = max(worst_ratio, prem / bound)
    print(f"  violations: {viol}/4000   worst premium/bound ratio: {worst_ratio:.4f}")


if __name__ == "__main__":
    main()
