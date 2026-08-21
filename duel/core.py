"""Exact evaluation of the four-action payment decision.

This module is the reference: every simulator and compiler in the
package is held to the numbers it produces.  It evaluates the decision
by dynamic programming under a geometric answer arrival, and also under
an arbitrary deadline-censored arrival distribution.

Model.  Stages i = 0..N, FINAL = N+1, FAILED absorbing with value 0.
  G(i,pi) = v(sigma(i)*kappa(pi) - 1),  kappa = 1 + (1-pi)m - pi*h
  reject  = 0
  W(i,pi) = -C - D(i) + (1-pi)A(i)          (commitment window)
  wait    = -c_w*v + (1-f_i)*V(i+1,pi)      (V(FAILED)=0)
  FINAL: wait disallowed.
Window recursion (tick order: pay wait cost, test arrival, transition):
  A(j,t) = rho*max(G_leg(j),0) + (1-rho)(1-f_j)A(j+1,t-1)
  D(j,t) = c_w*v + (1-rho)(1-f_j)D(j+1,t-1)
Settled closed forms: A_F = q*v*m, D_F = c_w*v*q/rho, q = 1-(1-rho)^tau.
"""
from __future__ import annotations

import numpy as np

# Worked-example parameters: an illustration, not a calibration.
DEFAULT = dict(m=0.3, h=1.0, C=0.06, cw=0.01, rho=0.5, tau=5)

ACTIONS = ("grant", "reject", "verify", "wait")
GRANT, REJECT, VERIFY, WAIT = 0, 1, 2, 3


def sigma_list(f):
    """f = [f_0..f_N]; returns sigma[0..N+1] with sigma[FINAL] = 1."""
    N = len(f) - 1
    sig = np.zeros(N + 2)
    sig[N + 1] = 1.0
    for i in range(N, -1, -1):
        sig[i] = (1.0 - f[i]) * sig[i + 1]
    return sig


def window_AD(f, sig, v, m, cw, rho, tau):
    """A(i), D(i) at the start of a commitment window opened in stage i,
    geometric arrival with per-tick answer probability rho."""
    N = len(f) - 1
    FIN = N + 1
    Gleg = np.array([v * (sig[j] * (1 + m) - 1.0) for j in range(N + 2)])
    ex = np.maximum(Gleg, 0.0)
    A = np.zeros((N + 2, tau + 1))
    D = np.zeros((N + 2, tau + 1))
    for t in range(1, tau + 1):
        for j in range(FIN, -1, -1):
            if j == FIN:
                cA, cD, surv = A[FIN, t - 1], D[FIN, t - 1], 1.0
            else:
                cA, cD, surv = A[j + 1, t - 1], D[j + 1, t - 1], 1.0 - f[j]
            A[j, t] = rho * ex[j] + (1 - rho) * surv * cA
            D[j, t] = cw * v + (1 - rho) * surv * cD
    return A[:, tau], D[:, tau]


def general_AD(f, v, pmf, m, cw, tau):
    """A(i), D(i) for an arbitrary censored arrival pmf
    pmf[s-1] = P(T = s), s = 1..tau, sum(pmf) = q <= 1."""
    N = len(f) - 1
    FIN = N + 1
    sig = sigma_list(f)
    ex = np.maximum(v * (sig * (1 + m) - 1.0), 0.0)
    A = np.zeros(FIN + 1)
    D = np.zeros(FIN + 1)
    for i in range(FIN + 1):
        S = np.ones(tau + 1)
        st = i
        for k in range(1, tau + 1):
            S[k] = S[k - 1] * (1.0 - (f[st] if st <= N else 0.0))
            st = min(st + 1, FIN)
        a = d = 0.0
        Pge = 1.0
        for s in range(1, tau + 1):
            j = min(i + s - 1, FIN)
            a += pmf[s - 1] * S[s - 1] * ex[j]
            d += Pge * S[s - 1]
            Pge -= pmf[s - 1]
        A[i] = a
        D[i] = cw * v * d
    return A, D


def action_values(f, v, pi_grid, p=DEFAULT, pmf=None):
    """Per-stage arrays of the four action values and V on a pi grid.
    pmf = None uses the geometric arrival with p['rho']; otherwise the
    censored pmf is used for the window and rho is ignored there."""
    m, h, C, cw, rho, tau = (p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau'])
    N = len(f) - 1
    FIN = N + 1
    sig = sigma_list(f)
    if pmf is None:
        A, D = window_AD(f, sig, v, m, cw, rho, tau)
    else:
        A, D = general_AD(f, v, pmf, m, cw, tau)
    pi = np.asarray(pi_grid)
    G = np.array([v * (sig[i] * (1 + (1 - pi) * m - pi * h) - 1.0) for i in range(N + 2)])
    W = np.array([-C - D[i] + (1 - pi) * A[i] for i in range(N + 2)])
    R = np.zeros_like(G)
    Wait = np.full_like(G, -np.inf)
    V = np.zeros_like(G)
    V[FIN] = np.maximum.reduce([G[FIN], R[FIN], W[FIN]])
    for i in range(N, -1, -1):
        Wait[i] = -cw * v + (1 - f[i]) * V[i + 1]
        V[i] = np.maximum.reduce([G[i], R[i], W[i], Wait[i]])
    return dict(G=G, R=R, W=W, Wait=Wait, V=V, sigma=sig, A=A, D=D, pi=pi)


def best_action(f, v, pi_grid, p=DEFAULT, pmf=None):
    """Per stage, index of the optimal action (GRANT, REJECT, VERIFY, WAIT).
    Tie order grant < reject < verify < wait."""
    out = action_values(f, v, pi_grid, p, pmf=pmf)
    N = len(f) - 1
    lab = np.zeros((N + 2, len(np.atleast_1d(pi_grid))), dtype=np.int8)
    for i in range(N + 2):
        stack = np.vstack([out['G'][i], out['R'][i], out['W'][i], out['Wait'][i]])
        lab[i] = np.argmax(stack, axis=0)
    return lab, out


# ---------- settled-state closed forms ----------
def derived(p=DEFAULT):
    m, h, C, cw, rho, tau = (p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau'])
    q = 1 - (1 - rho) ** tau
    mt = m * h / (m + h)
    vstar = C / (q * (mt - cw / rho)) if mt > cw / rho else np.inf
    rhostar = cw * (m + h) / (m * h)
    floor = C / (q * m)
    pihat = m / (m + h)
    return dict(q=q, mt=mt, vstar=vstar, rhostar=rhostar, floor=floor, pihat=pihat)


def settled_lines(v, pi, p=DEFAULT):
    m, h, C, cw, rho, tau = (p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau'])
    q = 1 - (1 - rho) ** tau
    G = v * ((1 - pi) * m - pi * h)
    W = -C - cw * v * q / rho + (1 - pi) * q * v * m
    return G, W


def verify_edges(v, p=DEFAULT):
    m, h, C, cw, rho, tau = (p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau'])
    q = 1 - (1 - rho) ** tau
    pgv = (m * (1 - q) + C / v + cw * q / rho) / ((m + h) - q * m)
    prv = 1 - (C / v + cw * q / rho) / (q * m)
    return pgv, prv


def rho_hat_from_q(q, tau):
    """A2's calibration point: the geometric rate that reproduces the
    observed within-deadline answer rate q at deadline tau."""
    return 1.0 - (1.0 - q) ** (1.0 / tau)
