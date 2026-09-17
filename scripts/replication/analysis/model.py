"""Exact model evaluation for the figures and the two computations of the article.

Model: stages i=0..N, FINAL=N+1, FAILED absorbing with value 0.
  G(i,pi) = v(sigma(i)*kappa(pi) - 1),  kappa = 1 + (1-pi)m - pi*h
  reject  = 0
  W(i,pi) = -C - D(i) + (1-pi)A(i)          (commitment window)
  wait    = -c_w*v + (1-f_i)*V(i+1,pi)      (V(FAILED)=0)
  FINAL: wait disallowed.
Commitment window recursion (tick order: pay wait cost, test arrival, transition):
  A(j,t) = rho*max(G_leg(j),0) + (1-rho)(1-f_j)A(j+1,t-1)
  D(j,t) = c_w*v + (1-rho)(1-f_j)D(j+1,t-1)
Settled closed forms: A_F = q*v*m, D_F = c_w*v*q/rho, q = 1-(1-rho)^tau.
"""
import numpy as np

DEFAULT = dict(m=0.3, h=1.0, C=0.06, cw=0.01, rho=0.5, tau=5)


def sigma_list(f):
    """f = [f_0..f_N]; returns sigma[0..N+1] with sigma[FINAL]=1."""
    N = len(f) - 1
    sig = np.zeros(N + 2)
    sig[N + 1] = 1.0
    for i in range(N, -1, -1):
        sig[i] = (1.0 - f[i]) * sig[i + 1]
    return sig


def window_AD(f, sig, v, m, cw, rho, tau):
    """A(i), D(i) at the start of a commitment window opened in stage i."""
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


def action_values(f, v, pi_grid, p=DEFAULT):
    """Returns dict with per-stage arrays of the four action values and V."""
    m, h, C, cw, rho, tau = (p['m'], p['h'], p['C'], p['cw'], p['rho'], p['tau'])
    N = len(f) - 1
    FIN = N + 1
    sig = sigma_list(f)
    A, D = window_AD(f, sig, v, m, cw, rho, tau)
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


def best_action(f, v, pi_grid, p=DEFAULT, tol=1e-12):
    """Per stage, index of the optimal action: 0 grant, 1 reject, 2 verify, 3 wait."""
    out = action_values(f, v, pi_grid, p)
    N = len(f) - 1
    lab = np.zeros((N + 2, len(pi_grid)), dtype=int)
    for i in range(N + 2):
        stack = np.vstack([out['G'][i], out['R'][i], out['W'][i], out['Wait'][i]])
        lab[i] = np.argmax(stack, axis=0)
    return lab, out


# ---------- settled-state closed forms (Theorem 2) ----------
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
