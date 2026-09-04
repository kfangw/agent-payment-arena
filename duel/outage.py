"""E-outage: two-state regime-switching settlement (normal / outage).

In the outage state settlement progress halts while waiting costs and
validBefore keep burning, and failures arise through the expiry path,
so the per-stage hazards are not inflated.  The payment's authorization expires H ticks after
arrival; a payment that has not reached FINAL by then fails.

State: (i, l, r) with stage i in 0..N+1 (FINAL = N+1), remaining
validity l in 0..H, regime r in {0 normal, 1 outage}.  Tick order
matches core.py's window recursion: pay any waiting cost, arrival test
(inside a window), chain event (advance or hazard failure; frozen in
outage), regime transition, l decreases by one.  At l = 0 an
uncommitted, unsettled payment has expired and can only reject.

The compiled policy conditions on (i, l, r), all operator-observable:
outages are public (the chain visibly halts) and validBefore is a
field of the authorization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT


@dataclass
class OutageEnv:
    """Channel constants for the E-outage cell (coarse tick).

    f    : per-stage hazards [f_0..f_N] at the coarse tick
    H    : validBefore in ticks
    tau  : answer deadline in ticks (window cap; the effective window
           is min(tau, l) at the moment the question is sent)
    rho  : per-tick answer hazard (geometric within the window)
    p01  : P(normal -> outage) per tick;  p10 : P(outage -> normal)
    """

    f: np.ndarray
    m: float
    h: float
    C: float
    cw: float
    tau: int
    H: int
    rho: float
    p01: float
    p10: float
    tick_seconds: int = 60

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)

    @property
    def N(self):
        return len(self.f) - 1

    @property
    def stationary_outage(self):
        return self.p01 / (self.p01 + self.p10)

    def trans(self):
        """P[r, r'] regime transition matrix."""
        return np.array([[1 - self.p01, self.p01], [self.p10, 1 - self.p10]])


# ------------------------------------------------------------ exact DP
def survival(env: OutageEnv) -> np.ndarray:
    """sig[i, l, r] = P(reach FINAL before expiry | state).  The tick-t
    chain event is included: committing now rides this tick's advance."""
    N, H = env.N, env.H
    FIN = N + 1
    P = env.trans()
    sig = np.zeros((FIN + 1, H + 1, 2))
    sig[FIN, :, :] = 1.0
    for l in range(1, H + 1):
        for i in range(N, -1, -1):
            sig[i, l, 0] = (1 - env.f[i]) * (sig[i + 1, l - 1, :] @ P[0])
            sig[i, l, 1] = sig[i, l - 1, :] @ P[1]
    return sig


def window_AD(env: OutageEnv, sig: np.ndarray):
    """A[t][i,l,r], D[t][i,l,r] per unit v, for a window with t answer
    ticks remaining.  Expiry (l = 0) kills the window regardless of t."""
    N, H, tau = env.N, env.H, env.tau
    FIN = N + 1
    P = env.trans()
    ex = np.maximum(sig * (1 + env.m) - 1.0, 0.0)
    A = np.zeros((tau + 1, FIN + 1, H + 1, 2))
    D = np.zeros((tau + 1, FIN + 1, H + 1, 2))
    for t in range(1, tau + 1):
        for l in range(1, H + 1):
            for i in range(FIN, -1, -1):
                for r in (0, 1):
                    if r == 0 and i <= N:
                        surv, ii = 1.0 - env.f[i], i + 1
                    else:
                        surv, ii = 1.0, i
                    cA = A[t - 1, ii, l - 1, :] @ P[r]
                    cD = D[t - 1, ii, l - 1, :] @ P[r]
                    A[t, i, l, r] = env.rho * ex[i, l, r] + (1 - env.rho) * surv * cA
                    D[t, i, l, r] = env.cw + (1 - env.rho) * surv * cD
    return A, D, ex


def value_labels(env: OutageEnv, v, pi_grid, drop=()):
    """Optimal values and labels over the state space at exposure v,
    with actions in `drop` removed from the argmax (the continuation V
    is recomputed under the mask).  Tie order grant<reject<verify<wait."""
    N, H, tau = env.N, env.H, env.tau
    FIN = N + 1
    P = env.trans()
    sig = survival(env)
    A, D, ex = window_AD(env, sig)
    pi = np.asarray(pi_grid)
    kappa = 1 + (1 - pi) * env.m - pi * env.h
    npi = len(pi)
    V = np.zeros((FIN + 1, H + 1, 2, npi))
    lab = np.zeros((FIN + 1, H + 1, 2, npi), dtype=np.int8)
    allowed = [a for a in (GRANT, REJECT, VERIFY, WAIT) if a not in drop]
    minus_inf = np.full(npi, -np.inf)
    for l in range(0, H + 1):
        for i in range(FIN, -1, -1):
            for r in (0, 1):
                G = v * (sig[i, l, r] * kappa - 1.0)
                R = np.zeros(npi)
                w = min(tau, l)
                if w > 0:
                    W = -env.C - v * D[w, i, l, r] + (1 - pi) * v * A[w, i, l, r]
                else:
                    W = np.full(npi, -env.C)
                if l == 0 or i == FIN:
                    Wait = minus_inf
                elif r == 0:
                    cont = np.tensordot(P[0], V[i + 1, l - 1, :, :], axes=(0, 0))
                    Wait = -env.cw * v + (1 - env.f[i]) * cont
                else:
                    cont = np.tensordot(P[1], V[i, l - 1, :, :], axes=(0, 0))
                    Wait = -env.cw * v + cont
                acts = {GRANT: G, REJECT: R, VERIFY: W, WAIT: Wait}
                stack = np.vstack([acts[a] for a in allowed])
                V[i, l, r] = stack.max(axis=0)
                lab[i, l, r] = np.asarray(allowed, dtype=np.int8)[stack.argmax(axis=0)]
    return lab, V, dict(sig=sig, A=A, D=D, ex=ex)


# ------------------------------------------------------------ compiler
PI_GRID_OUTAGE = np.linspace(0.0, 1.0, 501)


@dataclass
class CompiledOutagePolicy:
    name: str
    v_grid: np.ndarray
    tables: list  # per v: labels[i, l, r, pi_bin]

    def __call__(self, i, l, r, v, pi):
        iv = int(np.argmin(np.abs(np.log(self.v_grid) - np.log(max(v, 1e-12)))))
        ip = int(round(pi * (len(PI_GRID_OUTAGE) - 1)))
        return int(self.tables[iv][i, l, r, ip])


def compile_outage(
    env: OutageEnv, name: str, n_v=41, v_lo=0.5, v_hi=2000.0, drop=()
) -> CompiledOutagePolicy:
    v_grid = np.geomspace(v_lo, v_hi, n_v)
    tabs = [value_labels(env, v, PI_GRID_OUTAGE, drop=drop)[0] for v in v_grid]
    return CompiledOutagePolicy(name, v_grid, tabs)


# ------------------------------------------------------------ simulator
@dataclass
class OutageDraws:
    v: np.ndarray
    p_true: np.ndarray
    theta: np.ndarray
    pi0: np.ndarray
    u_stage: np.ndarray  # (n, N+1) uniforms for hazard firing
    t_ans: np.ndarray  # answer delay from query (geometric draw)
    paths: np.ndarray  # (n, H+1) regime path from arrival tick

    def __len__(self):
        return len(self.v)


def retime_answers_geom(theta, u_ans, rho_h, rho_m):
    """Geometric answer ticks by inverse-CDF from a fixed uniform, per
    intent.  t = ceil(log(1 - u) / log(1 - rho)) reproduces geometric(rho)
    and moves continuously with rho, so a response-bias sweep keeps common
    random numbers across its levels."""
    n = len(theta)
    t = np.ones(n, dtype=np.int64)
    for intent, rho in ((0, rho_h), (1, rho_m)):
        idx = np.where(theta == intent)[0]
        if len(idx) == 0:
            continue
        r = min(max(float(rho), 1e-9), 1.0 - 1e-12)
        u = np.clip(u_ans[idx], 1e-12, 1.0 - 1e-12)
        tt = np.ceil(np.log1p(-u) / np.log1p(-r))
        t[idx] = np.maximum(tt.astype(np.int64), 1)
    return t


def draw_outage_batch(
    env: OutageEnv,
    flow,
    n: int,
    rng,
    episode_len: int = 1440,
    payments_per_episode: int = 50,
    rho_m: float | None = None,
):
    """Episode-structured common random numbers.  Each episode owns one
    regime path (stationary start); its payments arrive at uniform
    offsets and read the same path, which is what makes regime shocks
    correlate within an episode (block bootstrap consumes this).

    rho_m, if given, is the misuse-intent answer hazard for the response
    bias axis; honest intent keeps env.rho."""
    return _draw_outage(env, flow, n, rng, episode_len, payments_per_episode, rho_m)[0]


def draw_outage_batch_crn(
    env: OutageEnv, flow, n: int, rng, episode_len: int = 1440, payments_per_episode: int = 50
):
    """Like draw_outage_batch, but also return the fixed answer uniforms
    for a response-bias sweep."""
    return _draw_outage(env, flow, n, rng, episode_len, payments_per_episode, None)


def _draw_outage(
    env: OutageEnv,
    flow,
    n: int,
    rng,
    episode_len: int,
    payments_per_episode: int,
    rho_m: float | None,
):
    """Shared outage draw body; returns the batch and the answer uniforms."""
    v, p_true, theta, pi0 = flow.sample(n, rng)
    u_stage = rng.random((n, env.N + 1))
    u_ans = rng.random(n)
    t_ans = retime_answers_geom(theta, u_ans, env.rho, env.rho if rho_m is None else rho_m)
    n_ep = (n + payments_per_episode - 1) // payments_per_episode
    paths = np.zeros((n, env.H + 1), dtype=np.int8)
    k = 0
    for _ in range(n_ep):
        m_here = min(payments_per_episode, n - k)
        L = episode_len + env.H + 1
        path = np.zeros(L, dtype=np.int8)
        path[0] = 1 if rng.random() < env.stationary_outage else 0
        u = rng.random(L - 1)
        for t in range(1, L):
            r = path[t - 1]
            flip = u[t - 1] < (env.p01 if r == 0 else env.p10)
            path[t] = (1 - r) if flip else r
        offs = rng.integers(0, episode_len, size=m_here)
        for j in range(m_here):
            paths[k + j] = path[offs[j] : offs[j] + env.H + 1]
        k += m_here
    d = OutageDraws(
        v=v, p_true=p_true, theta=theta, pi0=pi0, u_stage=u_stage, t_ans=t_ans, paths=paths
    )
    return d, u_ans


def _settle_from(env: OutageEnv, d: OutageDraws, k: int, i: int, t: int) -> int:
    """From stage i at tick t (this tick's chain event still pending),
    run the chain to completion.  Returns 1 settled, 0 failed."""
    N, H = env.N, env.H
    while True:
        if i > N:
            return 1
        if H - t <= 0:
            return 0
        if d.paths[k, t] == 0:
            if d.u_stage[k, i] < env.f[i]:
                return 0
            i += 1
        t += 1


def replay_outage(env: OutageEnv, d: OutageDraws, policy, ex_pos) -> np.ndarray:
    """Replay under `policy(i, l, r, v, pi)`.  ex_pos[i, l, r] > 0 marks
    states where the expected grant leg is positive (v-free sign), the
    window's exercise rule, shared by every policy as part of the
    environment's verify semantics."""
    n = len(d)
    N, H, FIN = env.N, env.H, env.N + 1
    out = np.zeros(n)
    for k in range(n):
        v, pi = float(d.v[k]), float(d.pi0[k])
        payoff = 0.0
        i, t = 0, 0
        while True:
            if i > N:
                i = FIN
            l = H - t
            r = int(d.paths[k, t])
            if l <= 0 and i < FIN:
                break  # expired uncommitted: reject
            a = policy(i, max(l, 0), r, v, pi)
            if a == GRANT:
                won = 1 if i == FIN else _settle_from(env, d, k, i, t)
                payoff += v * ((env.m if d.theta[k] == 0 else -env.h) if won else -1.0)
                break
            if a == REJECT:
                break
            if a == VERIFY:
                payoff -= env.C
                w = min(env.tau, max(l, 0))
                s = 0
                while s < w:
                    payoff -= env.cw * v
                    s += 1
                    if d.t_ans[k] == s:
                        li, lr = H - t, int(d.paths[k, t])
                        if d.theta[k] == 0 and ex_pos[i, li, lr] > 0:
                            won = 1 if i == FIN else _settle_from(env, d, k, i, t)
                            payoff += v * (env.m if won else -1.0)
                        break
                    if int(d.paths[k, t]) == 0 and i <= N:
                        if d.u_stage[k, i] < env.f[i]:
                            break  # failed inside the window
                        i += 1
                        if i > N:
                            i = FIN
                    t += 1
                    if H - t <= 0:
                        break  # expired inside the window
                break
            if a == WAIT:
                payoff -= env.cw * v
                if r == 0 and i <= N:
                    if d.u_stage[k, i] < env.f[i]:
                        break  # failed while waiting
                    i += 1
                t += 1
                continue
            break  # defensive: unknown action
        out[k] = payoff
    return out
