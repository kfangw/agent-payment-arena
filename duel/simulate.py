"""Realized replay of the four-action decision on pre-drawn randomness.

The harness draws every random object a payment will ever consume before
any policy sees it: intent, the settlement path, and the answer arrival.
Every policy then replays the same payments, so a difference in average
payoff is a difference between policies and not between draws.

Tick order inside a commitment window matches the recursion of core.py:
pay the waiting cost, test the arrival, then the settlement transition.
Outside a window, a wait action pays the cost and then the transition
fires.  Beliefs do not update while waiting; the belief axis moves only
through the delegator's answer, which reveals intent exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT


@dataclass
class Channel:
    """Everything a cell holds constant besides the flow.

    f        : per-stage settlement hazards [f_0..f_N]
    m, h     : margin and the operator-chosen misuse weight
    C, cw    : question fee and per-tick waiting cost rate
    tau      : verification deadline in ticks (validBefore remainder)
    pmf_h    : true censored arrival pmf for honest intent, s = 1..tau
    pmf_m    : same for misuse intent (equal to pmf_h unless a
               response-bias perturbation is being injected)
    """
    f: np.ndarray
    m: float
    h: float
    C: float
    cw: float
    tau: int
    pmf_h: np.ndarray
    pmf_m: np.ndarray | None = None

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)
        self.pmf_h = np.asarray(self.pmf_h, dtype=float)
        if self.pmf_m is None:
            self.pmf_m = self.pmf_h.copy()
        else:
            self.pmf_m = np.asarray(self.pmf_m, dtype=float)
        assert len(self.pmf_h) == self.tau and len(self.pmf_m) == self.tau
        assert self.pmf_h.sum() <= 1.0 + 1e-12 and self.pmf_m.sum() <= 1.0 + 1e-12

    @property
    def N(self) -> int:
        return len(self.f) - 1

    def params(self, rho: float | None = None) -> dict:
        """core.py parameter dict; rho only matters for geometric evaluation."""
        return dict(m=self.m, h=self.h, C=self.C, cw=self.cw,
                    rho=(rho if rho is not None else 0.5), tau=self.tau)


@dataclass
class Draws:
    """Common random numbers for a batch of payments.

    v        : exposures
    p_true   : true misuse probabilities
    theta    : true intents (1 = misuse)
    pi0      : scored suspicions handed to the policies
    fail_at  : first settlement stage whose hazard fires, or N+1 if none;
               a payment's settlement fate is this single number, shared
               by every policy and every action that reaches that stage
    t_ans    : answer delay in ticks from the moment a query is sent
               (np.iinfo.max = never answers within any deadline)
    """
    v: np.ndarray
    p_true: np.ndarray
    theta: np.ndarray
    pi0: np.ndarray
    fail_at: np.ndarray
    t_ans: np.ndarray

    def __len__(self):
        return len(self.v)


NEVER = np.iinfo(np.int64).max


def draw_batch(ch: Channel, flow, n: int, rng: np.random.Generator) -> Draws:
    """Pre-draw n payments from a flow on a channel.

    flow must provide sample(n, rng) -> (v, p_true, theta, pi0).
    """
    v, p_true, theta, pi0 = flow.sample(n, rng)
    N = ch.N
    u = rng.random((n, N + 1))
    fired = u < ch.f[None, :]
    fail_at = np.where(fired.any(axis=1), fired.argmax(axis=1), N + 1)
    t_ans = np.full(n, NEVER, dtype=np.int64)
    for intent, pmf in ((0, ch.pmf_h), (1, ch.pmf_m)):
        idx = np.where(theta == intent)[0]
        if len(idx) == 0:
            continue
        q = pmf.sum()
        probs = np.append(pmf, max(0.0, 1.0 - q))
        s = rng.choice(len(probs), size=len(idx), p=probs / probs.sum())
        t = np.where(s < ch.tau, s + 1, NEVER)
        t_ans[idx] = t
    return Draws(v=v, p_true=p_true, theta=theta, pi0=pi0,
                 fail_at=fail_at.astype(np.int64), t_ans=t_ans)


def _grant_leg(ch: Channel, k: int, d: Draws, stage: int) -> float:
    """Realized payoff of committing at `stage`: survive to FINAL or lose v."""
    if d.fail_at[k] >= stage and d.fail_at[k] <= ch.N:
        return -d.v[k]
    return d.v[k] * (ch.m if d.theta[k] == 0 else -ch.h)


def replay(ch: Channel, d: Draws, policy, ex_expected: np.ndarray) -> np.ndarray:
    """Replay every payment under `policy`; returns realized payoffs.

    policy(stage, v, pi) -> action in {GRANT, REJECT, VERIFY, WAIT}.
    ex_expected[j] must be the compiler's expected exercise values
    max(G_leg(j), 0) per stage — the window exercises the grant leg
    exactly when the EXPECTED leg value is positive, matching A(j,t).
    Note ex_expected scales with v: pass per-unit values (v = 1) and the
    replay multiplies by the payment's v; G_leg is linear in v so the
    sign, which is all the exercise rule uses, is v-free.
    """
    n = len(d)
    out = np.zeros(n)
    N, FIN = ch.N, ch.N + 1
    for k in range(n):
        v, pi = float(d.v[k]), float(d.pi0[k])
        payoff = 0.0
        stage = 0
        while True:
            a = policy(stage, v, pi)
            if a == WAIT and stage == FIN:
                a = REJECT  # wait is disallowed at FINAL; guard, never reached
            if a == GRANT:
                payoff += _grant_leg(ch, k, d, stage)
                break
            if a == REJECT:
                break
            if a == VERIFY:
                payoff -= ch.C
                j = stage
                answered = False
                for s in range(1, ch.tau + 1):
                    payoff -= ch.cw * v
                    if d.t_ans[k] == s:
                        answered = True
                        if d.theta[k] == 0 and ex_expected[min(j, FIN)] > 0.0:
                            payoff += _grant_leg(ch, k, d, j)
                        break
                    if j <= N and d.fail_at[k] == j:
                        break  # settlement failed inside the window
                    j = min(j + 1, FIN)
                break
            # WAIT
            payoff -= ch.cw * v
            if d.fail_at[k] == stage:
                break  # payment failed while waiting; resource never committed
            stage += 1
            if stage > FIN:
                break
    # unreachable
        out[k] = payoff
    return out
