"""The three policy families.

Family A (the model): compiled decision tables from the exact DP of
core.py.  A1 compiles from the true arrival pmf; A2 sees only the
calibration statistics (q, tau) and compiles under the geometric
arrival that reproduces them; A2v and A2w are A2 with the verify or
wait column removed (attribution panel).

Family B (tuned competitors): parametric rules tuned by grid search on
REALIZED profit over a tuning split, so they see the money, which A2
never does.  B1 rejects when expected loss pi*h*v clears a threshold;
B2 rejects on amount alone; B3 is a two-threshold suspicion rule with
verification in the middle band.

Family C (fixed rules, no parameters): grant-all, reject-all,
verify-all, always-wait-then-grant — the prevalence panel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (GRANT, REJECT, VERIFY, WAIT, action_values,
                   rho_hat_from_q)
from .simulate import Channel, Draws, replay


# ------------------------------------------------------------- family A
PI_GRID = np.linspace(0.0, 1.0, 2001)


def _masked_labels(out, drop=()):
    """Argmax over the four stacked action values with columns removed.
    Tie order grant < reject < verify < wait."""
    stacks = [out['G'], out['R'], out['W'], out['Wait']]
    n_st = out['G'].shape[0]
    lab = np.zeros((n_st, out['G'].shape[1]), dtype=np.int8)
    allowed = [a for a in (GRANT, REJECT, VERIFY, WAIT) if a not in drop]
    for i in range(n_st):
        stack = np.vstack([stacks[a][i] for a in allowed])
        lab[i] = np.asarray(allowed, dtype=np.int8)[np.argmax(stack, axis=0)]
    return lab


@dataclass
class CompiledPolicy:
    """Lookup table over (log-v bin, stage, pi bin)."""
    name: str
    v_grid: np.ndarray
    tables: np.ndarray          # (n_v, n_stage, n_pi) int8
    def __call__(self, stage, v, pi):
        iv = int(np.argmin(np.abs(np.log(self.v_grid) - np.log(max(v, 1e-12)))))
        ip = int(round(pi * (len(PI_GRID) - 1)))
        return int(self.tables[iv, stage, ip])


def compile_A(ch: Channel, name: str, pmf=None, rho: float | None = None,
              drop=(), n_v: int = 61, v_lo=0.5, v_hi=2000.0) -> CompiledPolicy:
    """Compile a family-A table on a log-spaced exposure grid.

    pmf given  -> window evaluated under that censored arrival pmf (A1).
    rho given  -> geometric window at that rate (A2 after calibration).
    drop       -> actions removed before the argmax (A2v, A2w).
    """
    v_grid = np.geomspace(v_lo, v_hi, n_v)
    tabs = []
    for v in v_grid:
        out = action_values(ch.f, v, PI_GRID, ch.params(rho=rho), pmf=pmf)
        tabs.append(_masked_labels(out, drop=drop))
    return CompiledPolicy(name, v_grid, np.stack(tabs))


def calibrate_q(ch: Channel, d: Draws) -> float:
    """A2's measurement campaign: the empirical within-deadline answer
    rate on the tuning split.  A2 sees this and tau, nothing else."""
    return float((d.t_ans <= ch.tau).mean())


def make_family_A(ch: Channel, tuning: Draws):
    q_hat = calibrate_q(ch, tuning)
    rho_hat = rho_hat_from_q(q_hat, ch.tau)
    return {
        'A_full': compile_A(ch, 'A_full', pmf=ch.pmf_h),
        'A': compile_A(ch, 'A', rho=rho_hat),
        'A_noV': compile_A(ch, 'A_noV', rho=rho_hat, drop=(VERIFY,)),
        'A_noW': compile_A(ch, 'A_noW', rho=rho_hat, drop=(WAIT,)),
    }, dict(q_hat=q_hat, rho_hat=rho_hat)


# ------------------------------------------------------------- family B
@dataclass
class B1:
    """Expected-loss threshold: reject when pi*h*v > theta."""
    theta: float
    h: float
    def __call__(self, stage, v, pi):
        return REJECT if pi * self.h * v > self.theta else GRANT


@dataclass
class B2:
    """Amount threshold, suspicion-blind: reject when v > theta."""
    theta: float
    def __call__(self, stage, v, pi):
        return REJECT if v > self.theta else GRANT


@dataclass
class B3:
    """Two suspicion thresholds: grant below a, reject above b,
    verify in between; exposure-blind."""
    a: float
    b: float
    def __call__(self, stage, v, pi):
        if stage > 0:
            return REJECT   # its verify already ran; terminal fallback
        if pi < self.a:
            return GRANT
        if pi > self.b:
            return REJECT
        return VERIFY


def tune(make_policy, grid, ch: Channel, tuning: Draws, ex_unit) -> tuple:
    """Grid-search a family-B rule on realized tuning profit.
    Returns (best_params, best_policy, table of (params, profit))."""
    rows = []
    best, best_val = None, -np.inf
    for params in grid:
        pol = make_policy(params)
        val = float(replay(ch, tuning, pol, ex_unit).mean())
        rows.append((params, val))
        if val > best_val:
            best, best_val = params, val
    return best, make_policy(best), rows


def suspicion_grid(n=21):
    """The two-threshold (a, b) grid at resolution n: a < b on
    linspace(0, 1, n).  n = 21 (step 0.05) is the declared default; finer
    n resolves how much of a cell's edge is grid coarseness."""
    axis = np.linspace(0.0, 1.0, n)
    return [(a, b) for a in axis for b in axis if a < b]


def default_grids(v_hi=2000.0, n=41):
    """Pre-declared tuning grids; generous by design (the asymmetry that
    favors the baselines is the defense against the strawman charge)."""
    return dict(
        B1=list(np.geomspace(1e-3, v_hi, n)),
        B2=list(np.geomspace(0.5, v_hi, n)),
        B3=suspicion_grid(21),
    )


# ------------------------------------------------------------- family C
class CGrant:
    name = 'C1'
    def __call__(self, s, v, pi):
        return GRANT


class CReject:
    name = 'C2'
    def __call__(self, s, v, pi):
        return REJECT


class CVerify:
    name = 'C3'
    def __call__(self, s, v, pi):
        return VERIFY if s == 0 else REJECT


@dataclass
class CWaitGrant:
    """Always wait, then grant at FINAL."""
    FIN: int
    name = 'C4'
    def __call__(self, s, v, pi):
        return GRANT if s >= self.FIN else WAIT


def make_family_C(ch: Channel):
    return {'C1': CGrant(), 'C2': CReject(), 'C3': CVerify(),
            'C4': CWaitGrant(ch.N + 1)}
