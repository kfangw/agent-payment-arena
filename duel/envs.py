"""The three environments (rail axis) and their channel specs.

Each environment owns one case of the regime discriminant: E-fast the
riskless immediate-decision world, E-outage the intermittently struck
world, E-slow the slowly sinking one.  A cell = environment x flow, and
the channel constants (C, c_w, T profile, tau) are declared per cell.

DRAFT STATUS.  Numbers marked TODO are working defaults for the gate
computation, to be replaced by measured values with sources.

Tick declarations (one shared clock per environment across settlement,
cost, and answer axes):
  E-fast   tick = 2 s   (one L2 block on the measured fast rail)
  E-outage tick = 60 s  (outage dynamics are hour-scale; c_w, rho, tau
                         rescaled to the coarser tick) -- TODO: confirm
  E-slow   tick = 2 s   (block time of the reconstructed slow rail)
"""
from __future__ import annotations

import numpy as np

from .simulate import Channel

# Cost anchors (c_w is per 2 s tick)
CW_PER_2S = 0.002       # sustained marginal delay cost ~0.1%/s
C_DOLLAR = 0.5          # conservative floor; $5/$50 are sensitivity axes
M_DEFAULT = 0.30        # TODO: replace with the measured margin range
H_DEFAULT = 1.0         # operator's choice; the objective's h > 0 clause


def geometric_pmf(rho, tau):
    return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])


def make_e_fast(n_unsafe=37, tau=300, rho=0.02):
    """Measured fast rail: f_i ~ 0 (stress ceiling 1e-6), p_t0 = 0.005 at
    submission, tick 2 s.  tau = 300 ticks is the ten-minute
    authentication window; rho TODO pending response measurements."""
    f = np.full(n_unsafe, 1e-6)
    f[0] = 0.005
    return Channel(f=f, m=M_DEFAULT, h=H_DEFAULT, C=C_DOLLAR,
                   cw=CW_PER_2S, tau=tau, pmf_h=geometric_pmf(rho, tau))


def make_e_slow(depth=8, f0=0.06, decay=0.5, tau=15, rho=0.15):
    """Slow rail with a depth-decaying hazard chain, reconstructed from a
    historical reorg-depth distribution.  Until that reconstruction
    lands, the fallback shape is f0 = 0.06 with decay 0.5.  tau, rho
    TODO pending response measurements."""
    f = f0 * decay ** np.arange(depth)
    return Channel(f=f, m=M_DEFAULT, h=H_DEFAULT, C=C_DOLLAR,
                   cw=CW_PER_2S, tau=tau, pmf_h=geometric_pmf(rho, tau))


# E-outage: two-state regime-switching Markov settlement (normal/outage);
# the model itself lives in outage.py.  Switch rates and durations
# calibrate to measured frequencies (2 events in 30 days, 0.5-1.5 h,
# matching the public incident record).
OUTAGE_CALIBRATION = dict(
    events_per_30d=2.0,
    duration_hours=(0.5, 1.5),
    tick_seconds=60,            # TODO: confirm the coarse tick
    p_enter_per_tick=2.0 / (30 * 24 * 60),
    mean_duration_ticks=60.0,   # 1 h at 60 s ticks
)
