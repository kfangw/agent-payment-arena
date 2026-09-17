#!/usr/bin/env python3
"""Commitment premium at the declared constants, integrated over the F2 flow.

For each rail, the flexible model of commitment_premium.py (a server free to
abandon a pending query to grant or reject) is compared with the committed
model at stage 0, and the episode-level premium V_flex - V_committed is
integrated by quadrature over the F2 suspicion law, Beta(1.6, 6.4), and the
exposure law, lognormal with median 30 and log s.d. 1 truncated to [0.5, 2000].
Reported in dollars per payment and in basis points of the declared mean
exposure 49.50.  Only the F2 law is integrated: the bimodal flows F1 and F3
put almost no mass where verifying is chosen, so their premium is small, but
it is not computed here.
"""
import sys, os, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commitment_premium as CP

Q = 0.75
MEAN_EXPOSURE = 49.50
RAILS = {
    'E-fast':   ([0.005] + [1e-6] * 40,              dict(m=0.35, h=1.0, C=0.5, cw=2e-4, tau=300)),
    'E-slow':   ([0.06 * 0.5 ** i for i in range(8)], dict(m=0.35, h=1.0, C=0.5, cw=2e-4, tau=300)),
    'E-outage': ([0.005] + [3e-5] * 17,              dict(m=0.35, h=1.0, C=0.5, cw=6e-3, tau=10)),
}

def grids(n_pi=81, n_v=81):
    # Beta(1.6, 6.4): mean 0.20, concentration 8
    pe = np.linspace(0, 1, n_pi + 1)
    pw = np.diff(stats.beta.cdf(pe, 1.6, 6.4)); pc = 0.5 * (pe[1:] + pe[:-1])
    ve = np.exp(np.linspace(np.log(0.5), np.log(2000), n_v + 1))
    ln = stats.lognorm(s=1.0, scale=30.0)
    vw = np.diff(ln.cdf(ve)); vw /= vw.sum(); vc = np.sqrt(ve[1:] * ve[:-1])
    return pc, pw, vc, vw

def main():
    pc, pw, vc, vw = grids()
    for name, (f, p) in RAILS.items():
        p = dict(p); p['rho'] = 1 - (1 - Q) ** (1 / p['tau'])
        total = 0.0; pos = 0.0
        for v, wv in zip(vc, vw):
            for pi, wp in zip(pc, pw):
                Vc, Vf, *_ = CP.episode_values(f, v, pi, p)
                d = float(Vf[0] - Vc[0])
                total += wv * wp * d; pos += wv * wp * (d > 1e-9)
        print(f"{name}: premium ${total:.3f} per payment = {total / MEAN_EXPOSURE * 1e4:.0f} bp of mean exposure; "
              f"flow mass with a positive premium {pos:.2f}")

if __name__ == '__main__':
    main()
