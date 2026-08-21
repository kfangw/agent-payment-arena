"""Acceptance checks for the common-random answer retiming (spec A4-2).

The refactor must keep draw_batch's distribution, and the retiming must
move answer times monotonically as the response bias grows.

Run:  python -m duel.validate_crn
"""
from __future__ import annotations

import numpy as np

from .gate import envs_for
from .outage import draw_outage_batch_crn, retime_answers_geom
from .simulate import draw_batch_crn, retime_answers


def _geom(rho, tau):
    return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])


def draw_batch_distribution_preserved() -> None:
    """The within-deadline answer rate matches the geometric target, so
    the inverse-CDF refactor did not change the distribution."""
    _, ch, rho = envs_for("mid")["E-slow"]
    from .flows import make_flows
    flow = make_flows()["F1"]
    d, u = draw_batch_crn(ch, flow, 200_000, np.random.default_rng(9001))
    assert u.shape == (200_000,)
    q_emp = float((d.t_ans <= ch.tau).mean())
    q_target = float(ch.pmf_h.sum())            # honest-intent within-deadline mass
    # most payments are honest; empirical q close to the honest target
    assert abs(q_emp - q_target) < 0.02, (q_emp, q_target)
    print(f"draw_batch ok: within-deadline rate {q_emp:.3f} ~ target {q_target:.3f}")


def a4_2_chain_monotone() -> None:
    """A4-2: as lambda grows, the fraction of misuse payments whose answer
    time changes grows monotonically with the level spacing."""
    _, ch, rho = envs_for("mid")["E-slow"]
    from .flows import make_flows
    flow = make_flows()["F1"]
    d, u_ans = draw_batch_crn(ch, flow, 200_000, np.random.default_rng(9002))
    tau = ch.tau
    lambdas = [1.0, 1.25, 1.5, 1.75, 2.0]
    base = None
    frac_changed = []
    for lam in lambdas:
        rho_m = min(rho * lam, 1.0)
        pmf_m = _geom(rho_m, tau)
        t = retime_answers(d.theta, u_ans, ch.pmf_h, pmf_m, tau)
        if base is None:
            base = t
            assert np.array_equal(t, d.t_ans), "lambda=1 must reproduce the base"
        else:
            mis = d.theta == 1
            frac_changed.append(float((t[mis] != base[mis]).mean()))
    assert all(x <= y + 1e-9 for x, y in zip(frac_changed, frac_changed[1:])), \
        frac_changed
    # honest answers never move (only misuse intent is perturbed)
    rho_m = min(rho * 2.0, 1.0)
    t2 = retime_answers(d.theta, u_ans, ch.pmf_h, _geom(rho_m, tau), tau)
    hon = d.theta == 0
    assert np.array_equal(t2[hon], d.t_ans[hon]), "honest answers moved"
    print(f"A4-2 chain ok: changed-fraction monotone {['%.3f' % f for f in frac_changed]}")


def a4_2_outage_monotone() -> None:
    """A4-2 for the regime cell: geometric retiming is monotone in lambda
    and identity at lambda = 1."""
    _, env, rho = envs_for("mid")["E-outage"]
    from .flows import make_flows
    flow = make_flows()["F1"]
    d, u_ans = draw_outage_batch_crn(env, flow, 120_000,
                                     np.random.default_rng(9003))
    base = retime_answers_geom(d.theta, u_ans, env.rho, env.rho)
    assert np.array_equal(base, d.t_ans), "lambda=1 must reproduce the base"
    frac = []
    for lam in [1.25, 1.5, 1.75, 2.0]:
        rho_m = min(env.rho * lam, 1.0 - 1e-12)
        t = retime_answers_geom(d.theta, u_ans, env.rho, rho_m)
        mis = d.theta == 1
        frac.append(float((t[mis] != base[mis]).mean()))
        hon = d.theta == 0
        assert np.array_equal(t[hon], base[hon]), "honest answers moved"
    assert all(x <= y + 1e-9 for x, y in zip(frac, frac[1:])), frac
    print(f"A4-2 outage ok: changed-fraction monotone {['%.3f' % f for f in frac]}")


def main() -> None:
    """Run every acceptance check; raise on the first failure."""
    draw_batch_distribution_preserved()
    a4_2_chain_monotone()
    a4_2_outage_monotone()
    print("\nCRN retiming: all checks passed")


if __name__ == "__main__":
    main()
