"""Acceptance test: five reference environments with known loss bounds.

Each fixed rule has an environment constructed against it, in which the
rule's per-payment loss against the optimum admits a closed-form lower
bound.  The harness must reproduce every gap at or above its bound; a
shortfall beyond the Monte Carlo margin is a defect of this code, not
of the bound, which makes this the gate on the simulator itself.

Run:  python -m duel.acceptance
"""
from __future__ import annotations

import numpy as np

from .core import GRANT, REJECT, VERIFY, WAIT, action_values, sigma_list
from .flows import Flow, LogNormalV
from .policies import make_family_C
from .simulate import Channel, draw_batch, replay


def geometric_pmf(rho, tau):
    return np.array([rho * (1 - rho) ** (s - 1) for s in range(1, tau + 1)])


class PointD:
    """Point-mass or two-point true-risk distribution."""
    def __init__(self, points, weights):
        self.points = np.asarray(points, float)
        self.weights = np.asarray(weights, float) / np.sum(weights)
    def sample(self, n, rng):
        idx = rng.choice(len(self.points), size=n, p=self.weights)
        return np.clip(self.points[idx], 1e-12, 1 - 1e-12)


def fixed_v(v):
    return LogNormalV(v, 1e-12, v, v)


class ConstPol:
    def __init__(self, a):
        self.a = a
    def __call__(self, s, v, pi):
        return self.a


class WaitGrantPol:
    def __init__(self, FIN):
        self.FIN = FIN
    def __call__(self, s, v, pi):
        return GRANT if s >= self.FIN else WAIT


class OptTablePol:
    """Exact optimal policy read by the payment's own pi column."""
    def __init__(self, lab_by_stage):
        self.lab = lab_by_stage
        self.idx = 0
    def __call__(self, s, v, pi):
        return int(self.lab[s, self.idx])


def opt_realized(ch, d, rho):
    """Replay each payment under its exact optimal policy (true pi)."""
    from .core import best_action
    lab, out = best_action(ch.f, 1.0, d.pi0, ch.params(rho=rho))
    # labels are v-dependent; recompute per distinct v (here v is fixed
    # per case, so one call at the common v suffices)
    v0 = float(d.v[0])
    lab, out = best_action(ch.f, v0, d.pi0, ch.params(rho=rho))
    sig = sigma_list(ch.f)
    ex_unit = np.maximum(sig * (1 + ch.m) - 1.0, 0.0)
    pol = OptTablePol(lab)
    real = np.zeros(len(d))
    for k in range(len(d)):
        pol.idx = k
        view = _View(d, k)
        real[k] = replay(ch, view, pol, ex_unit)[0]
    return real


class _View:
    def __init__(self, d, k):
        for name in ('v', 'p_true', 'theta', 'pi0', 'fail_at', 't_ans'):
            setattr(self, name, getattr(d, name)[k:k + 1])
    def __len__(self):
        return 1


def reference_suite(n=120_000, seed=20260823):
    """Returns a list of (name, gap_realized, bound, se, passed)."""
    rng = np.random.default_rng(seed)
    m, h, C = 0.35, 1.0, 0.5
    tau, rho = 300, 1.0 - 0.25 ** (1.0 / 300)     # q = 0.75 in 10 min
    q = 0.75
    pmf = geometric_pmf(rho, tau)
    results = []

    def run_case(name, ch, flow, rule_pol, bound):
        d = draw_batch(ch, flow, n, rng)
        sig = sigma_list(ch.f)
        ex_unit = np.maximum(sig * (1 + ch.m) - 1.0, 0.0)
        r_rule = replay(ch, d, rule_pol, ex_unit)
        r_opt = opt_realized(ch, d, rho)
        gap = r_opt - r_rule
        se = gap.std(ddof=1) / np.sqrt(n)
        passed = gap.mean() >= bound - 3 * se
        results.append((name, float(gap.mean()), float(bound), float(se), passed))

    # 1. grant everything / high-suspicion flow
    f1 = np.array([0.06, 0.03, 0.015])
    ch1 = Channel(f=f1, m=m, h=h, C=C, cw=2e-4, tau=tau, pmf_h=pmf)
    sig1 = sigma_list(f1)[0]
    v = 30.0
    run_case("grant-all", ch1, Flow('w1', PointD([1.0], [1]), fixed_v(v)),
             ConstPol(GRANT), v * (1 - sig1 * (1 - h)))

    # 2. reject everything / trusted flow on a rail with sig(0)(1+m) > 1
    f2 = np.array([0.005] + [1e-6] * 39)
    ch2 = Channel(f=f2, m=m, h=h, C=C, cw=2e-4, tau=tau, pmf_h=pmf)
    sig2 = sigma_list(f2)[0]
    run_case("reject-all", ch2, Flow('w2', PointD([0.0], [1]), fixed_v(v)),
             ConstPol(REJECT), v * (sig2 * (1 + m) - 1))

    # 3. verify everything / trusted flow of small payments v < C/(qm)
    v3 = 1.0
    assert v3 < C / (q * m)
    ch3 = ch2
    o3 = action_values(ch3.f, v3, np.array([0.0]), ch3.params(rho=rho))
    D0 = -(o3['W'][0][0] + C - (1 - 0.0) * o3['A'][0] * 1.0)  # recover D(0)
    D0 = o3['D'][0]
    run_case("verify-all", ch3, Flow('w3', PointD([0.0], [1]), fixed_v(v3)),
             ConstPol(VERIFY), C + D0 - q * v3 * m)

    # 4. amount threshold (best suspicion-blind policy, waiting included)
    #    equal amounts, half pi=0 half pi=1, cw >= f_i at every stage
    f4 = np.array([1e-6] * 40)
    cw4 = 2e-3
    ch4 = Channel(f=f4, m=m, h=h, C=C, cw=cw4, tau=tau, pmf_h=pmf)
    d4 = draw_batch(ch4, Flow('w4', PointD([0.0, 1.0], [1, 1]), fixed_v(v)),
                    n, rng)
    sig4 = sigma_list(f4)
    ex4 = np.maximum(sig4 * (1 + m) - 1.0, 0.0)
    blind = {a: replay(ch4, d4, ConstPol(a), ex4).mean()
             for a in (GRANT, REJECT, VERIFY)}
    blind[WAIT] = replay(ch4, d4, WaitGrantPol(ch4.N + 1), ex4).mean()
    r_opt4 = opt_realized(ch4, d4, rho)
    extreme_grant = v * (1 - sig4[0] * (1 - h))
    extreme_reject = v * (sig4[0] * (1 + m) - 1)
    bound4 = min(C, 0.5 * min(extreme_grant, extreme_reject))
    gap4 = r_opt4.mean() - max(blind.values())
    se4 = r_opt4.std(ddof=1) / np.sqrt(n)
    results.append(("amount-blind", float(gap4), float(bound4), float(se4),
                    gap4 >= bound4 - 3 * se4))

    # 5. always wait, then grant / trusted flow, cw > f_i at every stage
    ch5 = ch4
    surv = np.cumprod(np.concatenate([[1.0], 1 - f4[:-1]]))
    bound5 = v * float(np.sum(surv * (cw4 - f4)))
    run_case("always-wait", ch5, Flow('w5', PointD([0.0], [1]), fixed_v(v)),
             WaitGrantPol(ch5.N + 1), bound5)
    return results


def main():
    print("Reference-bound suite (gap must clear its closed-form bound)")
    ok = True
    for name, gap, bound, se, passed in reference_suite():
        ok &= passed
        print(f"  {name:12s} gap {gap:+9.5f}  bound {bound:+9.5f}  "
              f"se {se:.5f}  {'PASS' if passed else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
