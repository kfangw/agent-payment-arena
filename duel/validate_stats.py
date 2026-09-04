"""Acceptance checks for the judgement layer (spec 3.9).

Run:  python -m duel.validate_stats
"""

from __future__ import annotations

import numpy as np

from .stats import boot_ci, holm, perm_p, ratio_mean, units, verdict


def a5_1_coverage() -> None:
    """A5-1: the 95% block bootstrap covers a known mean 0.93..0.97."""
    rng = np.random.default_rng(1)
    n_ep, per, mu, sd = 80, 50, 0.4, 1.0
    counts = np.full(n_ep, per)
    trials, hit = 600, 0
    for t in range(trials):
        # episode diff sums with per-episode mean mu (counts equal)
        ep_mean = mu + sd / np.sqrt(per) * rng.standard_normal(n_ep)
        sums = ep_mean * per
        lo, hi = boot_ci(sums, counts, n_boot=1500, seed=1000 + t)
        if lo <= mu <= hi:
            hit += 1
    cov = hit / trials
    assert 0.93 <= cov <= 0.97, f"coverage {cov:.3f} out of band"
    print(f"A5-1 ok: coverage {cov:.3f}")


def a5_2_block_widens() -> None:
    """A5-2: with within-episode correlation the block interval is wider
    than the payment-unit interval that ignores it."""
    rng = np.random.default_rng(2)
    n_ep, per = 100, 50
    shift = rng.standard_normal(n_ep) * 1.0  # shared per-episode shock
    payments = np.repeat(shift, per) + 0.1 * rng.standard_normal(n_ep * per)
    episodes = np.repeat(np.arange(n_ep), per)
    counts = np.full(n_ep, per)
    sums = np.array([payments[episodes == e].sum() for e in range(n_ep)])
    lo_b, hi_b = boot_ci(sums, counts, seed=3)
    # payment-unit bootstrap: resample individual payments (independence lie)
    rng2 = np.random.default_rng(3)
    n = len(payments)
    idx = rng2.integers(0, n, size=(10_000, n))
    pm = payments[idx].mean(axis=1)
    lo_p, hi_p = float(np.quantile(pm, 0.025)), float(np.quantile(pm, 0.975))
    assert (hi_b - lo_b) > 1.5 * (hi_p - lo_p), (hi_b - lo_b, hi_p - lo_p)
    print(f"A5-2 ok: block width {hi_b - lo_b:.4f} > payment width {hi_p - lo_p:.4f}")


def a5_3_perm_uniform() -> None:
    """A5-3: under the null the permutation p value is uniform (hand KS)."""
    rng = np.random.default_rng(4)
    n_ep, per, m = 40, 50, 300
    counts = np.full(n_ep, per)
    ps = []
    for j in range(m):
        sums = rng.standard_normal(n_ep) * per * 0.02  # mean 0 episodes
        ps.append(perm_p(sums, counts, n_perm=2000, seed=5000 + j))
    ps = np.sort(np.asarray(ps))
    fn = np.arange(1, m + 1) / m
    d = float(np.max(np.abs(fn - ps)))
    crit = 1.36 / np.sqrt(m)
    assert d < crit, f"KS D={d:.3f} exceeds {crit:.3f}"
    print(f"A5-3 ok: KS D={d:.3f} < {crit:.3f} (uniform)")


def a5_4_holm_handcalc() -> None:
    """A5-4: Holm matches a hand-computed vector."""
    got = holm([0.01, 0.02, 0.03, 0.04])
    want = [0.04, 0.06, 0.06, 0.06]
    assert all(abs(g - w) < 1e-12 for g, w in zip(got, want)), got
    # order independence: same multiset, permuted, maps back correctly
    got2 = holm([0.03, 0.01, 0.04, 0.02])
    assert abs(got2[1] - 0.04) < 1e-12 and abs(got2[0] - 0.06) < 1e-12, got2
    print(f"A5-4 ok: holm {got}")


def a5_5_verdict_truth_table() -> None:
    """A5-5: the four verdicts classify their boundary cases."""
    eps = 0.05
    assert verdict(0.1, 0.3, 0.01, eps) == "superior"
    assert verdict(-0.3, -0.1, 0.01, eps) == "inferior"
    assert verdict(-0.02, 0.02, 0.5, eps) == "equivalent"
    assert verdict(-0.1, 0.2, 0.5, eps) == "undetermined"
    # real but sub-margin significance without containment -> undetermined
    assert verdict(0.001, 0.2, 0.2, eps) == "undetermined"
    # positive and significant but not contained -> superior (priority)
    assert verdict(0.001, 0.02, 0.01, eps) == "superior"
    print("A5-5 ok: verdict truth table holds")


def a5_6_units() -> None:
    """A5-6: per-1000 and bp convert through the mean exposure."""
    mean_diff, exposure = 0.005, 50.0
    u = units(mean_diff, exposure)
    assert abs(u["per_1000"] - 5.0) < 1e-12
    assert abs(u["bp"] - 1.0) < 1e-12
    # cross conversion: bp == per_1000 * 10 / exposure
    assert abs(u["bp"] - u["per_1000"] * 10.0 / exposure) < 1e-12
    assert abs(ratio_mean(np.array([5.0]), np.array([1000.0])) - 0.005) < 1e-12
    print("A5-6 ok: per-1000 and bp inter-convert")


def main() -> None:
    """Run every acceptance check; raise on the first failure."""
    a5_1_coverage()
    a5_2_block_widens()
    a5_3_perm_uniform()
    a5_4_holm_handcalc()
    a5_5_verdict_truth_table()
    a5_6_units()
    print("\nS5 statistics: all checks passed")


if __name__ == "__main__":
    main()
