"""Terminal accounting regressions and independent payoff checks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arena.experiments.settlement.core import GRANT, VERIFY
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (
    OutageEnv,
    draw_outage_batch,
    replay_outage,
    survival,
    value_labels,
    window_AD,
)
from arena.experiments.settlement.recovery import replay, rule, settlement


def example() -> OutageEnv:
    return OutageEnv(
        f=np.zeros(2), m=0.35, h=1.0, C=0.5, cw=0.006, tau=1, H=1, rho=0.75, p01=0.0, p10=1.0
    )


def test_executed_transfer_continues_but_unexecuted_authorization_expires() -> None:
    env = example()
    d = SimpleNamespace(paths=np.zeros((1, 3), dtype=int), u_stage=np.ones((1, 2)))
    assert settlement(env, d, 0, 0, 0, 0.0, 0.2) == (False, False, False)
    assert settlement(env, d, 0, 0, 0, 1.0, 0.2) == (True, True, False)
    d.paths[:] = 1
    assert settlement(env, d, 0, 0, 0, 1.0, 0.2) == (False, False, True)


def test_continuation_still_pays_remaining_failure_risk() -> None:
    env = replace(example(), f=np.array([0.0, 0.3]))
    assert survival(env, recovery=1)[0, 1, 0] == pytest.approx(0.7)
    assert survival(env, recovery=0.5)[0, 1, 0] == pytest.approx(0.35)
    d = SimpleNamespace(paths=np.zeros((1, 3), dtype=int), u_stage=np.array([[1.0, 0.1]]))
    assert settlement(env, d, 0, 0, 0, 1.0, 0.2)[0] is False


def test_uncommitted_cutoff_does_not_acquire_continuation_release_value() -> None:
    env = example()
    _, values, _ = value_labels(env, 1.0, np.array([0.0, 0.5, 1.0]), recovery=1.0)
    np.testing.assert_array_equal(values[:2, 0], 0.0)
    assert values[0, 1, 0, 0] == pytest.approx(0.35)


@pytest.mark.parametrize("action", [GRANT, VERIFY])
def test_zero_recovery_matches_legacy_replay(action: int) -> None:
    _, env, _ = envs_for("mid")["E-outage"]
    d = draw_outage_batch(
        env, make_flows()["F2"], 100, np.random.default_rng(3821), payments_per_episode=50
    )
    _, _, ex = window_AD(env, survival(env))
    for watch in (0, 3, 18):
        policy = rule(env, watch, force=action)
        expected = replay_outage(env, d, policy, ex)
        actual = replay(env, d, policy, ex, 0.0, np.zeros(len(d)))
        np.testing.assert_array_equal(actual[:, 0], expected)


def test_simulated_grant_matches_analytic_continuation_value() -> None:
    env = replace(example(), f=np.array([0.0, 0.3]))
    n = 20000
    rng = np.random.default_rng(555)
    coins = rng.random(n)
    d = SimpleNamespace(paths=np.zeros((n, 3), dtype=int), u_stage=rng.random((n, 2)))
    paid = np.array([settlement(env, d, k, 0, 0, 0.5, coins[k])[0] for k in range(n)])
    assert paid.mean() == pytest.approx(survival(env, recovery=0.5)[0, 1, 0], abs=0.012)


def test_invalid_recovery_rejected() -> None:
    with pytest.raises(ValueError):
        survival(example(), recovery=1.1)
    with pytest.raises(ValueError):
        survival(replace(example(), p10=0.0), recovery=1.0)


def test_halt_minutes_sets_mean_duration_and_keeps_start_rate() -> None:
    from arena.experiments.settlement.recovery import with_halt_minutes

    _, base, _ = envs_for('mid')['E-outage']
    assert with_halt_minutes(base, None) is base
    short = with_halt_minutes(base, 10)
    assert short.p10 == pytest.approx(1 / 10)
    assert short.p01 == base.p01
    assert short.stationary_outage < base.stationary_outage


def test_keep_halt_share_rescales_start_rate() -> None:
    from arena.experiments.settlement.recovery import with_halt_minutes

    _, base, _ = envs_for('mid')['E-outage']
    short = with_halt_minutes(base, 10, keep_halt_share=True)
    assert short.p10 == pytest.approx(1 / 10)
    assert short.stationary_outage == pytest.approx(base.stationary_outage)
    assert survival(short, recovery=0)[0, base.H, 1] > survival(base, recovery=0)[0, base.H, 1]


def test_invalid_halt_minutes_rejected() -> None:
    from arena.experiments.settlement.recovery import with_halt_minutes

    _, base, _ = envs_for('mid')['E-outage']
    with pytest.raises(ValueError):
        with_halt_minutes(base, 0)
    with pytest.raises(ValueError):
        with_halt_minutes(base, 0.5)
