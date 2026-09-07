"""The instrumented replays must reproduce the payoff replays exactly."""

from dataclasses import replace

import numpy as np

from arena.experiments.settlement.core import rho_hat_from_q, sigma_list
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (
    compile_outage, draw_outage_batch, replay_outage, survival, window_AD,
)
from arena.experiments.settlement.policies import B3, compile_A
from arena.experiments.settlement.run import OB
from arena.experiments.settlement.security_replay import (
    FIELDS, replay_chain, replay_outage_fields, summarize,
)
from arena.experiments.settlement.simulate import draw_batch, replay
from arena.experiments.settlement.watch import OutageWatchBandPolicy, WatchBandPolicy


def test_chain_fields_match_payoff_replay() -> None:
    _, ch, _ = envs_for('mid')['E-slow']
    d = draw_batch(ch, make_flows()['F2'], 3000, np.random.default_rng(3))
    ex = np.maximum(sigma_list(ch.f) * (1 + ch.m) - 1.0, 0.0)
    for pol in (compile_A(ch, 'A', rho=rho_hat_from_q(0.75, ch.tau)), B3(0.1, 0.8),
                WatchBandPolicy(3, 0.1, 0.8)):
        out = replay_chain(ch, d, pol, ex)
        assert np.array_equal(out[:, 0], replay(ch, d, pol, ex))
        s = summarize(out, d.theta)
        assert 0 <= s['release_rate'] <= 1 and s['misuse_exposure'] >= 0
        assert np.all(out[:, 2] <= out[:, 1]) and np.all((out[:, 4] > 0) <= (out[:, 1] > 0))


def test_outage_fields_match_payoff_replay() -> None:
    _, env, _ = envs_for('mid')['E-outage']
    d = draw_outage_batch(env, make_flows()['F2'], 2000, np.random.default_rng(5))
    _, _, ex = window_AD(env, survival(env))
    for pol in (compile_outage(replace(env, rho=rho_hat_from_q(0.75, env.tau)), 'A'),
                OB(B3(0.1, 0.8)), OutageWatchBandPolicy(2, 0.1, 0.8, env.H, env.N)):
        out = replay_outage_fields(env, d, pol, ex)
        assert np.array_equal(out[:, 0], replay_outage(env, d, pol, ex))
        assert out.shape[1] == len(FIELDS)
