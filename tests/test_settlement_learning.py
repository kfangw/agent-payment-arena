"""Deterministic checks and a separate tick-tree oracle for repeated decisions."""

from collections.abc import Callable
from dataclasses import fields, replace
from functools import cache

import pytest

from arena.experiments.settlement.core import action_values
from arena.experiments.settlement_learning.model import PublicState, Request, Setting
from arena.experiments.settlement_learning.simulate import History, replay, uniform
from arena.experiments.settlement_learning.solver import ACTIONS, MODES, Solver


def setting(
    hazards: tuple[float, ...] = (),
    *,
    prior: float = 0.5,
    low: float = 0.01,
    high: float = 0.99,
    query_cost: float = 0.1,
    wait_cost: float = 0.0,
    response_rate: float = 1.0,
    deadline: int = 2,
) -> Setting:
    return Setting(
        schedule=(((1.0, Request(0.05, hazards)),), ((1.0, Request(1.0)),)),
        prior=prior,
        low=low,
        high=high,
        query_cost=query_cost,
        wait_cost=wait_cost,
        response_rate=response_rate,
        deadline=deadline,
    )


@pytest.mark.parametrize("hazards", [(), (0.1,), (0.2, 0.03)])
@pytest.mark.parametrize("rate", [0.0, 0.7, 1.0])
def test_last_request_matches_reference_actions(hazards: tuple[float, ...], rate: float) -> None:
    s = replace(setting(response_rate=rate), schedule=(((1.0, Request(0.4, hazards)),),))
    solver = Solver(s)
    request = s.schedule[0][0][1]
    p = {
        "m": s.margin,
        "h": s.harm,
        "C": s.query_cost,
        "cw": s.wait_cost,
        "rho": s.response_rate,
        "tau": s.deadline,
    }
    # action_values lives in the settlement package, which the type checker excludes.
    reference = action_values(hazards, request.amount, [s.risk(0, 0)], p)  # type: ignore[no-untyped-call]
    for stage in range(len(hazards) + 1):
        expected = [float(reference[key][stage, 0]) for key in ("G", "R", "W", "Wait")]
        assert solver.values(0, request, stage, 0, 0) == pytest.approx(expected)
        assert (
            solver.action(PublicState(0, request, stage))
            == ACTIONS[max(range(4), key=expected.__getitem__)]
        )


def test_eta_matches_independent_survival_sum() -> None:
    s = setting((0.2, 0.3), response_rate=0.4, deadline=4)
    r = s.schedule[0][0][1]
    solver = Solver(s)
    for stage in range(3):
        survival = 1.0
        eta = 0.0
        for t in range(1, 5):
            eta += 0.4 * 0.6 ** (t - 1) * survival
            index = stage + t - 1
            survival *= 1 - (r.hazards[index] if index < 2 else 0)
        assert solver.window(r)[3][stage] == pytest.approx(eta)
    one = Solver(replace(s, deadline=1))
    assert one.window(r)[3][0] == pytest.approx(0.4)


def test_published_hand_calculation_and_uncertain_variant() -> None:
    assert Solver(setting()).evaluate() == pytest.approx(0.4052)
    assert Solver(setting(), "myopic").evaluate() == pytest.approx(0.4)
    s = setting((0.01,), response_rate=0.99, wait_cost=0.0001)
    optimal = Solver(s)
    assert Solver(s, "query_first").evaluate() == pytest.approx(0.404678985651)
    assert Solver(s, "myopic").evaluate() == pytest.approx(0.399849)
    assert optimal.evaluate() == pytest.approx(0.405126060751)
    assert optimal.action(PublicState(0, s.schedule[0][0][1], 0)) == "wait"


@pytest.mark.parametrize("prior", [0.0, 1.0])
def test_known_type_has_no_learning_bonus(prior: float) -> None:
    s = setting((0.1,), prior=prior, response_rate=0.6)
    assert Solver(s).evaluate() == pytest.approx(Solver(s, "myopic").evaluate())


def test_no_answer_cannot_create_a_label() -> None:
    s = setting(response_rate=0)
    solver = Solver(s)
    assert solver.window(s.schedule[0][0][1])[3][0] == 0
    assert solver.evaluate() == pytest.approx(Solver(s, "myopic").evaluate())
    _, events = replay(Solver(s, "query_first"), 7, "e0", 0)
    assert all(e["label"] is None for e in events if e["event"] == "close")


def test_observation_contract_is_idempotent_and_public() -> None:
    history = History()
    history.close(0, 0)
    history.close(0, 1)
    history.close(1, None)
    history.close(1, 1)
    assert (history.misuse, history.normal) == (0, 1)
    names = {f.name for f in fields(PublicState)}
    assert names == {
        "request_index",
        "request",
        "stage",
        "misuse_labels",
        "normal_labels",
    }
    with pytest.raises(ValueError):
        Solver(setting()).action(PublicState(0, Request(1), 0, 1, 0))
    _, events = replay(Solver(setting(), "query_first"), 9, "e0", 1)
    forbidden = {"latent_type", "latent_misuse", "reward", "unpaid_loss", "risk"}
    assert all(not forbidden.intersection(e) for e in events)


def test_label_survives_when_conditional_release_is_unprofitable() -> None:
    s = setting((1.0,), response_rate=1.0, prior=0.0)
    _, events = replay(Solver(s, "query_first"), 1, "e0", 0)
    first = next(e for e in events if e["event"] == "close")
    assert first["label"] is not None
    assert not first["released"]


def test_named_streams_reproduce_and_split() -> None:
    assert uniform(1, "pilot", 2, 3, "intent") == uniform(1, "pilot", 2, 3, "intent")
    assert uniform(1, "pilot", 2, 3, "intent") != uniform(1, "evaluation", 2, 3, "intent")
    solver = Solver(setting())
    assert replay(solver, 4, "e0", 1) == replay(solver, 4, "e0", 1)


def tick_tree_value(s: Setting) -> float:
    """Independent full-value recursion, without A/D/eta or reduced values."""

    @cache
    def future(n: int, b: float) -> float:
        if n == len(s.schedule):
            return 0.0
        return sum(w * stage_value(n, r, 0, b) for w, r in s.schedule[n])

    def risk(b: float) -> float:
        return (1 - b) * s.low + b * s.high

    def posterior(b: float, label: int) -> float:
        return b * (s.high if label else 1 - s.high) / (risk(b) if label else 1 - risk(b))

    def grant(r: Request, stage: int, pi: float) -> float:
        survival = 1.0
        for hazard in r.hazards[stage:]:
            survival *= 1 - hazard
        return r.amount * (survival * (1 + (1 - pi) * s.margin - pi * s.harm) - 1)

    @cache
    def query(n: int, r: Request, stage: int, b: float, ticks: int) -> float:
        unchanged = future(n + 1, b)
        if ticks == 0:
            return unchanged
        pi = risk(b)
        answered = pi * future(n + 1, posterior(b, 1)) + (1 - pi) * (
            max(grant(r, stage, 0), 0) + future(n + 1, posterior(b, 0))
        )
        failure = r.hazards[stage] if stage < len(r.hazards) else 0
        next_stage = min(stage + 1, len(r.hazards))
        return float(
            -s.wait_cost * r.amount
            + s.response_rate * answered
            + (1 - s.response_rate)
            * (failure * unchanged + (1 - failure) * query(n, r, next_stage, b, ticks - 1))
        )

    @cache
    def stage_value(n: int, r: Request, stage: int, b: float) -> float:
        unchanged = future(n + 1, b)
        values = [
            unchanged,
            grant(r, stage, risk(b)) + unchanged,
            -s.query_cost + query(n, r, stage, b, s.deadline),
        ]
        if stage < len(r.hazards):
            values.append(
                -s.wait_cost * r.amount
                + r.hazards[stage] * unchanged
                + (1 - r.hazards[stage]) * stage_value(n, r, stage + 1, b)
            )
        return float(max(values))

    return future(0, s.prior)


@pytest.mark.parametrize("rate", [0.0, 0.45, 1.0])
def test_three_requests_against_tick_tree_and_baselines(rate: float) -> None:
    s = Setting(
        schedule=(
            ((0.4, Request(0.03, (0.3,))), (0.6, Request(0.3))),
            ((1.0, Request(0.7, (0.04, 0.1))),),
            ((1.0, Request(1.2)),),
        ),
        response_rate=rate,
        wait_cost=0.002,
    )
    optimum = Solver(s)
    assert optimum.future(0, 0, 0) == pytest.approx(tick_tree_value(s), abs=1e-11)
    assert optimum.evaluate() == pytest.approx(optimum.future(0, 0, 0), abs=1e-11)
    for mode in MODES:
        assert Solver(s, mode).evaluate() <= optimum.evaluate() + 1e-10


@pytest.mark.parametrize(
    "build",
    [
        lambda: setting(response_rate=-0.1),
        lambda: setting(deadline=0),
        lambda: setting(low=0.9, high=0.1),
        lambda: setting(query_cost=-1),
        lambda: setting(wait_cost=float("nan")),
    ],
)
def test_invalid_settings_rejected(build: Callable[[], Setting]) -> None:
    with pytest.raises(ValueError):
        build()
