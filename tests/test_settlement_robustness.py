"""Cross-model and accounting checks independent of threshold tuning outcomes."""

from dataclasses import replace
from pathlib import Path

import pytest

from arena.experiments.settlement_learning.model import PublicState, Request, Setting
from arena.experiments.settlement_learning.robustness import (
    Evaluator,
    Planner,
    Threshold,
    candidates,
    tune,
)
from arena.experiments.settlement_learning.run_robustness import design
from arena.experiments.settlement_learning.solver import MODES, Solver


def environment(hazards=(0.2, 0.03), **kwargs):
    return Setting(
        schedule=(((1.0, Request(0.05, hazards)),), ((1.0, Request(1.0)),)),
        wait_cost=0.002,
        response_rate=0.7,
        deadline=3,
        **kwargs,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("hazards", [(), (0.2, 0.03)])
def test_original_values_and_metrics_preserved(mode, hazards):
    s = environment(hazards)
    new = Evaluator(s, Planner(s, mode)).report()
    assert new["reward"] == pytest.approx(Solver(s, mode).evaluate(), abs=1e-11)
    assert new["normal_requests"] == pytest.approx(2 * (1 - s.risk(0, 0)))
    assert 0 <= new["normal_nonrelease_rate"] <= 1


@pytest.mark.parametrize("mode", ["no_wait", "no_verify"])
@pytest.mark.parametrize("accounting", ["basic", "additive"])
def test_recomputed_ablation_and_evaluator_agree(mode, accounting):
    s = environment()
    solver = Planner(s, mode, accounting)
    assert solver.future(0, 0, 0) == pytest.approx(
        Evaluator(s, solver, accounting).future()[0]
    )
    assert (
        solver.future(0, 0, 0)
        <= Planner(s, accounting=accounting).future(0, 0, 0) + 1e-12
    )
    for i in range(3):
        assert solver.action(PublicState(0, s.schedule[0][0][1], i)) != mode[3:]


def test_actual_posterior_not_assumed_posterior():
    actual = environment((), prior=0.1)
    assumed = replace(actual, prior=0.9)
    policy = Threshold(assumed, "B1", theta=1e5)
    expected = sum(r.amount for dist in actual.schedule for _, r in dist) * (
        1 - 2 * actual.risk(0, 0)
    )
    assert Evaluator(actual, policy).future()[0] == pytest.approx(expected)
    assert Evaluator(actual, policy).future()[0] != pytest.approx(
        Evaluator(assumed, policy).future()[0]
    )


def test_normal_answer_keeps_assumed_release_decision():
    actual = Setting(
        schedule=(((1.0, Request(1.0, (0.9,))),),), response_rate=1, deadline=1
    )
    assumed = replace(actual, schedule=(((1.0, Request(1.0, (0.0,))),),))
    policy = Planner(assumed, "query_first")
    # A certain normal answer still causes an unpaid-risk release under the optimistic policy.
    expected = -0.1 + 0.5 * (0.1 * 2 - 1)
    assert Evaluator(actual, policy).future()[0] == pytest.approx(expected)
    assert Evaluator(actual, policy).report()["unpaid_release_amount"] == pytest.approx(
        0.45
    )
    assert Evaluator(actual, Planner(actual, "query_first")).future()[
        0
    ] == pytest.approx(-0.1)


def test_response_precedes_certain_settlement_failure():
    actual = Setting(
        schedule=(((1.0, Request(1.0, (1.0,))),),), response_rate=0.4, deadline=2
    )
    policy = Planner(actual, "query_first")
    assert Evaluator(actual, policy).window(0, 0, 0)[0] == pytest.approx(0.4)


def test_additive_fixed_policy_loss_uses_actual_unpaid_misuse():
    actual = environment((0.2,))
    policy = Threshold(actual, "B1", theta=100)
    basic = Evaluator(actual, policy).future()[0]
    additive = Evaluator(actual, policy, "additive").future()[0]
    assert basic - additive == pytest.approx(
        0.05 * 0.2 * actual.risk(0, 0) * actual.harm
    )


def test_failure_preserves_future_requests_and_metrics():
    actual = environment((1.0,))
    policy = Threshold(actual, "B4", lower=1, upper=1, watch=1)
    result = Evaluator(actual, policy).report()
    assert result["normal_requests"] == pytest.approx(1)
    assert result["normal_nonrelease"] == pytest.approx(0.5)
    assert result["misuse_release_amount"] == pytest.approx(0.5)


@pytest.mark.parametrize("family", ["B1", "B2", "B3", "B4"])
def test_grouped_search_keeps_entire_grid_and_matches_exhaustive(family):
    s = environment()
    groups = list(candidates(s, family, True, 5))
    expected = 5 if family in ("B1", "B2") else 15 * (3 if family == "B4" else 1)
    assert sum(len(parameters) for _, parameters in groups) == expected
    brute = []
    for representative, parameters in groups:
        for parameters_item in parameters:
            policy = Threshold(s, family, True, **parameters_item)
            value = Evaluator(s, policy).future()[0]
            assert value == pytest.approx(Evaluator(s, representative).future()[0])
            brute.append(value)
    _, fit = tune(s, family, True, 5)
    assert fit["value"] == pytest.approx(max(brute))
    assert fit["tied_parameters"]


def test_all_threshold_ties_recorded():
    s = environment()
    # Amounts are below multiple thresholds; all grant-all candidates tie in this example.
    _, fit = tune(s, "B2", False, 21)
    assert fit["grid_candidates"] == 21
    assert len(fit["tied_parameters"]) >= 1


def test_invalid_public_support_fails_before_evaluation():
    s = environment()
    other = replace(s, schedule=(((1.0, Request(2.0, (0.2, 0.03))),), s.schedule[1]))
    with pytest.raises(ValueError, match="support"):
        Evaluator(s, Planner(other))


def test_design_contains_18_bases_and_deduplicates_no_hazard_cases():
    cases = design(Path("configs/settlement_learning"))
    assert len({c["base"] for c in cases}) == 18
    assert len(cases) == 168
    assert sum("additive_refit" in c["tags"] for c in cases) == 18
    assert sum("additive_held" in c["tags"] for c in cases) == 18
    for c in cases:
        if "-final-" in c["id"] and "matched" in c["tags"]:
            assert {"hazard05", "hazard2"} <= set(c["tags"])
