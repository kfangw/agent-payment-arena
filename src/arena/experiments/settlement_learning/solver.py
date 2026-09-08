"""Exact finite recursion and evaluation under the true observation kernel."""

from __future__ import annotations

from functools import cache

from arena.experiments.settlement.core import sigma_list, window_AD

from .model import PublicState, Request, Setting

ACTIONS = ("grant", "reject", "verify", "wait")
MODES = ("optimal", "fixed", "myopic", "query_first", "q_assumed")


class Solver:
    """Compile decisions from public counts; evaluate without label leakage."""

    def __init__(self, setting: Setting, mode: str = "optimal") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown policy: {mode}")
        self.setting = setting
        self.mode = mode
        # Caches belong to this solver, not a class-wide registry of instances.
        self.window = cache(self.window)
        self.future = cache(self.future)
        self.values = cache(self.values)
        self.evaluate = cache(self.evaluate)
        self._evaluate_stage = cache(self._evaluate_stage)

    def window(self, request: Request) -> tuple[tuple[float, ...], ...]:
        """Reuse single-request execution and cost calculations."""
        s = self.setting
        sigma = sigma_list(request.hazards)
        execution, delay = window_AD(
            request.hazards,
            sigma,
            request.amount,
            s.margin,
            s.wait_cost,
            s.response_rate,
            s.deadline,
        )
        final = len(request.hazards)
        eta = [0.0] * (final + 1)
        for _ in range(s.deadline):
            previous = eta
            eta = [
                s.response_rate
                + (1 - s.response_rate)
                * (1 - (request.hazards[i] if i < final else 0))
                * previous[min(i + 1, final)]
                for i in range(final + 1)
            ]
        return (
            tuple(map(float, sigma)),
            tuple(map(float, execution)),
            tuple(map(float, delay)),
            tuple(eta),
        )

    def current(self, request: Request, stage: int, bad: int, good: int) -> tuple[float, float]:
        """Current grant and query expected payoffs at the actual belief."""
        s = self.setting
        pi = s.risk(bad, good)
        sigma, execution, delay, _ = self.window(request)
        grant = request.amount * (sigma[stage] * (1 + (1 - pi) * s.margin - pi * s.harm) - 1)
        query = -s.query_cost - delay[stage] + (1 - pi) * execution[stage]
        return grant, query

    def future(self, n: int, bad: int, good: int) -> float:
        """Planner's future value; q_assumed deliberately uses a wrong kernel."""
        if n == len(self.setting.schedule):
            return 0.0
        return self.future(n + 1, bad, good) + sum(
            weight * max(self.values(n, request, 0, bad, good))
            for weight, request in self.setting.schedule[n]
        )

    def values(
        self, n: int, request: Request, stage: int, bad: int, good: int
    ) -> tuple[float, ...]:
        """Reduced action values in canonical tie order."""
        s = self.setting
        grant, query = self.current(request, stage, bad, good)
        if self.mode in ("optimal", "q_assumed"):
            pi = s.risk(bad, good)
            information = (
                pi * self.future(n + 1, bad + 1, good)
                + (1 - pi) * self.future(n + 1, bad, good + 1)
                - self.future(n + 1, bad, good)
            )
            eta = (
                s.response_probability
                if self.mode == "q_assumed"
                else self.window(request)[3][stage]
            )
            query += eta * information
        wait = float("-inf")
        if stage < len(request.hazards):
            wait = -s.wait_cost * request.amount + (1 - request.hazards[stage]) * max(
                self.values(n, request, stage + 1, bad, good)
            )
        return grant, 0.0, query, wait

    def action(self, state: PublicState) -> str:
        """Choose using only public information and fixed design parameters."""
        if not 0 <= state.request_index < len(self.setting.schedule):
            raise ValueError("request index outside schedule")
        if not 0 <= state.stage <= len(state.request.hazards):
            raise ValueError("stage outside chain")
        if min(state.misuse_labels, state.normal_labels) < 0:
            raise ValueError("negative label count")
        if state.misuse_labels + state.normal_labels > state.request_index:
            raise ValueError("more observations than previous requests")
        if self.mode == "query_first" and state.request_index == 0:
            return "verify"
        bad, good = state.misuse_labels, state.normal_labels
        if self.mode == "fixed":
            bad = good = 0
        values = self.values(state.request_index, state.request, state.stage, bad, good)
        return ACTIONS[max(range(4), key=values.__getitem__)]

    def evaluate(self, n: int = 0, bad: int = 0, good: int = 0) -> float:
        """Expected deployed policy reward under the true kernel for every mode."""
        if n == len(self.setting.schedule):
            return 0.0
        return sum(
            w * self._evaluate_stage(n, r, 0, bad, good) for w, r in self.setting.schedule[n]
        )

    def _evaluate_stage(self, n: int, request: Request, stage: int, bad: int, good: int) -> float:
        follow = self.evaluate(n + 1, bad, good)
        action = self.action(PublicState(n, request, stage, bad, good))
        grant, query = self.current(request, stage, bad, good)
        if action == "grant":
            return grant + follow
        if action == "reject":
            return follow
        if action == "wait":
            failure = request.hazards[stage]
            return (
                -self.setting.wait_cost * request.amount
                + failure * follow
                + (1 - failure) * self._evaluate_stage(n, request, stage + 1, bad, good)
            )
        pi = self.setting.risk(bad, good)
        eta = self.window(request)[3][stage]
        return (
            query
            + (1 - eta) * follow
            + eta
            * (
                pi * self.evaluate(n + 1, bad + 1, good)
                + (1 - pi) * self.evaluate(n + 1, bad, good + 1)
            )
        )
