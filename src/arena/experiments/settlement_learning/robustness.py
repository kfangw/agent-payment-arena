"""Fixed-policy evaluation, action ablations, and tuned finite threshold rules.

Planning and evaluation settings are separate. Observed label counts are public;
the evaluator uses the actual posterior and the policy uses its assumed one.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Protocol, TypedDict

from .model import PublicState, Request, Setting
from .solver import MODES, Solver

METRICS = (
    "reward",
    "queries",
    "normal_requests",
    "normal_nonrelease",
    "misuse_release_amount",
    "unpaid_release_amount",
)
ZERO = (0.0,) * len(METRICS)


def add(*vectors: tuple[float, ...]) -> tuple[float, ...]:
    """Sum vectors without changing their units."""
    return tuple(map(sum, zip(*vectors, strict=True)))


def scale(weight: float, vector: tuple[float, ...]) -> tuple[float, ...]:
    """Weight each outcome by its event probability."""
    return tuple(weight * x for x in vector)


class Policy(Protocol):
    """Public policy interface including the committed normal-answer action."""

    # Read-only so that a frozen dataclass also satisfies the protocol.
    @property
    def setting(self) -> Setting:
        """Return the model the policy plans against."""
        ...

    @property
    def accounting(self) -> str:
        """Return the loss convention the policy assumes."""
        ...

    def action(self, state: PublicState) -> str:
        """Select an action from public state."""
        ...

    def release_normal(self, request: Request, stage: int) -> bool:
        """Decide whether a normal answer warrants release."""
        ...


class Planner(Solver):
    """Original planner with explicitly recomputed action or accounting changes."""

    def __init__(self, setting: Setting, mode: str = "optimal", accounting: str = "basic") -> None:
        if accounting not in ("basic", "additive"):
            raise ValueError("invalid accounting")
        if mode not in (*MODES, "no_verify", "no_wait"):
            raise ValueError("invalid policy")
        self.accounting = accounting
        self.ablation = mode if mode.startswith("no_") else None
        super().__init__(setting, "optimal" if self.ablation else mode)

    def current(self, request: Request, stage: int, bad: int, good: int) -> tuple[float, float]:
        """Return the grant and query values, adjusted for the accounting."""
        grant, query = super().current(request, stage, bad, good)
        if self.accounting == "additive":
            sigma = self.window(request)[0][stage]
            grant -= request.amount * self.setting.risk(bad, good) * self.setting.harm * (1 - sigma)
        return grant, query

    def values(
        self, n: int, request: Request, stage: int, bad: int, good: int
    ) -> tuple[float, ...]:
        """Return the four action values, with the ablated action removed."""
        values = list(super().values(n, request, stage, bad, good))
        if self.ablation == "no_verify":
            values[2] = -math.inf
        if self.ablation == "no_wait":
            values[3] = -math.inf
        return tuple(values)

    def release_normal(self, request: Request, stage: int) -> bool:
        """Decide whether a normal answer warrants release."""
        return bool(self.window(request)[0][stage] * (1 + self.setting.margin) >= 1)


@dataclass(frozen=True)
class Threshold:
    """Relationship-wide threshold parameters, with optional posterior updates."""

    setting: Setting
    family: str
    posterior: bool = True
    theta: float = 0.0
    lower: float = 0.0
    upper: float = 1.0
    watch: int = 0
    accounting: str = "basic"

    def __post_init__(self) -> None:
        """Reject parameters outside the rule family's domain."""
        if self.family not in ("B1", "B2", "B3", "B4"):
            raise ValueError("invalid threshold family")
        if not 0 <= self.lower <= self.upper <= 1 or self.theta < 0:
            raise ValueError("invalid thresholds")
        if type(self.watch) is not int or self.watch < 0:
            raise ValueError("invalid watch length")
        if self.accounting not in ("basic", "additive"):
            raise ValueError("invalid accounting")

    def action(self, state: PublicState) -> str:
        """Select an action from public state."""
        bad, good = (state.misuse_labels, state.normal_labels) if self.posterior else (0, 0)
        risk = self.setting.risk(bad, good)
        if self.family == "B1":
            return (
                "grant"
                if risk * self.setting.harm * state.request.amount <= self.theta
                else "reject"
            )
        if self.family == "B2":
            return "grant" if state.request.amount <= self.theta else "reject"
        if self.family == "B4" and state.stage < min(self.watch, len(state.request.hazards)):
            return "wait"
        if risk < self.lower:
            return "grant"
        if risk > self.upper:
            return "reject"
        return "verify"

    def release_normal(self, request: Request, stage: int) -> bool:
        """Decide whether a normal answer warrants release."""
        return bool(
            math.prod(1 - f for f in request.hazards[stage:]) * (1 + self.setting.margin) >= 1
        )


def aligned(actual: Setting, assumed: Setting) -> None:
    """Reject changes to observable support or protocol configuration."""
    if len(actual.schedule) != len(assumed.schedule) or actual.deadline != assumed.deadline:
        raise ValueError("schedule and deadline must agree")
    for left, right in zip(actual.schedule, assumed.schedule, strict=True):
        if len(left) != len(right):
            raise ValueError("setting support mismatch")
        for (w, r), (z, t) in zip(left, right, strict=True):
            if w != z or r.amount != t.amount or len(r.hazards) != len(t.hazards):
                raise ValueError("public request support mismatch")


class Evaluator:
    """Exact vector expectations for a fixed policy under an actual environment.

    The normal-response decision is also kept fixed under the assumed model.
    No true-model replanning occurs after a label or settlement transition.
    """

    def __init__(self, actual: Setting, policy: Policy, accounting: str = "basic") -> None:
        aligned(actual, policy.setting)
        if accounting not in ("basic", "additive"):
            raise ValueError("invalid actual accounting")
        self.actual, self.policy, self.accounting = actual, policy, accounting
        # Memoize per instance; the recursion below revisits the same states.
        self.future = cache(self.future)  # type: ignore[method-assign]
        self.stage = cache(self.stage)  # type: ignore[method-assign]
        self.window = cache(self.window)  # type: ignore[method-assign]

    def window(self, n: int, choice: int, stage: int) -> tuple[float, float, float, float, float]:
        """Enumerate response opportunities independently of the planner window."""
        s = self.actual
        request = s.schedule[n][choice][1]
        assumed = self.policy.setting.schedule[n][choice][1]
        mass, eta, delay, normal_value, released, unpaid = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        i = stage
        for _ in range(s.deadline):
            delay += mass * s.wait_cost * request.amount
            answer = mass * s.response_rate
            eta += answer
            if self.policy.release_normal(assumed, i):
                survival = math.prod(1 - f for f in request.hazards[i:])
                normal_value += answer * request.amount * (survival * (1 + s.margin) - 1)
                released += answer
                unpaid += answer * (1 - survival) * request.amount
            failure = request.hazards[i] if i < len(request.hazards) else 0
            mass *= (1 - s.response_rate) * (1 - failure)
            i = min(i + 1, len(request.hazards))
        return eta, delay, normal_value, released, unpaid

    def future(self, n: int = 0, bad: int = 0, good: int = 0) -> tuple[float, ...]:
        """Return the expected outcome vector from request n onward."""
        if n == len(self.actual.schedule):
            return ZERO
        result = ZERO
        pi = self.actual.risk(bad, good)
        for choice, (weight, _) in enumerate(self.actual.schedule[n]):
            result = add(result, scale(weight, self.stage(n, choice, 0, bad, good)))
        return add(result, (0, 0, 1 - pi, 0, 0, 0))

    def stage(self, n: int, choice: int, stage: int, bad: int, good: int) -> tuple[float, ...]:
        """Return the expected outcome vector from one settlement stage."""
        s = self.actual
        request = s.schedule[n][choice][1]
        assumed = self.policy.setting.schedule[n][choice][1]
        action = self.policy.action(PublicState(n, assumed, stage, bad, good))
        pi = s.risk(bad, good)
        follow = self.future(n + 1, bad, good)
        if action == "grant":
            sigma = math.prod(1 - f for f in request.hazards[stage:])
            reward = request.amount * (sigma * (1 + (1 - pi) * s.margin - pi * s.harm) - 1)
            if self.accounting == "additive":
                reward -= request.amount * (1 - sigma) * pi * s.harm
            return add(
                follow,
                (reward, 0, 0, 0, pi * request.amount, (1 - sigma) * request.amount),
            )
        if action == "reject":
            return add(follow, (0, 0, 0, 1 - pi, 0, 0))
        if action == "wait":
            if stage >= len(request.hazards):
                raise ValueError("policy waited after final settlement")
            failure = request.hazards[stage]
            return add(
                (-s.wait_cost * request.amount, 0, 0, failure * (1 - pi), 0, 0),
                scale(failure, follow),
                scale(1 - failure, self.stage(n, choice, stage + 1, bad, good)),
            )
        if action != "verify":
            raise ValueError("invalid action")
        eta, delay, normal_value, released, unpaid = self.window(n, choice, stage)
        return add(
            (
                -s.query_cost - delay + (1 - pi) * normal_value,
                1,
                0,
                (1 - pi) * (1 - released),
                0,
                (1 - pi) * unpaid,
            ),
            scale(1 - eta, follow),
            scale(eta * pi, self.future(n + 1, bad + 1, good)),
            scale(eta * (1 - pi), self.future(n + 1, bad, good + 1)),
        )

    def report(self) -> dict[str, float]:
        """Return relation-level expectations and the pooled service ratio."""
        out = dict(zip(METRICS, self.future(), strict=True))
        out["normal_nonrelease_rate"] = out["normal_nonrelease"] / out["normal_requests"]
        return out


def candidates(
    setting: Setting,
    family: str,
    posterior: bool,
    points: int,
    accounting: str = "basic",
) -> Iterator[tuple[Threshold, list[dict[str, float]]]]:
    """Group grid tuples with identical decisions on every possible public risk.

    This is exact behavior deduplication, not threshold subsampling. Every original
    parameter tuple belongs to one group, including unused and equality thresholds.
    """
    if points < 2:
        raise ValueError("at least two grid points required")
    risks = sorted(
        {
            setting.risk(a, d)
            for n in range(len(setting.schedule))
            for a in range(n + 1)
            for d in range(n - a + 1)
        }
        if posterior
        else {setting.risk(0, 0)}
    )
    amounts = sorted({r.amount for dist in setting.schedule for _, r in dist})
    groups: dict[object, list[dict[str, float]]] = {}
    if family in ("B1", "B2"):
        maximum = max(amounts) * (setting.harm if family == "B1" else 1)
        scores = (
            [p * setting.harm * v for p in risks for v in amounts] if family == "B1" else amounts
        )
        for j in range(points):
            theta = maximum * j / (points - 1)
            key = tuple(x <= theta for x in scores)
            groups.setdefault(key, []).append({"theta": theta})
    else:
        grid = [j / (points - 1) for j in range(points)]
        watches = (
            range(1 + max(len(r.hazards) for dist in setting.schedule for _, r in dist))
            if family == "B4"
            else (0,)
        )
        # Form risk signatures once per threshold pair, then add the watch length.
        for a in grid:
            for b in grid:
                if a > b:
                    continue
                signature = tuple("g" if p < a else "r" if p > b else "v" for p in risks)
                for watch in watches:
                    groups.setdefault((watch, signature), []).append(
                        {"lower": a, "upper": b, "watch": watch}
                    )
    for params in groups.values():
        yield (
            Threshold(setting, family, posterior, accounting=accounting, **params[0]),
            params,
        )


class Fit(TypedDict):
    """Summary of one tuning run over a threshold family."""

    value: float
    numerical_tie_tolerance: float
    tied_parameters: list[dict[str, float]]
    behavioral_groups: int
    grid_candidates: int
    selected: dict[str, float]


def tune(
    setting: Setting,
    family: str,
    posterior: bool = True,
    points: int = 21,
    accounting: str = "basic",
) -> tuple[Threshold, Fit]:
    """Tune on exact assumed-model relation reward and retain all numerical ties."""
    exposure = sum(sum(w * r.amount for w, r in dist) for dist in setting.schedule)
    tolerance = 1e-10 * (1 + exposure)
    evaluated: list[tuple[float, Threshold, list[dict[str, float]]]] = []
    for policy, params in candidates(setting, family, posterior, points, accounting):
        value = Evaluator(setting, policy, accounting).future()[0]
        evaluated.append((value, policy, params))
    best = max(row[0] for row in evaluated)
    # Select the actual largest computed value. Record tolerance ties separately.
    selected = max(evaluated, key=lambda row: row[0])[1]
    ties = [p for value, _, params in evaluated if best - value <= tolerance for p in params]
    return selected, {
        "value": best,
        "numerical_tie_tolerance": tolerance,
        "tied_parameters": ties,
        "behavioral_groups": len(evaluated),
        "grid_candidates": sum(len(row[2]) for row in evaluated),
        "selected": {k: getattr(selected, k) for k in ("theta", "lower", "upper", "watch")},
    }
