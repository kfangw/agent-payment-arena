"""Seeded synthetic replay with separate policy and evaluator records."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .model import PublicState, Request, Setting
from .solver import Solver


def uniform(
    seed: int, split: str, relation: int, request: int, channel: str, tick: int = 0
) -> float:
    """Independent named streams, stable across policies and execution order."""
    key = f"{seed}|{split}|{relation}|{request}|{channel}|{tick}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big")).random()


@dataclass
class History:
    """Accept one terminal observation per logical request; late replies are ignored."""

    misuse: int = 0
    normal: int = 0
    closed: set[int] = field(default_factory=set)

    def close(self, request: int, label: int | None) -> None:
        """Close even without a label, so retries cannot add late information."""
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("intent label must be 0, 1 or None")
        if request in self.closed:
            return
        self.closed.add(request)
        self.misuse += int(label == 1)
        self.normal += int(label == 0)


def select_request(setting: Setting, n: int, draw: float) -> Request:
    """Draw an exogenous request from the public finite distribution."""
    cumulative = 0.0
    for weight, request in setting.schedule[n]:
        cumulative += weight
        if draw < cumulative:
            return request
    return setting.schedule[n][-1][1]


def replay(
    solver: Solver, seed: int, split: str, relation: int
) -> tuple[dict, list[dict]]:
    """Replay a relation; return private evaluation metrics and public events."""
    s = solver.setting
    high = uniform(seed, split, relation, -1, "type") < s.prior
    risk = s.high if high else s.low
    history = History()
    total = dict(
        reward=0.0,
        queries=0,
        query_cost=0.0,
        wait_ticks=0,
        wait_cost=0.0,
        misuse_release_amount=0.0,
        unpaid_loss=0.0,
        normal_requests=0,
        normal_rejections=0,
        labels=0,
    )
    events = []
    for n in range(len(s.schedule)):
        request = select_request(s, n, uniform(seed, split, relation, n, "request"))
        latent_misuse = uniform(seed, split, relation, n, "intent") < risk
        total["normal_requests"] += int(not latent_misuse)
        final = len(request.hazards)
        stage = 0
        released = False
        label = None
        reason = "reject"

        def fails(at: int, r: Request = request, index: int = n) -> bool:
            return (
                uniform(seed, split, relation, index, "settlement", at) < r.hazards[at]
            )

        def release(
            at: int,
            r: Request = request,
            fin: int = final,
            misuse: bool = latent_misuse,
            has_failed: Callable[[int], bool] = fails,
        ) -> None:
            nonlocal released
            released = True
            paid = not any(has_failed(j) for j in range(at, fin))
            if misuse:
                total["misuse_release_amount"] += r.amount
            if not paid:
                total["reward"] -= r.amount
                total["unpaid_loss"] += r.amount
            else:
                total["reward"] += r.amount * (-s.harm if misuse else s.margin)

        def wait_cost(r: Request = request) -> None:
            total["wait_ticks"] += 1
            total["wait_cost"] += s.wait_cost * r.amount
            total["reward"] -= s.wait_cost * r.amount

        while True:
            state = PublicState(n, request, stage, history.misuse, history.normal)
            action = solver.action(state)
            events.append(
                dict(
                    relation_id=relation,
                    request_id=n,
                    stage=stage,
                    amount=request.amount,
                    action=action,
                    misuse_labels=history.misuse,
                    normal_labels=history.normal,
                    belief=s.belief(history.misuse, history.normal),
                    event="decision",
                )
            )
            if action == "grant":
                release(stage)
                reason = "grant"
                break
            if action == "reject":
                break
            if action == "wait":
                wait_cost()
                if fails(stage):
                    reason = "settlement_failed"
                    break
                stage += 1
                continue
            total["queries"] += 1
            total["query_cost"] += s.query_cost
            total["reward"] -= s.query_cost
            reason = "timeout"
            for tick in range(s.deadline):
                wait_cost()
                if (
                    uniform(seed, split, relation, n, "response", tick)
                    < s.response_rate
                ):
                    label = int(latent_misuse)
                    reason = "answered"
                    sigma = solver.window(request)[0][stage]
                    if (
                        label == 0
                        and request.amount * (sigma * (1 + s.margin) - 1) >= 0
                    ):
                        release(stage)
                    break
                if stage < final:
                    if fails(stage):
                        reason = "settlement_failed_before_reply"
                        break
                    stage += 1
            break
        history.close(n, label)
        total["labels"] += int(label is not None)
        total["normal_rejections"] += int(not latent_misuse and not released)
        events.append(
            dict(
                relation_id=relation,
                request_id=n,
                event="close",
                reason=reason,
                label=label,
                released=released,
                posterior=s.belief(history.misuse, history.normal),
            )
        )
    return total, events
