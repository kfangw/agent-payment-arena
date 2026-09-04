"""Single-relationship simulation with hidden exogenous paths."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from arena.experiments.relationship.belief import update_belief
from arena.experiments.relationship.model import (
    LatentRelationship,
    PaymentDisposition,
    PolicyContext,
    RelationshipAction,
    RelationshipEnd,
    RelationshipObservation,
    RelationshipParameters,
    RelationshipResult,
    SellerType,
    TransactionEvent,
    Verdict,
    policy_stream_id,
)
from arena.experiments.relationship.policies import RelationshipPolicy

_STREAM_SELLER_TYPE = 10
_STREAM_CHEATING = 20
_STREAM_VERDICT = 30
_STREAM_PUBLIC_COIN = 40
_STREAM_CONTINUATION = 50
_STREAM_TYPE_SWITCHING = 60
_STREAM_DELIVERY = 70


def draw_latent_relationship(
    parameters: RelationshipParameters,
    *,
    seed: int,
) -> LatentRelationship:
    """Draw one policy-independent latent relationship from stable streams."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    horizon = parameters.max_transactions

    def uniforms(stream_id: int, size: int = horizon) -> NDArray[np.float64]:
        sequence = np.random.SeedSequence(seed, spawn_key=(stream_id,))
        return np.random.Generator(np.random.PCG64DXSM(sequence)).random(size)

    seller_draw = uniforms(_STREAM_SELLER_TYPE, 1)[0]
    seller_type = SellerType.BAD if seller_draw < parameters.prior_bad else SellerType.HONEST
    return LatentRelationship(
        initial_type=seller_type,
        cheating_uniforms=uniforms(_STREAM_CHEATING),
        verdict_uniforms=uniforms(_STREAM_VERDICT),
        public_coin_uniforms=uniforms(_STREAM_PUBLIC_COIN),
        continuation_uniforms=uniforms(_STREAM_CONTINUATION),
        type_uniforms=uniforms(_STREAM_TYPE_SWITCHING),
        delivery_uniforms=uniforms(_STREAM_DELIVERY),
    )


def simulate_relationship(
    parameters: RelationshipParameters,
    latent: LatentRelationship,
    policy: RelationshipPolicy,
    *,
    policy_seed: int,
) -> RelationshipResult:
    """Replay one policy on one hidden relationship path."""
    if len(latent.cheating_uniforms) < parameters.max_transactions:
        raise ValueError("latent path is shorter than max_transactions")
    if policy_seed < 0:
        raise ValueError("policy_seed must be nonnegative")

    stream_id = 1000 + policy_stream_id(policy.policy_id)
    sequence = np.random.SeedSequence(policy_seed, spawn_key=(stream_id,))
    policy_rng = np.random.Generator(np.random.PCG64DXSM(sequence))
    policy.reset(PolicyContext(parameters), policy_rng)

    belief = parameters.prior_bad
    last_verdict = Verdict.NOT_CHECKED
    public_checks = 0
    discretionary_checks = 0
    events: list[TransactionEvent] = []

    for transaction_index in range(parameters.max_transactions):
        observation = RelationshipObservation(
            transaction_index=transaction_index,
            belief_bad=belief,
            deposit=parameters.deposit,
            pending_verdicts=0,
            last_verdict=last_verdict,
            cumulative_public_checks=public_checks,
            cumulative_discretionary_checks=discretionary_checks,
        )
        action = policy.act(observation)
        if not isinstance(action, RelationshipAction):
            raise TypeError("relationship policies must return RelationshipAction")
        if action is RelationshipAction.EXIT:
            return _result(
                policy.policy_id,
                latent,
                events,
                RelationshipEnd.POLICY_EXIT,
                belief,
                public_checks,
                discretionary_checks,
            )

        cheated = bool(
            latent.initial_type is SellerType.BAD
            and latent.cheating_uniforms[transaction_index] < parameters.bad_cheat_rate
        )
        public_check = bool(
            action is RelationshipAction.PAY
            and latent.public_coin_uniforms[transaction_index] < parameters.public_sampling_rate
        )
        discretionary_check = action in {
            RelationshipAction.VERIFY,
            RelationshipAction.DEFER,
        }
        checked = public_check or discretionary_check
        if public_check:
            public_checks += 1
        if discretionary_check:
            discretionary_checks += 1

        verdict = _verdict(
            checked=checked,
            cheated=cheated,
            uniform=latent.verdict_uniforms[transaction_index],
            parameters=parameters,
        )
        belief_after = update_belief(
            belief,
            verdict,
            alpha=parameters.alpha,
            beta=parameters.beta,
            cheat_rate=parameters.bad_cheat_rate,
        )
        event = _event(
            transaction_index=transaction_index,
            belief_before=belief,
            belief_after=belief_after,
            action=action,
            cheated=cheated,
            public_check=public_check,
            verdict=verdict,
            parameters=parameters,
        )
        events.append(event)
        belief = belief_after
        last_verdict = verdict

        if latent.continuation_uniforms[transaction_index] >= parameters.continuation_probability:
            return _result(
                policy.policy_id,
                latent,
                events,
                RelationshipEnd.EXOGENOUS_DEPARTURE,
                belief,
                public_checks,
                discretionary_checks,
            )

    return _result(
        policy.policy_id,
        latent,
        events,
        RelationshipEnd.HORIZON,
        belief,
        public_checks,
        discretionary_checks,
    )


def _verdict(
    *,
    checked: bool,
    cheated: bool,
    uniform: float,
    parameters: RelationshipParameters,
) -> Verdict:
    if not checked:
        return Verdict.NOT_CHECKED
    if cheated:
        return Verdict.FAIL if uniform < 1 - parameters.beta else Verdict.PASS
    return Verdict.FAIL if uniform < parameters.alpha else Verdict.PASS


def _event(
    *,
    transaction_index: int,
    belief_before: float,
    belief_after: float,
    action: RelationshipAction,
    cheated: bool,
    public_check: bool,
    verdict: Verdict,
    parameters: RelationshipParameters,
) -> TransactionEvent:
    checked = verdict is not Verdict.NOT_CHECKED
    check_cost = parameters.verification_cost if checked else 0.0
    failed = verdict is Verdict.FAIL
    compensation = 0.0

    if action is RelationshipAction.DEFER:
        payment = PaymentDisposition.REFUNDED if failed else PaymentDisposition.PAID
        delivery_utility = 0.0 if failed else _delivery_utility(cheated, parameters)
        buyer_utility = delivery_utility - check_cost - parameters.escrow_cost_rate * parameters.v
    else:
        payment = PaymentDisposition.PAID
        if failed:
            compensation = min(parameters.deposit, parameters.v)
        buyer_utility = _delivery_utility(cheated, parameters) - check_cost + compensation

    return TransactionEvent(
        transaction_index=transaction_index,
        belief_before=belief_before,
        belief_after=belief_after,
        action=action,
        cheated=cheated,
        checked=checked,
        public_check=public_check,
        verdict=verdict,
        payment=payment,
        buyer_utility=buyer_utility,
        verification_cost=check_cost,
        buyer_compensation=compensation,
        deposit_forfeited=failed and parameters.deposit > 0,
    )


def _delivery_utility(
    cheated: bool,
    parameters: RelationshipParameters,
) -> float:
    return -parameters.v if cheated else parameters.u * parameters.v


def _result(
    policy_id: str,
    latent: LatentRelationship,
    events: list[TransactionEvent],
    end: RelationshipEnd,
    belief: float,
    public_checks: int,
    discretionary_checks: int,
) -> RelationshipResult:
    return RelationshipResult(
        policy_id=policy_id,
        initial_type=latent.initial_type,
        events=tuple(events),
        end=end,
        final_belief_bad=belief,
        public_checks=public_checks,
        discretionary_checks=discretionary_checks,
    )
