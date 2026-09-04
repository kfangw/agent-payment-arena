"""Tests for the single-relationship event simulator."""

from collections.abc import Mapping
from dataclasses import fields

import numpy as np
import pytest

from arena.experiments.artifacts import JsonValue
from arena.experiments.relationship.model import (
    LatentRelationship,
    PaymentDisposition,
    PolicyContext,
    RelationshipAction,
    RelationshipEnd,
    RelationshipObservation,
    RelationshipParameters,
    SellerType,
    Verdict,
)
from arena.experiments.relationship.simulate import (
    draw_latent_relationship,
    simulate_relationship,
)


class FixedPolicy:
    """A deterministic policy used to isolate simulator behavior."""

    def __init__(self, policy_id: str, action: RelationshipAction) -> None:
        """Store the fixed action and stable stream identifier."""
        self.policy_id = policy_id
        self.action = action
        self.observations: list[RelationshipObservation] = []

    def reset(self, context: PolicyContext, rng: np.random.Generator) -> None:
        """Discard observations from the prior relationship."""
        self.observations = []

    def act(self, observation: RelationshipObservation) -> RelationshipAction:
        """Record the public observation and return the fixed action."""
        self.observations.append(observation)
        return self.action

    def snapshot(self) -> Mapping[str, JsonValue]:
        """Describe the complete deterministic policy configuration."""
        return {"action": self.action.value}


def _parameters(**updates: float | int) -> RelationshipParameters:
    values: dict[str, float | int] = {
        "v": 1.0,
        "u": 0.2,
        "alpha": 0.02,
        "beta": 0.10,
        "bad_cheat_rate": 1.0,
        "verification_cost": 3.0,
        "escrow_cost_rate": 0.05,
        "prior_bad": 0.05,
        "continuation_probability": 0.0,
        "public_sampling_rate": 0.0,
        "deposit": 10.0,
        "max_transactions": 3,
    }
    values.update(updates)
    return RelationshipParameters(**values)  # type: ignore[arg-type]


def _latent(
    seller_type: SellerType,
    *,
    cheating: tuple[float, ...] = (0.0, 0.0, 0.0),
    verdict: tuple[float, ...] = (0.5, 0.5, 0.5),
    public: tuple[float, ...] = (0.5, 0.5, 0.5),
    continuation: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> LatentRelationship:
    zeros = np.zeros(len(cheating))
    return LatentRelationship(
        initial_type=seller_type,
        cheating_uniforms=np.asarray(cheating),
        verdict_uniforms=np.asarray(verdict),
        public_coin_uniforms=np.asarray(public),
        continuation_uniforms=np.asarray(continuation),
        type_uniforms=zeros.copy(),
        delivery_uniforms=zeros.copy(),
    )


def test_honest_unchecked_payment_records_one_departing_transaction() -> None:
    policy = FixedPolicy("always_pay", RelationshipAction.PAY)
    result = simulate_relationship(
        _parameters(),
        _latent(SellerType.HONEST),
        policy,
        policy_seed=1,
    )

    assert result.end is RelationshipEnd.EXOGENOUS_DEPARTURE
    assert result.realized_transactions == 1
    assert result.total_buyer_utility == pytest.approx(0.2)
    assert result.public_checks == 0
    event = result.events[0]
    assert event.verdict is Verdict.NOT_CHECKED
    assert event.payment is PaymentDisposition.PAID
    assert event.belief_before == event.belief_after == 0.05


def test_failed_prepaid_verification_compensates_once_and_updates_belief() -> None:
    policy = FixedPolicy("always_verify", RelationshipAction.VERIFY)
    result = simulate_relationship(
        _parameters(),
        _latent(SellerType.BAD),
        policy,
        policy_seed=2,
    )

    event = result.events[0]
    assert event.cheated is True
    assert event.verdict is Verdict.FAIL
    assert event.deposit_forfeited is True
    assert event.buyer_compensation == 1.0
    assert event.buyer_utility == pytest.approx(-3.0)
    assert event.belief_after == pytest.approx(0.703125)
    assert result.discretionary_checks == 1


def test_failed_deferred_payment_refunds_without_loss_compensation() -> None:
    policy = FixedPolicy("always_verify", RelationshipAction.DEFER)
    result = simulate_relationship(
        _parameters(),
        _latent(SellerType.BAD),
        policy,
        policy_seed=3,
    )

    event = result.events[0]
    assert event.payment is PaymentDisposition.REFUNDED
    assert event.buyer_compensation == 0.0
    assert event.buyer_utility == pytest.approx(-3.05)
    assert event.deposit_forfeited is True


def test_policy_exit_creates_no_transaction_event() -> None:
    policy = FixedPolicy("always_pay", RelationshipAction.EXIT)
    result = simulate_relationship(
        _parameters(),
        _latent(SellerType.HONEST),
        policy,
        policy_seed=4,
    )

    assert result.end is RelationshipEnd.POLICY_EXIT
    assert result.events == ()
    assert result.total_buyer_utility == 0.0


def test_public_check_uses_public_stream_and_is_counted_separately() -> None:
    policy = FixedPolicy("always_pay", RelationshipAction.PAY)
    result = simulate_relationship(
        _parameters(public_sampling_rate=0.25),
        _latent(SellerType.HONEST, public=(0.1, 0.5, 0.5)),
        policy,
        policy_seed=5,
    )

    assert result.public_checks == 1
    assert result.discretionary_checks == 0
    assert result.events[0].public_check is True
    assert result.events[0].verdict is Verdict.PASS
    assert result.total_buyer_utility == pytest.approx(-2.8)


def test_latent_draws_are_reproducible_and_read_only() -> None:
    first = draw_latent_relationship(_parameters(), seed=123)
    second = draw_latent_relationship(_parameters(), seed=123)

    assert first.initial_type is second.initial_type
    assert np.array_equal(first.cheating_uniforms, second.cheating_uniforms)
    assert np.array_equal(first.verdict_uniforms, second.verdict_uniforms)
    assert first.cheating_uniforms.flags.writeable is False
    with pytest.raises(ValueError):
        first.cheating_uniforms[0] = 0.9


def test_public_observation_has_no_latent_or_future_fields() -> None:
    public_fields = {field.name for field in fields(RelationshipObservation)}

    assert "initial_type" not in public_fields
    assert "cheated" not in public_fields
    assert not any("uniform" in name for name in public_fields)


def test_short_latent_path_is_rejected() -> None:
    policy = FixedPolicy("always_pay", RelationshipAction.PAY)

    with pytest.raises(ValueError, match="shorter"):
        simulate_relationship(
            _parameters(max_transactions=4),
            _latent(SellerType.HONEST),
            policy,
            policy_seed=6,
        )
