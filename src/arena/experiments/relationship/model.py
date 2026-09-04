"""Stable public identifiers for repeated relationship experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray


class RelationshipAction(StrEnum):
    """A buyer action inside an active seller relationship."""

    PAY = "pay"
    VERIFY = "verify"
    DEFER = "defer"
    EXIT = "exit"


class Verdict(StrEnum):
    """The observable outcome of a verification attempt."""

    PASS = "pass"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"
    PENDING = "pending"


class SellerType(StrEnum):
    """The latent seller type used by the instrument."""

    HONEST = "honest"
    BAD = "bad"


class SellerMode(StrEnum):
    """How a bad seller chooses whether to cheat."""

    BEHAVIORAL = "behavioral"
    STRATEGIC = "strategic"


class Phase(StrEnum):
    """A protocol phase with an isolated seed range."""

    PILOT = "pilot"
    TUNING = "tuning"
    EVALUATION = "evaluation"


class DataGrade(StrEnum):
    """The provenance of one experiment input."""

    GENERATED = "G"
    PARAMETERIZED = "P"
    MEASURED = "M"


class PaymentDisposition(StrEnum):
    """The final payment state of a realized transaction."""

    PAID = "paid"
    REFUNDED = "refunded"


class RelationshipEnd(StrEnum):
    """Why the simulator stopped a relationship."""

    POLICY_EXIT = "policy_exit"
    EXOGENOUS_DEPARTURE = "exogenous_departure"
    HORIZON = "horizon"


@dataclass(frozen=True)
class RelationshipParameters:
    """Resolved parameters needed to simulate one base-model relationship."""

    v: float
    u: float
    alpha: float
    beta: float
    bad_cheat_rate: float
    verification_cost: float
    escrow_cost_rate: float
    prior_bad: float
    continuation_probability: float
    public_sampling_rate: float
    deposit: float
    max_transactions: int

    def __post_init__(self) -> None:
        """Reject invalid values at the simulator boundary."""
        if self.v <= 0 or self.u <= 0:
            raise ValueError("v and u must be positive")
        if not 0 <= self.alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if not 0 <= self.bad_cheat_rate <= 1:
            raise ValueError("bad_cheat_rate must be between zero and one")
        if not 0 < self.prior_bad < 1:
            raise ValueError("prior_bad must be strictly between zero and one")
        if not 0 <= self.continuation_probability < 1:
            raise ValueError("continuation_probability must be in [0, 1)")
        if not 0 <= self.public_sampling_rate <= 1:
            raise ValueError("public_sampling_rate must be between zero and one")
        if 1 - self.beta <= self.alpha:
            raise ValueError("predicate must satisfy 1 - beta > alpha")
        if self.verification_cost < 0 or self.escrow_cost_rate < 0:
            raise ValueError("verification and escrow costs must be nonnegative")
        if self.deposit < 0:
            raise ValueError("deposit must be nonnegative")
        if self.max_transactions < 1:
            raise ValueError("max_transactions must be positive")


@dataclass(frozen=True)
class PolicyContext:
    """Public parameters supplied to a policy before a relationship."""

    parameters: RelationshipParameters


@dataclass(frozen=True)
class RelationshipObservation:
    """The complete public state visible to a buyer policy."""

    transaction_index: int
    belief_bad: float
    deposit: float
    pending_verdicts: int
    last_verdict: Verdict
    cumulative_public_checks: int
    cumulative_discretionary_checks: int


@dataclass(frozen=True)
class LatentRelationship:
    """Exogenous draws owned by the evaluation instrument."""

    initial_type: SellerType
    cheating_uniforms: NDArray[np.float64]
    verdict_uniforms: NDArray[np.float64]
    public_coin_uniforms: NDArray[np.float64]
    continuation_uniforms: NDArray[np.float64]
    type_uniforms: NDArray[np.float64]
    delivery_uniforms: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate aligned immutable random streams."""
        streams = (
            self.cheating_uniforms,
            self.verdict_uniforms,
            self.public_coin_uniforms,
            self.continuation_uniforms,
            self.type_uniforms,
            self.delivery_uniforms,
        )
        lengths = {len(stream) for stream in streams}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("latent streams must have the same positive length")
        for stream in streams:
            if stream.ndim != 1 or not np.all(np.isfinite(stream)):
                raise ValueError("latent streams must be finite one-dimensional arrays")
            if np.any(stream < 0) or np.any(stream >= 1):
                raise ValueError("latent uniforms must lie in [0, 1)")
            stream.flags.writeable = False


@dataclass(frozen=True)
class TransactionEvent:
    """One immutable realized transaction."""

    transaction_index: int
    belief_before: float
    belief_after: float
    action: RelationshipAction
    cheated: bool
    checked: bool
    public_check: bool
    verdict: Verdict
    payment: PaymentDisposition
    buyer_utility: float
    verification_cost: float
    buyer_compensation: float
    deposit_forfeited: bool


@dataclass(frozen=True)
class RelationshipResult:
    """The complete result of one policy on one latent relationship."""

    policy_id: str
    initial_type: SellerType
    events: tuple[TransactionEvent, ...]
    end: RelationshipEnd
    final_belief_bad: float
    public_checks: int
    discretionary_checks: int

    @property
    def realized_transactions(self) -> int:
        """Return the number of transactions that actually occurred."""
        return len(self.events)

    @property
    def total_buyer_utility(self) -> float:
        """Return undiscounted realized utility before seed aggregation."""
        return sum(event.buyer_utility for event in self.events)


POLICY_STREAM_IDS: Final = MappingProxyType(
    {
        "compiled_deposit_exit": 1,
        "always_pay": 2,
        "always_verify": 3,
        "adaptive_sampling": 4,
        "wald_sprt": 5,
        "one_strike": 6,
        "two_strike": 7,
        "no_public_sampling": 8,
        "no_deposit_optimum": 9,
    }
)


def policy_stream_id(policy_id: str) -> int:
    """Return the permanent random-stream id assigned to a policy."""
    try:
        return POLICY_STREAM_IDS[policy_id]
    except KeyError as error:
        raise ValueError(f"policy has no stable stream id: {policy_id}") from error
