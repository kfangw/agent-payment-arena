"""Stable public identifiers for repeated relationship experiments."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final


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
