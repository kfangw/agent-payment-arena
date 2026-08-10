"""The contract is a public API, so it is pinned by tests, not by convention."""

import pytest

from arena.gateway.contract import Action, ErrorCode, default_code_for


def test_decision_space_is_exactly_five_outcomes() -> None:
    # The arena and the reference gateway share this space. Adding or removing
    # an outcome on one side without the other silently breaks the round trip
    # between a policy evaluated here and a policy shipped there.
    assert {a.value for a in Action} == {"approve", "reject", "defer", "ask", "bond"}


def test_approve_carries_no_refusal_code() -> None:
    assert default_code_for(Action.APPROVE) is None


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action.REJECT, ErrorCode.POLICY_REJECTED),
        (Action.DEFER, ErrorCode.PAYMENT_DEFERRED),
        (Action.ASK, ErrorCode.CONFIRMATION_REQUIRED),
        (Action.REQUIRE_BOND, ErrorCode.BOND_REQUIRED),
    ],
)
def test_every_refusal_has_a_default_code(action: Action, expected: ErrorCode) -> None:
    assert default_code_for(action) is expected


def test_error_codes_are_stable_snake_case() -> None:
    # Clients branch on these strings. Renaming one is a breaking change, and
    # the casing is part of the wire format.
    for code in ErrorCode:
        assert code.value == code.value.lower()
        assert " " not in code.value
        assert "-" not in code.value


def test_error_code_values_are_unique() -> None:
    values = [code.value for code in ErrorCode]
    assert len(values) == len(set(values))
