"""The x402 contract the arena evaluates against.

This module is the seam between the arena and the system under test. It states
what a gateway must do, in terms both backends can be held to: the shape of a
402 response, the machine-readable reason a payment was refused, and the space
of decisions an accept policy may return.

Scope note: this file is deliberately incomplete. It currently pins the two
pieces that are already fixed by the reference implementation, the decision
space and the error codes. The 402 payload, the mandate schema, and the ask
flow are documented in a later change once the reference gateway has been read
end to end.
"""

from enum import StrEnum


class Action(StrEnum):
    """What an accept policy decided to do with a payment attempt.

    The arena and the reference gateway share this decision space on purpose.
    A policy written against one can be evaluated against the other, and a
    result measured here can be read back as a change to a real policy.

    Only `APPROVE` lets money move immediately. The other four are distinct
    kinds of "not yet", and telling them apart is most of the point: a policy
    that converts everything into `REJECT` looks excellent against attacks and
    is useless in practice.
    """

    APPROVE = "approve"
    """Settle the payment now."""

    REJECT = "reject"
    """Refuse the payment. The agent may not retry it as-is."""

    DEFER = "defer"
    """Hold the payment for re-evaluation rather than settling or refusing."""

    ASK = "ask"
    """Refuse until the delegator signs a confirmation bound to this payment."""

    REQUIRE_BOND = "bond"
    """Refuse until the payer posts a bond that can be slashed."""


class ErrorCode(StrEnum):
    """Stable machine-readable reasons a gateway refused a payment.

    Carried alongside the human-readable message in a 402 response so a client
    can branch on cause without parsing prose. The arena treats these as part
    of the contract: a backend that refuses for the right reason with the wrong
    code is a contract violation, because an agent's recovery path depends on
    the code.
    """

    # Transport and verification.
    PAYMENT_REQUIRED = "payment_required"
    INVALID_HEADER = "invalid_payment_header"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_ERROR = "verification_error"

    # Identity registry.
    IDENTITY_UNREGISTERED = "identity_unregistered"
    IDENTITY_CHECK_FAILED = "identity_check_failed"

    # Settlement.
    SETTLEMENT_FAILED = "settlement_failed"
    SETTLEMENT_ERROR = "settlement_error"

    # Policy outcomes that are not an outright rejection.
    POLICY_REJECTED = "policy_rejected"
    PAYMENT_DEFERRED = "payment_deferred"
    CONFIRMATION_REQUIRED = "confirmation_required"
    BOND_REQUIRED = "bond_required"

    # Mandate scope. These are the codes the arena cares about most: they are
    # the ones a valid signature can still run into.
    MANDATE_MISSING = "mandate_missing"
    MANDATE_INVALID = "mandate_invalid"
    MANDATE_EXPIRED = "mandate_expired"
    MANDATE_REVOKED = "mandate_revoked"
    MANDATE_EXCEEDED = "mandate_exceeded"
    MANDATE_BUDGET_EXCEEDED = "mandate_budget_exceeded"
    MANDATE_RATE_EXCEEDED = "mandate_rate_exceeded"


DEFAULT_CODE_FOR_ACTION: dict[Action, ErrorCode] = {
    Action.REJECT: ErrorCode.POLICY_REJECTED,
    Action.DEFER: ErrorCode.PAYMENT_DEFERRED,
    Action.ASK: ErrorCode.CONFIRMATION_REQUIRED,
    Action.REQUIRE_BOND: ErrorCode.BOND_REQUIRED,
}
"""The code a gateway reports when a policy declines without naming its own."""


def default_code_for(action: Action) -> ErrorCode | None:
    """Return the error code a refusal carries when the policy names none.

    Args:
        action: The action a policy returned.

    Returns:
        The code to report, or `None` for `Action.APPROVE`, which is not a
        refusal and therefore carries no code.
    """
    return DEFAULT_CODE_FOR_ACTION.get(action)
