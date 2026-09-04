"""Wire-compatible data models for the x402 payment contract."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
BYTES32 = re.compile(r"^0x[0-9a-fA-F]{64}$")
SIGNATURE = re.compile(r"^0x[0-9a-fA-F]{130}$")


class WireModel(BaseModel):
    """Base model that accepts field names and emits camel-case aliases."""

    model_config = ConfigDict(populate_by_name=True)


def _decimal(value: str) -> str:
    if not value.isdecimal():
        raise ValueError("must be a non-negative decimal integer")
    return value


def _address(value: str) -> str:
    if not ADDRESS.fullmatch(value):
        raise ValueError("must be a 20-byte 0x-prefixed address")
    return value


def _bytes32(value: str) -> str:
    if not BYTES32.fullmatch(value):
        raise ValueError("must be a 32-byte 0x-prefixed value")
    return value


class PaymentRequirements(WireModel):
    """One exact-scheme payment option from a 402 response."""

    scheme: str = "exact"
    network: str
    max_amount_required: str = Field(alias="maxAmountRequired")
    resource: str
    description: str
    mime_type: str = Field(alias="mimeType")
    pay_to: str = Field(alias="payTo")
    max_timeout_seconds: int = Field(alias="maxTimeoutSeconds", gt=0)
    asset: str
    extra: dict[str, str] = Field(default_factory=dict)

    _amount = field_validator("max_amount_required")(_decimal)
    _addresses = field_validator("pay_to", "asset")(_address)


class AskRequest(WireModel):
    """The exact payment a delegator is asked to confirm."""

    mandate_id: str = Field(alias="mandateId")
    authorization_nonce: str = Field(alias="authorizationNonce")
    amount: str
    resource: str

    _ids = field_validator("mandate_id", "authorization_nonce")(_bytes32)
    _amount = field_validator("amount")(_decimal)


class RequirementsResponse(WireModel):
    """The JSON body returned with HTTP status 402."""

    x402_version: int = Field(1, alias="x402Version")
    error: str
    error_code: str | None = Field(default=None, alias="errorCode")
    accepts: tuple[PaymentRequirements, ...]
    ask: AskRequest | None = None


class Authorization(WireModel):
    """Wire representation of an EIP-3009 transfer authorization."""

    from_address: str = Field(alias="from")
    to: str
    value: str
    valid_after: str = Field(alias="validAfter")
    valid_before: str = Field(alias="validBefore")
    nonce: str

    _addresses = field_validator("from_address", "to")(_address)
    _numbers = field_validator("value", "valid_after", "valid_before")(_decimal)
    _nonce = field_validator("nonce")(_bytes32)


class Mandate(WireModel):
    """A delegator-signed grant of spending authority."""

    delegator: str
    agent: str
    max_amount_per_payment: str = Field(alias="maxAmountPerPayment")
    allowed_payees: tuple[str, ...] = Field(default=(), alias="allowedPayees")
    allowed_resources: tuple[str, ...] = Field(default=(), alias="allowedResources")
    valid_after: str = Field(alias="validAfter")
    valid_before: str = Field(alias="validBefore")
    budget_amount: str = Field(alias="budgetAmount")
    budget_window_seconds: str = Field(alias="budgetWindowSeconds")
    max_payments_per_window: str = Field(alias="maxPaymentsPerWindow")
    rate_window_seconds: str = Field(alias="rateWindowSeconds")
    mandate_id: str = Field(alias="mandateId")

    _principals = field_validator("delegator", "agent")(_address)
    _payees = field_validator("allowed_payees")(
        lambda values: tuple(_address(value) for value in values)
    )
    _numbers = field_validator(
        "max_amount_per_payment",
        "valid_after",
        "valid_before",
        "budget_amount",
        "budget_window_seconds",
        "max_payments_per_window",
        "rate_window_seconds",
    )(_decimal)
    _id = field_validator("mandate_id")(_bytes32)


class SignedMandate(WireModel):
    """A mandate and its delegator signature."""

    mandate: Mandate
    signature: str

    _signature = field_validator("signature")(
        lambda value: value if SIGNATURE.fullmatch(value) else _invalid_signature()
    )


class Confirmation(WireModel):
    """Delegator approval bound to one over-limit payment."""

    mandate_id: str = Field(alias="mandateId")
    authorization_nonce: str = Field(alias="authorizationNonce")
    amount: str
    resource: str
    valid_before: str = Field(alias="validBefore")
    signature: str

    _ids = field_validator("mandate_id", "authorization_nonce")(_bytes32)
    _numbers = field_validator("amount", "valid_before")(_decimal)
    _signature = field_validator("signature")(
        lambda value: value if SIGNATURE.fullmatch(value) else _invalid_signature()
    )


class ExactPayload(WireModel):
    """Payload for the exact payment scheme."""

    signature: str
    authorization: Authorization
    nonce_seed: str | None = Field(default=None, alias="nonceSeed")

    _signature = field_validator("signature")(
        lambda value: value if SIGNATURE.fullmatch(value) else _invalid_signature()
    )
    _seed = field_validator("nonce_seed")(lambda value: None if value is None else _bytes32(value))


class PaymentPayload(WireModel):
    """Full payment payload carried in the X-PAYMENT header."""

    x402_version: int = Field(1, alias="x402Version")
    scheme: str = "exact"
    network: str
    payload: ExactPayload
    mandate: SignedMandate | None = None
    confirmation: Confirmation | None = None


class SettlementResponse(WireModel):
    """Successful or failed settlement response."""

    success: bool
    transaction: str
    network: str
    payer: str
    error_reason: str | None = Field(default=None, alias="errorReason")

    _payer = field_validator("payer")(_address)


def _invalid_signature() -> str:
    raise ValueError("must be a 65-byte 0x-prefixed signature")
