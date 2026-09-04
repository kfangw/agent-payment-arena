"""EIP-712 mandate and confirmation signatures matching the reference gateway."""

from __future__ import annotations

from eth_account import Account
from eth_account.messages import SignableMessage, encode_typed_data

from arena.gateway.schemas import AskRequest, Confirmation, Mandate, SignedMandate

DOMAIN_NAME = "stablecoin-x402-gateway mandate"
DOMAIN_VERSION = "1"

MANDATE_FIELDS: tuple[dict[str, str], ...] = (
    {"name": "delegator", "type": "address"},
    {"name": "agent", "type": "address"},
    {"name": "maxAmountPerPayment", "type": "uint256"},
    {"name": "allowedPayees", "type": "address[]"},
    {"name": "allowedResources", "type": "string[]"},
    {"name": "validAfter", "type": "uint256"},
    {"name": "validBefore", "type": "uint256"},
    {"name": "budgetAmount", "type": "uint256"},
    {"name": "budgetWindowSeconds", "type": "uint256"},
    {"name": "maxPaymentsPerWindow", "type": "uint256"},
    {"name": "rateWindowSeconds", "type": "uint256"},
    {"name": "mandateId", "type": "bytes32"},
)
CONFIRMATION_FIELDS: tuple[dict[str, str], ...] = (
    {"name": "mandateId", "type": "bytes32"},
    {"name": "authorizationNonce", "type": "bytes32"},
    {"name": "amount", "type": "uint256"},
    {"name": "resource", "type": "string"},
    {"name": "validBefore", "type": "uint256"},
)


def _message(
    primary_type: str, fields: tuple[dict[str, str], ...], data: dict[str, object], chain_id: int
) -> SignableMessage:
    return encode_typed_data(
        full_message={
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                ],
                primary_type: list(fields),
            },
            "primaryType": primary_type,
            "domain": {"name": DOMAIN_NAME, "version": DOMAIN_VERSION, "chainId": chain_id},
            "message": data,
        }
    )


def mandate_message(mandate: Mandate, chain_id: int) -> SignableMessage:
    """Build the exact EIP-712 message signed by a mandate delegator."""
    data = mandate.model_dump(by_alias=True, mode="json")
    for name in (
        "maxAmountPerPayment",
        "validAfter",
        "validBefore",
        "budgetAmount",
        "budgetWindowSeconds",
        "maxPaymentsPerWindow",
        "rateWindowSeconds",
    ):
        data[name] = int(data[name])
    return _message("Mandate", MANDATE_FIELDS, data, chain_id)


def sign_mandate(mandate: Mandate, private_key: str, chain_id: int) -> SignedMandate:
    """Sign a mandate and return its wire representation."""
    signed = Account.sign_message(mandate_message(mandate, chain_id), private_key)
    return SignedMandate(mandate=mandate, signature="0x" + signed.signature.hex())


def recover_mandate_signer(signed: SignedMandate, chain_id: int) -> str:
    """Recover the address that signed a mandate."""
    return str(
        Account.recover_message(
            mandate_message(signed.mandate, chain_id),
            signature=bytes.fromhex(signed.signature[2:]),
        )
    )


def verify_mandate(signed: SignedMandate, chain_id: int) -> bool:
    """Return whether the declared delegator produced the signature."""
    return recover_mandate_signer(signed, chain_id).lower() == signed.mandate.delegator.lower()


def confirmation_message(request: AskRequest, valid_before: int, chain_id: int) -> SignableMessage:
    """Build a confirmation message bound to one payment request."""
    data: dict[str, object] = {
        "mandateId": request.mandate_id,
        "authorizationNonce": request.authorization_nonce,
        "amount": int(request.amount),
        "resource": request.resource,
        "validBefore": valid_before,
    }
    return _message("Confirmation", CONFIRMATION_FIELDS, data, chain_id)


def sign_confirmation(
    request: AskRequest, valid_before: int, private_key: str, chain_id: int
) -> Confirmation:
    """Sign a confirmation bound to one payment request."""
    signed = Account.sign_message(
        confirmation_message(request, valid_before, chain_id), private_key
    )
    return Confirmation(
        mandateId=request.mandate_id,
        authorizationNonce=request.authorization_nonce,
        amount=request.amount,
        resource=request.resource,
        validBefore=str(valid_before),
        signature="0x" + signed.signature.hex(),
    )


def verify_confirmation(confirmation: Confirmation, delegator: str, chain_id: int) -> bool:
    """Return whether a confirmation was signed by the delegator."""
    request = AskRequest(
        mandateId=confirmation.mandate_id,
        authorizationNonce=confirmation.authorization_nonce,
        amount=confirmation.amount,
        resource=confirmation.resource,
    )
    signer = str(
        Account.recover_message(
            confirmation_message(request, int(confirmation.valid_before), chain_id),
            signature=bytes.fromhex(confirmation.signature[2:]),
        )
    )
    return signer.lower() == delegator.lower()
