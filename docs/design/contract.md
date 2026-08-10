# Contract

## Purpose

State what a gateway must do, in terms that hold for both backends. The
arena's results are only transferable if the fake backend and the real one
answer the same question the same way, and that is a claim about a contract,
not about either implementation.

The contract is also where the arena's independence is enforced. Every fact
the arena knows about the gateway passes through this module. Nothing else
imports gateway internals, so a change on the gateway side has one place to
land.

## Behavior

The contract covers four things.

**The 402 response.** What a gateway sends when a request arrives without
payment: the amount, the payee, the resource identifier, the error code, and
whatever metadata accompanies them. The arena cares which of these fields an
adversary controls, because a controlled field is an injection point.

**Refusal codes.** Stable snake_case strings carried alongside the
human-readable message, so a client can branch on cause without parsing prose.
The arena treats a wrong code as a contract violation even when the decision
was right, because the agent's recovery path is selected by the code.

**The mandate.** The delegator's signed authorization: scope, limits, expiry,
allowed payees and resources. Reproduced with its EIP-712 types rather than
abstracted, for the reason given below.

**The ask flow.** The confirmation a delegator signs when a payment falls
outside the mandate, and the binding that keeps that confirmation from
applying to any other payment.

## Design decisions

**The decision space is shared with the reference gateway.** Five outcomes:
approve, reject, defer, ask, bond. The arena did not invent a vocabulary of
its own. Sharing it is what allows a policy written for the gateway to be
evaluated here, and a finding here to be carried back as a change there.
Diverging would make every result a translation.

**Signatures are real.** The fake backend verifies EIP-712 signatures through
`eth-account` rather than treating a signature as an opaque valid or invalid
token. The condition this repository measures is a payment whose signature is
valid and whose purpose is not authorized. If the signature were stubbed, that
condition would be an assertion in the test harness rather than something the
gateway actually observed, and the result would be about the stub.

**The contract is written down separately from both implementations.** Not
derived from the fake backend, and not a Python translation of the Go gateway.
A contract inferred from one implementation cannot detect that implementation
being wrong.

## Limits

The contract covers what a payment attempt looks like and how it is answered.
It does not cover settlement mechanics, gas, reorgs, or anything else that
happens after a gateway decides to settle. Results that depend on those belong
to the HTTP backend.

Fidelity to the reference gateway is asserted by tests, not proven. Until the
HTTP backend exists, the contract records the reference implementation as read
at a specific commit, and drift after that commit is invisible.

## Status

The decision space and the refusal codes are implemented and pinned by tests
in `src/arena/gateway/contract.py`, transcribed from the reference gateway's
`x402/errcodes.go` and `x402/policy.go`.

The 402 payload, the mandate schema, and the ask flow are not written. They
land in M0, from `x402/gateway.go`, `x402/mandate.go`, and
`x402/confirmation.go` respectively.
