# Backends

## Purpose

Let the arena run with no external dependency, without giving up the claim
that its results hold against a real gateway. These two goals pull in opposite
directions, and the resolution is two backends plus a test that ties them
together.

## Behavior

`FakeGateway` is in-memory and is the default. It answers an unpaid request
with a 402, verifies the EIP-712 signature on the payment that follows,
applies the accept policy, and reports the outcome with a refusal code. It
keeps whatever state a policy needs, such as spend within a window and the
count of confirmations already requested. It settles nothing, because there is
no chain.

`HttpGateway` speaks to a running stablecoin-x402-gateway over HTTP. It
carries the same operations across the wire and returns the same result types,
so the run loop cannot tell which backend it holds.

The differential test runs the same scenarios through both and asserts the
outcome and the refusal code agree.

## Design decisions

**The fake reproduces the contract, not the implementation.** Its job is to
answer the same way, not to compute the answer the same way. Porting the
gateway's logic to Python would create two implementations of one policy that
drift apart, and would make the differential test tautological rather than
informative.

**The fake is the default, not the fallback.** A reader who clones this
repository runs the evaluation without Docker, without a node, and without a
Go toolchain. An instrument that is inconvenient to run does not get run, and
results nobody reproduces are not results.

**Signature verification is not faked.** See
[contract.md](contract.md). This is the one place the fake backend does real
cryptographic work.

**The differential test is the load-bearing one.** Everything else in this
repository produces numbers; that test is what makes the numbers about
something other than the fake backend. It is written first among the tests
that touch both backends, and a failure in it blocks a release rather than
being triaged.

## Limits

The fake backend settles nothing. Settlement failure, gas exhaustion, reorgs,
and delivery after settlement are outside it, and any result that depends on
them has to be reproduced against the HTTP backend.

The differential test covers the scenarios written for it. Agreement on those
is not agreement in general, and the gap grows as the fake accumulates
behavior the test does not exercise.

The HTTP backend pins one gateway commit. Contract drift after that commit
appears as a differential failure, which is the intended signal, but only when
someone runs it.

## Status

Both backends are implemented. `HttpGateway` maps the shared operations to the
reference server's 402 body and base64 `X-PAYMENT` header. The default test
suite uses `FakeGateway`; a separate CI job checks out reference revision
`ddd20fe3ec7c2109a006e1112bcac9fdeabf9b32`, starts its Compose stack, and
runs the valid signed-payment differential test against a real node.
