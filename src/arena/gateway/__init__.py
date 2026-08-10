"""Gateway backends and the contract they are both required to satisfy.

Two implementations exist. `FakeGateway` is in-memory and is the default: it
reproduces the x402 *contract* without reproducing the gateway's
implementation, so the arena runs with no chain, no node, and no Go toolchain.
`HttpGateway` talks to a real stablecoin-x402-gateway over HTTP.

A contract test runs the same scenarios against both and asserts they agree.
That test is what makes results produced against the fake backend worth
anything.
"""

from arena.gateway.contract import Action, ErrorCode

__all__ = ["Action", "ErrorCode"]
