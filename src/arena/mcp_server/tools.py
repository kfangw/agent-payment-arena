"""Payment tool service and optional MCP transport adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol, cast

from arena.gateway.protocol import GatewayResult
from arena.gateway.schemas import Confirmation
from arena.payment import PaymentAuthority
from arena.scenarios import Resource


class ToolRegistrar(Protocol):
    """Small portion of FastMCP used to register tool functions."""

    def tool(self) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Return a tool registration decorator."""
        ...


class PaymentTools:
    """Transport-independent implementation of the agent's payment tools."""

    def __init__(self, resource: Resource, authority: PaymentAuthority, now: datetime) -> None:
        self.resource = resource
        self.authority = authority
        self.now = now

    def fetch_resource(self) -> dict[str, object]:
        """Return content and payment terms for the protected resource."""
        terms = self.authority.gateway.requirements(self.resource.url)
        return {
            "url": self.resource.url,
            "content": self.resource.content,
            "paymentRequired": terms.model_dump(by_alias=True, mode="json"),
        }

    def inspect_mandate(self) -> dict[str, object]:
        """Return the agent's signed spending authority without ground truth."""
        return self.authority.signed_mandate.model_dump(by_alias=True, mode="json")

    def ask_delegator(self, amount: int, nonce_key: str) -> dict[str, object] | None:
        """Request a confirmation bound to one prospective payment."""
        if self.authority.delegator is None:
            return None
        request = self.authority.ask_request(self.resource.url, amount, nonce_key)
        confirmation = self.authority.delegator.confirm(request, now=self.now)
        if confirmation is None:
            return None
        return cast(dict[str, object], confirmation.model_dump(by_alias=True, mode="json"))

    def pay(
        self,
        payee: str,
        amount: int,
        nonce_key: str,
        confirmation: dict[str, object] | None = None,
    ) -> GatewayResult:
        """Attempt one signed payment."""
        return self.authority.pay(
            self.resource.url,
            payee,
            amount,
            now=self.now,
            nonce_key=nonce_key,
            confirmation=None
            if confirmation is None
            else Confirmation.model_validate(confirmation),
        )


def register_tools(server: ToolRegistrar, tools: PaymentTools) -> None:
    """Register payment operations on an MCP-compatible server."""
    server.tool()(tools.fetch_resource)
    server.tool()(tools.inspect_mandate)
    server.tool()(tools.ask_delegator)
    server.tool()(tools.pay)


def create_mcp_server(tools: PaymentTools) -> object:
    """Create a FastMCP server when the optional MCP dependency is installed."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("agent-payment-arena")
    register_tools(server, tools)
    return server
