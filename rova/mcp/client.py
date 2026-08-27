from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Protocol
from urllib.parse import urlsplit

import httpx
from mcp.client.streamable_http import StreamableHTTPError
from mcp.shared.exceptions import MCPError

try:
    from builtins import BaseExceptionGroup
except ImportError:
    from exceptiongroup import BaseExceptionGroup


class MCPConnectionError(RuntimeError):
    """The MCP transport could not be established or retained."""


class MCPProtocolError(RuntimeError):
    """The remote MCP server returned a protocol-level failure."""


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.transport != "streamable_http":
            raise ValueError("only streamable_http transport is supported")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("MCP server URL must be an http/https URL without credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("MCP server URL must be without query or fragment")
        object.__setattr__(self, "headers", dict(self.headers))


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPCallResult:
    text_blocks: list[str]
    structured_content: Any
    is_error: bool


class _MCPToolSession(Protocol):
    async def list_tools(self): ...

    async def call_tool(self, name: str, arguments: dict[str, Any]): ...


SessionFactory = Callable[[MCPServerConfig], AsyncContextManager[_MCPToolSession]]


class MCPClient:
    """Thin lifecycle wrapper over the official Streamable HTTP MCP SDK client."""

    def __init__(self, config: MCPServerConfig, *, session_factory: SessionFactory | None = None) -> None:
        self.config = config
        self._session_factory = session_factory or _sdk_session_factory
        self._context: AsyncContextManager[_MCPToolSession] | None = None
        self._session: _MCPToolSession | None = None

    async def __aenter__(self) -> "MCPClient":
        self._context = self._session_factory(self.config)
        try:
            self._session = await self._context.__aenter__()
        except Exception as error:
            _raise_normalized_error(self.config, error)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        context, self._context, self._session = self._context, None, None
        if context is not None:
            try:
                await context.__aexit__(exc_type, exc, traceback)
            except Exception as error:
                _raise_normalized_error(self.config, error)

    async def initialize(self) -> None:
        """Confirm the SDK-managed handshake has already completed."""
        self._require_session()

    async def list_tools(self) -> list[MCPToolDefinition]:
        try:
            result = await self._require_session().list_tools()
        except Exception as error:
            _raise_normalized_error(self.config, error)
        return [MCPToolDefinition(item.name, item.description or "", dict(item.input_schema)) for item in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        try:
            result = await self._require_session().call_tool(name, arguments)
        except Exception as error:
            _raise_normalized_error(self.config, error)
        text_blocks = [item.text for item in result.content if getattr(item, "type", None) == "text"]
        return MCPCallResult(text_blocks, result.structured_content, result.is_error)

    def _require_session(self) -> _MCPToolSession:
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        return self._session


@asynccontextmanager
async def _sdk_session_factory(config: MCPServerConfig):
    import httpx
    from mcp.client import Client
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(headers=dict(config.headers)) as http_client:
        transport = streamable_http_client(config.url, http_client=http_client)
        async with Client(transport) as session:
            yield session


def _safe_connection_message(config: MCPServerConfig, error: BaseException) -> str:
    return f"MCP connection to {config.name} failed ({_safe_error_summary(config, error)})"


def _safe_protocol_message(config: MCPServerConfig, error: BaseException) -> str:
    return f"MCP protocol operation on {config.name} failed ({_safe_error_summary(config, error)})"


def _raise_normalized_error(config: MCPServerConfig, error: Exception) -> None:
    category = _known_error_category(error)
    if category == "protocol":
        raise MCPProtocolError(_safe_protocol_message(config, error)) from error
    if category == "connection":
        raise MCPConnectionError(_safe_connection_message(config, error)) from error
    raise error


def _known_error_category(error: BaseException) -> str | None:
    leaves = _exception_leaves(error)
    if not leaves:
        return None
    if not all(isinstance(item, (MCPError, StreamableHTTPError, httpx.HTTPError, OSError, TimeoutError)) for item in leaves):
        return None
    return "protocol" if any(isinstance(item, MCPError) for item in leaves) else "connection"


def _exception_leaves(error: BaseException) -> tuple[BaseException, ...]:
    if isinstance(error, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for nested in error.exceptions:
            leaves.extend(_exception_leaves(nested))
        return tuple(leaves)
    return (error,)


def _safe_error_summary(config: MCPServerConfig, error: BaseException) -> str:
    outer = f"{type(error).__name__}: {_safe_text(config, str(error))}"
    if not isinstance(error, BaseExceptionGroup):
        return outer[:600]
    nested = "; ".join(
        f"{type(item).__name__}: {_safe_text(config, str(item))}"
        for item in _exception_leaves(error)
    )
    return f"{outer}; nested=[{nested}]"[:600]


def _safe_text(config: MCPServerConfig, value: str) -> str:
    secrets = set(config.headers.values())
    secrets.update(value.rsplit(" ", 1)[-1] for value in config.headers.values() if " " in value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value
