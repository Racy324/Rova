from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rova.agent_core.tools import ToolRegistry

from .config import MCPServerSettings
from .tools import build_mcp_tools


class MCPClientProtocol(Protocol):
    async def initialize(self) -> None: ...
    async def list_tools(self): ...
    async def call_tool(self, name: str, arguments: dict): ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class MCPIssue:
    server_id: str
    message: str


class MCPManager:
    def __init__(self, servers: Sequence[MCPServerSettings], registry: ToolRegistry, client_factory: Callable[[MCPServerSettings], MCPClientProtocol]) -> None:
        self._servers, self._registry = tuple(servers), registry
        self._client_factory = client_factory
        self._task: asyncio.Task[None] | None = None
        self._clients: list[MCPClientProtocol] = []
        self.issues: list[MCPIssue] = []
        self.server_states: dict[str, str] = {server.server_id: "pending" for server in self._servers}

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(self._discover_all())
        return self._task

    async def _discover_all(self) -> None:
        await asyncio.gather(*(self._discover_server(server) for server in self._servers), return_exceptions=True)

    async def _discover_server(self, server: MCPServerSettings) -> None:
        self.server_states[server.server_id] = "connecting"
        try:
            client = self._client_factory(server)
            self._clients.append(client)
            await client.initialize()
            tools = build_mcp_tools(server.server_id, await client.list_tools(), include_tools=server.include_tools, call_tool=client.call_tool)
            if not tools:
                raise ValueError("include_tools matched no discovered MCP tools")
            self._registry.register_tools(tools)
            self.server_states[server.server_id] = "ready"
        except asyncio.CancelledError:
            self.server_states[server.server_id] = "cancelled"
            raise
        except Exception as error:
            self.server_states[server.server_id] = "failed"
            self.issues.append(MCPIssue(server.server_id, _safe_issue_message(server, error)))

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for client in reversed(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()


def _safe_issue_message(server: MCPServerSettings, error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    secrets = [*server.headers.values(), *server.environment.values()]
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    return message[:600]
