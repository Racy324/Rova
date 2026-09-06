from __future__ import annotations

import asyncio

import pytest

from rova.agent_core.tools import ToolRegistry
from rova.mcp.client import MCPCallResult, MCPToolDefinition
from rova.mcp.config import MCPServerSettings
from rova.mcp.manager import MCPManager


class FakeClient:
    def __init__(self, tools, gate=None): self.tools, self.gate, self.closed = tools, gate, False
    async def initialize(self):
        if self.gate: await self.gate.wait()
    async def list_tools(self): return self.tools
    async def call_tool(self, name, arguments): return MCPCallResult([], None, False)
    async def close(self): self.closed = True


@pytest.mark.asyncio
async def test_fast_server_registers_without_waiting_for_slow_server() -> None:
    gate = asyncio.Event()
    fast = MCPServerSettings("fast", "stdio", ("search",), command="fake")
    slow = MCPServerSettings("slow", "stdio", ("search",), command="fake")
    clients = {"fast": FakeClient([MCPToolDefinition("search", "Search", {"type": "object"})]), "slow": FakeClient([MCPToolDefinition("search", "Search", {"type": "object"})], gate)}
    registry = ToolRegistry([])
    manager = MCPManager((fast, slow), registry, lambda setting: clients[setting.server_id])

    task = manager.start()
    for _ in range(20):
        if registry.schemas: break
        await asyncio.sleep(0)
    assert [tool.name for tool in registry.schemas] == ["mcp_fast_search"]
    assert registry._tools["mcp_fast_search"].metadata["origin"] == "mcp"
    assert not task.done()
    await manager.close()
    assert clients["fast"].closed and clients["slow"].closed


@pytest.mark.asyncio
async def test_failed_server_is_isolated_and_redacts_configured_values() -> None:
    server = MCPServerSettings("broken", "stdio", ("search",), command="fake", environment={"TOKEN": "private-value"})

    class BrokenClient(FakeClient):
        async def initialize(self):
            raise RuntimeError("connection used private-value")

    registry = ToolRegistry([])
    manager = MCPManager((server,), registry, lambda _setting: BrokenClient([]))
    await manager.start()

    assert manager.server_states == {"broken": "failed"}
    assert "private-value" not in manager.issues[0].message
    assert registry.schemas == []
