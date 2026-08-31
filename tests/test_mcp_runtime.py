from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock
from rova.ai.models import Model
from rova.app import runtime as runtime_module
from rova.app.runtime import build_rova_runtime
from rova.mcp.client import MCPCallResult, MCPToolDefinition


class _FakeClient:
    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate
        self.closed = False

    async def initialize(self) -> None:
        await self.gate.wait()

    async def list_tools(self):
        return [MCPToolDefinition("search", "Search", {"type": "object"})]

    async def call_tool(self, name: str, arguments: dict):
        return MCPCallResult(["ok"], None, False)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_runtime_lazily_starts_mcp_discovery_and_closes_clients(monkeypatch, tmp_path: Path):
    config = tmp_path / "mcp.toml"
    config.write_text("[mcp_servers.demo]\nenabled = true\ntransport = 'stdio'\ncommand = 'fake-server'\ninclude_tools = ['search']\n", encoding="utf-8")
    gate = asyncio.Event()
    fake_client = _FakeClient(gate)
    monkeypatch.setattr(runtime_module, "_create_mcp_client", lambda _server: fake_client)

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, permission_mode="full",
        mcp_config_path=config, session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )
    assert runtime.mcp_manager is not None
    assert runtime.mcp_manager._task is None

    responses = await runtime.prompt("hello")
    assert responses[-1].text == "done"
    assert runtime.mcp_manager._task is not None
    assert "mcp_demo_search" not in runtime.agent.registry._tools

    gate.set()
    for _ in range(20):
        if "mcp_demo_search" in runtime.agent.registry._tools:
            break
        await asyncio.sleep(0)
    assert "mcp_demo_search" in runtime.agent.registry._tools
    await runtime.close()
    assert fake_client.closed


def test_runtime_only_creates_mcp_manager_when_explicit_config_is_supplied(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, permission_mode="full",
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    assert runtime.mcp_manager is None
