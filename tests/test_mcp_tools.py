from __future__ import annotations

import pytest

from rova.mcp.client import MCPCallResult, MCPToolDefinition
from rova.mcp.tools import MCPToolAdapterError, build_mcp_tools
from rova.app.workspace.approval import AlwaysApprove, AlwaysDeny
from rova.app.workspace.controlled_tool import ControlledTool
from rova.app.workspace.policy import DefaultCodingToolPolicy


def test_mcp_tools_have_stable_public_names_and_raw_mapping() -> None:
    tools = build_mcp_tools(
        "Google Scholar",
        [MCPToolDefinition("search-papers", "Search", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})],
        include_tools=("search-papers",),
        call_tool=lambda _name, _args: None,
    )
    assert tools[0].tool.name == "mcp_google_scholar_search_papers"
    assert tools[0].metadata == {"origin": "mcp", "server_id": "Google Scholar", "raw_tool_name": "search-papers", "public_name": "mcp_google_scholar_search_papers"}


def test_normalization_collision_rejects_entire_server_batch() -> None:
    definitions = [
        MCPToolDefinition("foo-bar", "one", {"type": "object"}),
        MCPToolDefinition("foo_bar", "two", {"type": "object"}),
    ]
    with pytest.raises(MCPToolAdapterError, match="normalization collision"):
        build_mcp_tools("server", definitions, include_tools=("foo-bar", "foo_bar"), call_tool=lambda _name, _args: None)


@pytest.mark.asyncio
async def test_mcp_adapter_uses_raw_name_and_returns_tool_error() -> None:
    calls = []

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        return MCPCallResult(["untrusted external content"], {"count": 1}, False)

    tool = build_mcp_tools("github", [MCPToolDefinition("search_code", "Search", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})], include_tools=("search_code",), call_tool=call_tool)[0]
    result = await tool.execute("call", {"query": "rova"})
    assert calls == [("search_code", {"query": "rova"})]
    assert "untrusted external content" in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_tool_is_denied_by_the_same_approval_wrapper_before_client_call() -> None:
    calls = []
    async def call_tool(name, arguments):
        calls.append((name, arguments))
        return MCPCallResult([], None, False)
    inner = build_mcp_tools("github", [MCPToolDefinition("create_issue", "Create", {"type": "object"})], include_tools=("create_issue",), call_tool=call_tool)[0]
    controlled = ControlledTool(inner, DefaultCodingToolPolicy(), AlwaysDeny())
    with pytest.raises(Exception, match="not approved"):
        await controlled.execute("call", {})
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_identity_metadata_is_preserved_after_governed_execution() -> None:
    async def call_tool(name, arguments):
        return MCPCallResult(["ok"], None, False)

    inner = build_mcp_tools(
        "github", [MCPToolDefinition("search", "Search", {"type": "object"})],
        include_tools=("search",), call_tool=call_tool,
    )[0]
    result = await ControlledTool(inner, DefaultCodingToolPolicy(), AlwaysApprove()).execute("call", {})

    assert result.metadata["origin"] == "mcp"
    assert result.metadata["server_id"] == "github"
    assert result.metadata["raw_tool_name"] == "search"
