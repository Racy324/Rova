from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import httpx
from mcp.client.streamable_http import StreamableHTTPError
from exceptiongroup import ExceptionGroup

from rova.mcp.client import MCPClient, MCPConnectionError, MCPServerConfig


@dataclass
class FakeTool:
    name: str
    description: str
    input_schema: dict


@dataclass
class FakeText:
    type: str
    text: str


@dataclass
class FakeImage:
    type: str = "image"


@dataclass
class FakeListToolsResult:
    tools: list[FakeTool]


@dataclass
class FakeCallToolResult:
    content: list[object]
    structured_content: object = None
    is_error: bool = False


class FakeSession:
    async def list_tools(self):
        return FakeListToolsResult([
            FakeTool("search", "Search documents", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        ])

    async def call_tool(self, name, arguments):
        assert name == "search"
        assert arguments == {"query": "pyproject"}
        return FakeCallToolResult([FakeText("text", "first"), FakeImage(), FakeText("text", "second")], {"results": 2})


@pytest.mark.asyncio
async def test_client_lists_complete_json_schema_and_joins_all_text_blocks():
    @asynccontextmanager
    async def fake_factory(config):
        assert config.name == "tavily"
        yield FakeSession()

    async with MCPClient(MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp"), session_factory=fake_factory) as client:
        await client.initialize()
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "pyproject"})

    assert tools[0].input_schema["required"] == ["query"]
    assert result.text_blocks == ["first", "second"]
    assert result.structured_content == {"results": 2}
    assert result.is_error is False


@pytest.mark.asyncio
async def test_client_keeps_server_is_error_as_remote_tool_result():
    class ErrorSession(FakeSession):
        async def call_tool(self, name, arguments):
            return FakeCallToolResult([FakeText("text", "quota exceeded")], is_error=True)

    @asynccontextmanager
    async def fake_factory(config):
        yield ErrorSession()

    async with MCPClient(MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp"), session_factory=fake_factory) as client:
        result = await client.call_tool("search", {"query": "pyproject"})

    assert result.is_error is True
    assert result.text_blocks == ["quota exceeded"]


@pytest.mark.asyncio
async def test_client_hides_header_secrets_when_connection_fails():
    @asynccontextmanager
    async def failing_factory(config):
        raise OSError("connection refused for Bearer top-secret")
        yield None

    config = MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp", {"Authorization": "Bearer top-secret"})
    assert "top-secret" not in repr(config)
    with pytest.raises(MCPConnectionError) as captured:
        async with MCPClient(config, session_factory=failing_factory):
            pass
    assert "top-secret" not in str(captured.value)


@pytest.mark.asyncio
async def test_client_maps_sdk_http_transport_error_to_connection_error():
    @asynccontextmanager
    async def failing_factory(config):
        raise httpx.ConnectError("unreachable", request=httpx.Request("POST", config.url))
        yield None

    with pytest.raises(MCPConnectionError, match="ConnectError"):
        async with MCPClient(MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp"), session_factory=failing_factory):
            pass


@pytest.mark.asyncio
async def test_client_maps_sdk_streamable_http_error_to_connection_error():
    @asynccontextmanager
    async def failing_factory(config):
        raise StreamableHTTPError("remote transport failure")
        yield None

    with pytest.raises(MCPConnectionError, match="StreamableHTTPError"):
        async with MCPClient(MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp"), session_factory=failing_factory):
            pass


@pytest.mark.asyncio
async def test_client_maps_cleanup_exception_group_to_redacted_connection_error():
    class CleanupFailureContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, exc_type, exc, traceback):
            raise ExceptionGroup(
                "cleanup failed for Bearer top-secret",
                [StreamableHTTPError("transport failed for Bearer top-secret")],
            )

    config = MCPServerConfig(
        "tavily", "streamable_http", "https://mcp.example.test/mcp",
        {"Authorization": "Bearer top-secret"},
    )
    with pytest.raises(MCPConnectionError) as captured:
        async with MCPClient(config, session_factory=lambda _: CleanupFailureContext()):
            pass

    message = str(captured.value)
    assert "ExceptionGroup" in message
    assert "StreamableHTTPError" in message
    assert "top-secret" not in message


@pytest.mark.asyncio
async def test_client_keeps_programming_error_from_cleanup_unwrapped():
    class ProgrammingFailureContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, exc_type, exc, traceback):
            raise ValueError("programming mistake")

    with pytest.raises(ValueError, match="programming mistake"):
        async with MCPClient(
            MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp"),
            session_factory=lambda _: ProgrammingFailureContext(),
        ):
            pass


def test_server_config_rejects_query_strings_so_secret_urls_cannot_be_retained():
    with pytest.raises(ValueError, match="without query or fragment"):
        MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp?tavilyApiKey=top-secret")


def test_server_config_accepts_stdio_without_a_url():
    config = MCPServerConfig("local", "stdio", command="python", args=("server.py",), environment={"SAFE": "1"})

    assert config.url is None
    assert config.command == "python"
    assert config.args == ("server.py",)


@pytest.mark.asyncio
async def test_client_preserves_nested_mcp_input_schema_without_ai_tool_conversion():
    schema = {
        "type": "object",
        "properties": {
            "depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"},
            "urls": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object", "properties": {"domain": {"type": "string"}}},
        },
        "required": ["urls"],
    }

    class SchemaSession(FakeSession):
        async def list_tools(self):
            return FakeListToolsResult([FakeTool("extract", "Extract", schema)])

    @asynccontextmanager
    async def fake_factory(config):
        yield SchemaSession()

    async with MCPClient(MCPServerConfig("tavily", "streamable_http", "https://mcp.example.test/mcp"), session_factory=fake_factory) as client:
        tools = await client.list_tools()

    assert tools[0].input_schema == schema
