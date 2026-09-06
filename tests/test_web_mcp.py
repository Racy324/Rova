from __future__ import annotations

import pytest

from rova.app.web.mcp_backends import McpWebFetchBackend, McpWebSearchBackend
from rova.app.web.sources import FetchedPage
from rova.app.web.tools import create_fetch_webpage_tool
from rova.app.web.http import WebNetworkError
from rova.agent_core.tools import ToolRegistry, ToolRuntime
from rova.ai.messages import ToolCall
from rova.mcp.client import MCPCallResult, MCPConnectionError


class FakeMCPClient:
    def __init__(self, result: MCPCallResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


@pytest.mark.asyncio
async def test_mcp_search_backend_maps_structured_results_to_existing_search_hits():
    client = FakeMCPClient(MCPCallResult([], {
        "results": [{"title": "PyPA", "url": "https://packaging.python.org/", "content": "Packaging guidance"}],
    }, False))

    hits = await McpWebSearchBackend(lambda: client).search("pyproject", 3)

    assert [(item.title, item.url, item.snippet) for item in hits] == [("PyPA", "https://packaging.python.org/", "Packaging guidance")]
    assert client.calls == [("tavily_search", {"query": "pyproject", "max_results": 3})]


@pytest.mark.asyncio
async def test_mcp_fetch_backend_maps_structured_extract_result_to_domain_page():
    client = FakeMCPClient(MCPCallResult([], {
        "results": [{"url": "https://packaging.python.org/", "title": "PyPA", "raw_content": "Readable page"}],
    }, False))

    page = await McpWebFetchBackend(lambda: client).fetch("https://packaging.python.org/")

    assert page == FetchedPage("PyPA", "Readable page")
    assert client.calls == [("tavily_extract", {"urls": ["https://packaging.python.org/"]})]


@pytest.mark.asyncio
async def test_mcp_fetch_backend_rejects_a_result_for_a_different_url_to_preserve_provenance():
    client = FakeMCPClient(MCPCallResult([], {
        "results": [{"url": "https://other.example/", "title": "Other", "raw_content": "Wrong page"}],
    }, False))

    with pytest.raises(WebNetworkError, match="contains no result"):
        await McpWebFetchBackend(lambda: client).fetch("https://packaging.python.org/")


@pytest.mark.asyncio
async def test_mcp_backend_maps_remote_tool_error_to_research_tool_failure():
    client = FakeMCPClient(MCPCallResult(["TAVILY_API_KEY=should-not-leak"], None, True))

    with pytest.raises(WebNetworkError, match="MCP search tool returned an error") as captured:
        await McpWebSearchBackend(lambda: client).search("pyproject", 3)

    assert "should-not-leak" not in str(captured.value)


@pytest.mark.asyncio
async def test_mcp_cleanup_connection_error_becomes_regular_research_tool_error():
    class CleanupErrorClient(FakeMCPClient):
        async def __aexit__(self, exc_type, exc, traceback):
            raise MCPConnectionError("MCP connection to tavily failed (ExceptionGroup: [StreamableHTTPError])")

    from rova.app.web.sources import ResearchSourceStore, SearchHit

    store = ResearchSourceStore()
    source = store.register(SearchHit("PyPA", "https://packaging.python.org/", "packaging"))
    registry = ToolRegistry([
        create_fetch_webpage_tool(
            store,
            McpWebFetchBackend(lambda: CleanupErrorClient(MCPCallResult([], None, False))),
        )
    ])

    result = await ToolRuntime(registry).execute(ToolCall("fetch", "fetch_webpage", {"source_id": source.source_id}))

    assert result.is_error is True
    assert result.metadata["outcome"] == "tool_execution_error"
    assert "ExceptionGroup" in result.text
