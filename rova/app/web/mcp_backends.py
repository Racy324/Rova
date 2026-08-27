from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from rova.mcp.client import MCPCallResult, MCPClient, MCPConnectionError, MCPProtocolError

from .sources import FetchedPage, SearchHit
from .http import WebNetworkError


MCPClientFactory = Callable[[], AbstractAsyncContextManager[MCPClient]]


class McpWebSearchBackend:
    """Tavily MCP adapter that returns the existing Research search domain model."""

    def __init__(self, client_factory: MCPClientFactory, *, tool_name: str = "tavily_search") -> None:
        self._client_factory = client_factory
        self._tool_name = tool_name

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        result = await _call(self._client_factory, self._tool_name, {"query": query, "max_results": max_results}, "search")
        items = _result_items(result, "search")
        hits: list[SearchHit] = []
        for item in items:
            title, url = item.get("title"), item.get("url")
            snippet = item.get("content", item.get("snippet", ""))
            if not isinstance(title, str) or not isinstance(url, str) or not isinstance(snippet, str):
                raise WebNetworkError("MCP search response contains an invalid result")
            hits.append(SearchHit(title, url, snippet))
        return hits[:max_results]


class McpWebFetchBackend:
    """Tavily MCP adapter that returns the existing Research fetched-page model."""

    def __init__(self, client_factory: MCPClientFactory, *, tool_name: str = "tavily_extract") -> None:
        self._client_factory = client_factory
        self._tool_name = tool_name

    async def fetch(self, url: str) -> FetchedPage:
        result = await _call(self._client_factory, self._tool_name, {"urls": [url]}, "extract")
        items = _result_items(result, "extract")
        item = next((candidate for candidate in items if candidate.get("url") == url), None)
        if not isinstance(item, dict):
            raise WebNetworkError("MCP extract response contains no result")
        content = item.get("raw_content", item.get("content"))
        title = item.get("title", "")
        if not isinstance(title, str) or not isinstance(content, str):
            raise WebNetworkError("MCP extract response contains an invalid result")
        return FetchedPage(title, content)


async def _call(client_factory: MCPClientFactory, tool_name: str, arguments: dict[str, Any], operation: str) -> MCPCallResult:
    try:
        async with client_factory() as client:
            result = await client.call_tool(tool_name, arguments)
    except (MCPConnectionError, MCPProtocolError) as error:
        raise WebNetworkError(f"MCP {operation} backend failure: {error}") from error
    if result.is_error:
        raise WebNetworkError(f"MCP {operation} tool returned an error")
    return result


def _result_items(result: MCPCallResult, operation: str) -> list[dict[str, Any]]:
    content = result.structured_content
    if isinstance(content, dict):
        items = content.get("results")
    else:
        items = content
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise WebNetworkError(f"MCP {operation} response has no structured results")
    return items
