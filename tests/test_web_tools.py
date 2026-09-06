from __future__ import annotations

import pytest

from rova.ai.messages import ToolCall
from rova.agent_core.tools import ToolRegistry, ToolRuntime
from rova.app.web.sources import FetchedPage, ResearchSourceStore, SearchHit
from rova.app.web.tools import create_fetch_webpage_tool, create_web_search_tool
from rova.app.web.http import WebNetworkError


class FakeSearch:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        assert query == "pyproject"
        assert max_results == 3
        return [SearchHit("Python docs", "https://docs.python.org/", "docs")]


class FakeFetcher:
    async def fetch(self, url: str) -> FetchedPage:
        assert url == "https://docs.python.org/"
        return FetchedPage("Python docs", "Readable text")


@pytest.mark.asyncio
async def test_search_registers_sources_and_fetch_uses_source_id_only():
    store = ResearchSourceStore()
    search = create_web_search_tool(store, FakeSearch())
    search_result = await search.execute("search-1", {"query": "pyproject", "max_results": 3})

    assert "[S1]" in search_result.content[0].text
    assert "status=search_only" in search_result.content[0].text
    assert "must not be used as a final evidence citation" in search_result.content[0].text
    fetch = create_fetch_webpage_tool(store, FakeFetcher())
    fetch_result = await fetch.execute("fetch-1", {"source_id": "S1"})
    assert "SOURCE S1" in fetch_result.content[0].text
    assert "status=fetched" in fetch_result.content[0].text
    assert "may be used as a final evidence citation" in fetch_result.content[0].text
    assert store.get("S1").content == "Readable text"


@pytest.mark.asyncio
async def test_fetch_returns_normal_tool_error_for_network_failure():
    class FailingFetcher:
        async def fetch(self, url: str):
            raise WebNetworkError("HTTP 404")

    store = ResearchSourceStore()
    store.register(SearchHit("Python docs", "https://docs.python.org/", "docs"))
    result = await ToolRuntime(ToolRegistry([create_fetch_webpage_tool(store, FailingFetcher())])).execute(
        ToolCall("fetch-1", "fetch_webpage", {"source_id": "S1"})
    )

    assert result.is_error is True
    assert "HTTP 404" in result.text


@pytest.mark.asyncio
async def test_fetch_does_not_turn_programming_errors_into_tool_observations():
    class BrokenFetcher:
        async def fetch(self, url: str):
            raise TypeError("implementation defect")

    store = ResearchSourceStore()
    store.register(SearchHit("Python docs", "https://docs.python.org/", "docs"))
    with pytest.raises(TypeError, match="implementation defect"):
        await create_fetch_webpage_tool(store, BrokenFetcher()).execute("fetch-1", {"source_id": "S1"})
