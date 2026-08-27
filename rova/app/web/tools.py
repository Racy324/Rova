from __future__ import annotations

from typing import Protocol

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError

from .sources import FetchedPage, ResearchSourceStore, SearchHit
from .http import UnsafeUrlError, WebNetworkError


class WebSearchBackend(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class WebFetchBackend(Protocol):
    async def fetch(self, url: str) -> FetchedPage: ...


def create_web_search_tool(store: ResearchSourceStore, backend: WebSearchBackend) -> AgentTool:
    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        try:
            hits = await backend.search(params["query"], params.get("max_results", 5))
        except WebNetworkError as error:
            raise ToolExecutionError(str(error), metadata={"outcome": "tool_execution_error"}) from error
        sources = [store.register(hit) for hit in hits]
        text = "\n\n".join(
            f"[{source.source_id}]\nstatus=search_only\nTitle: {source.title}\nURL: {source.url}\nSnippet: {source.snippet}"
            for source in sources
        )
        if text:
            text = (
                "Search results are registered sources only. Each status=search_only source must not be used as a final "
                "evidence citation; fetch it first with fetch_webpage.\n\n"
                + text
            )
        return AgentToolResult([TextBlock(text or "No results.")], {"result_count": len(sources)})

    return AgentTool(Tool("web_search", "Search public web pages and register returned sources.", {"query": str, "max_results": int}, required=("query",)), execute)


def create_fetch_webpage_tool(store: ResearchSourceStore, fetcher: WebFetchBackend) -> AgentTool:
    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        try:
            source = store.get(params["source_id"])
        except KeyError as error:
            raise ToolExecutionError(f"Unknown source ID: {params['source_id']}", metadata={"outcome": "tool_input_error"}) from error
        try:
            page = await fetcher.fetch(source.url)
        except (WebNetworkError, UnsafeUrlError) as error:
            raise ToolExecutionError(str(error), metadata={"outcome": "tool_execution_error"}) from error
        source.title = page.title or source.title
        store.set_content(source.source_id, page.content)
        return AgentToolResult(
            [
                TextBlock(
                    f"SOURCE {source.source_id}\nstatus=fetched\nThis source may be used as a final evidence citation.\n"
                    f"Title: {source.title}\nURL: {source.url}\n\nExtracted content:\n{page.content}"
                )
            ]
        )

    return AgentTool(Tool("fetch_webpage", "Fetch one already registered public-web source by source ID.", {"source_id": str}), execute)
