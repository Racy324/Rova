from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class FetchedPage:
    title: str
    content: str


@dataclass
class ResearchSource:
    source_id: str
    url: str
    title: str
    snippet: str = ""
    content: str | None = None


class ResearchSourceStore:
    """Run-local source identity and fetched-content store."""

    def __init__(self) -> None:
        self._sources: dict[str, ResearchSource] = {}
        self._source_ids_by_url: dict[str, str] = {}

    def register(self, hit: SearchHit) -> ResearchSource:
        existing = self._source_ids_by_url.get(hit.url)
        if existing is not None:
            return self._sources[existing]
        source_id = f"S{len(self._sources) + 1}"
        source = ResearchSource(source_id, hit.url, hit.title, hit.snippet)
        self._sources[source_id] = source
        self._source_ids_by_url[hit.url] = source_id
        return source

    def get(self, source_id: str) -> ResearchSource:
        return self._sources[source_id]

    def set_content(self, source_id: str, content: str) -> ResearchSource:
        source = self.get(source_id)
        source.content = content
        return source

    def all(self) -> list[ResearchSource]:
        return list(self._sources.values())
