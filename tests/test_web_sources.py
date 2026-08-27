from __future__ import annotations

import pytest

from rova.app.web.sources import ResearchSourceStore, SearchHit


def test_source_store_assigns_stable_ids_and_deduplicates_urls():
    store = ResearchSourceStore()

    first = store.register(SearchHit("Python", "https://example.test/python", "first"))
    duplicate = store.register(SearchHit("Python docs", "https://example.test/python", "new"))
    second = store.register(SearchHit("Packaging", "https://example.test/packaging", "second"))

    assert first.source_id == "S1"
    assert duplicate.source_id == "S1"
    assert second.source_id == "S2"
    assert store.get("S1").snippet == "first"


def test_source_store_only_updates_registered_sources():
    store = ResearchSourceStore()
    source = store.register(SearchHit("Python", "https://example.test/python", ""))

    store.set_content(source.source_id, "extracted")

    assert store.get(source.source_id).content == "extracted"
    with pytest.raises(KeyError, match="S9"):
        store.get("S9")
