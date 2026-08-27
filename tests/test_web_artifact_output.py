from rova.app.web.artifact_output import render_research_artifact
from rova.app.context.local import LocalContextItem, LocalResearchContext
from rova.app.web.sources import ResearchSourceStore, SearchHit


def test_research_artifact_renders_answer_and_source_snapshot():
    sources = ResearchSourceStore()
    fetched = sources.register(SearchHit("Fetched source", "https://example.test/fetched", "snippet"))
    search_only = sources.register(SearchHit("Search-only source", "https://example.test/search", "snippet"))
    sources.set_content(fetched.source_id, "fetched content")

    output = render_research_artifact(
        "What is the result?",
        "The fetched finding is supported by [S1].",
        sources,
    )

    assert output.startswith("# Research Output\n")
    assert "## Request\nWhat is the result?" in output
    assert "## Answer\nThe fetched finding is supported by [S1]." in output
    assert "## Source Snapshot" in output
    assert "- [S1] Fetched source" in output
    assert "URL: https://example.test/fetched" in output
    assert "status: fetched" in output
    assert "- [S2] Search-only source" in output
    assert "URL: https://example.test/search" in output
    assert "status: search_only" in output
    assert "fetched content" not in output


def test_research_artifact_marks_absent_external_sources():
    output = render_research_artifact("Summarize this text.", "A concise summary.", ResearchSourceStore())

    assert "## Source Snapshot\nNo external sources were used." in output


def test_research_artifact_records_local_context_manifest_without_raw_content():
    local_context = LocalResearchContext((
        LocalContextItem(
            label="L1",
            filename="private-notes.md",
            byte_count=27,
            sha256="c" * 64,
            content="private local research detail",
        ),
    ))

    output = render_research_artifact(
        "Summarize my notes.",
        "Summary completed.",
        ResearchSourceStore(),
        local_context=local_context,
    )

    assert "## Local Context Manifest" in output
    assert "- [L1] private-notes.md" in output
    assert "byte_count: 27" in output
    assert f"sha256: {'c' * 64}" in output
    assert "private local research detail" not in output
