from __future__ import annotations

from ..context.local import LocalResearchContext
from .sources import ResearchSourceStore


def render_research_artifact(
    request: str,
    final_response: str,
    sources: ResearchSourceStore,
    *,
    local_context: LocalResearchContext | None = None,
) -> str:
    lines = ["# Research Output", "", "## Request", request, "", "## Answer", final_response, "", "## Source Snapshot"]
    records = sources.all()
    if not records:
        lines.append("No external sources were used.")
    else:
        for source in records:
            status = "fetched" if source.content is not None else "search_only"
            lines.extend([f"- [{source.source_id}] {source.title}", f"  - URL: {source.url}", f"  - status: {status}"])
    if local_context is not None and local_context.items:
        lines.extend(["", "## Local Context Manifest"])
        for item in local_context.items:
            lines.extend([
                f"- [{item.label}] {item.filename}",
                f"  - byte_count: {item.byte_count}",
                f"  - sha256: {item.sha256}",
            ])
    return "\n".join(lines) + "\n"
