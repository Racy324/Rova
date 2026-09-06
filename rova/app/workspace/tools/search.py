from __future__ import annotations

import asyncio
import os
from pathlib import Path

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..workspace import CodingToolError, Workspace


DEFAULT_MAX_RESULTS = 100


def create_search_tool(workspace: Workspace) -> AgentTool:
    def search(query: str, path: str, max_results: int) -> str:
        if not query:
            raise CodingToolError("query must not be empty")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise CodingToolError("max_results must be at least 1")
        target = workspace.resolve(path)
        if not target.exists():
            raise CodingToolError("path not found")
        candidates = _candidate_files(target)
        matches: list[str] = []
        for candidate in candidates:
            try:
                text = workspace.read_text(candidate)
            except CodingToolError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(f"{workspace.display_path(candidate)}:{line_number}: {line}")
                    if len(matches) == max_results:
                        return "\n".join(matches)
        return "\n".join(matches) if matches else "no results"

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result = await asyncio.to_thread(
            search,
            params["query"],
            params.get("path", "."),
            params.get("max_results", DEFAULT_MAX_RESULTS),
        )
        return AgentToolResult([TextBlock(result)])

    return AgentTool(
        Tool("search", "Search UTF-8 workspace files for exact text", {"query": str, "path": str, "max_results": int}, required=("query",)),
        execute,
        execution_mode=ToolExecutionMode.PARALLEL,
    )


def _candidate_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise CodingToolError("not a file or directory")
    candidates: list[Path] = []
    for current, directory_names, file_names in os.walk(target, followlinks=False):
        current_path = Path(current)
        directory_names[:] = sorted(
            name for name in directory_names
            if name != ".git" and not (current_path / name).is_symlink()
        )
        for file_name in sorted(file_names):
            candidate = current_path / file_name
            if not candidate.is_symlink():
                candidates.append(candidate.resolve())
    return sorted(candidates, key=lambda candidate: candidate.as_posix())
