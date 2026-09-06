from __future__ import annotations

import asyncio

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..workspace import CodingToolError, Workspace


def create_read_tool(workspace: Workspace) -> AgentTool:
    def read(path: str, start_line: int | None, end_line: int | None) -> str:
        first = 1 if start_line is None else start_line
        if first < 1:
            raise CodingToolError("start_line must be at least 1")
        if end_line is not None and end_line < first:
            raise CodingToolError("end_line must be at least start_line")
        resolved = workspace.resolve(path)
        lines = workspace.read_text(resolved).splitlines()
        last = end_line if end_line is not None else len(lines)
        selected = lines[first - 1:last]
        rendered = [f"{workspace.display_path(resolved)} (lines {first}-{last})"]
        rendered.extend(f"{line_number} | {line}" for line_number, line in enumerate(selected, start=first))
        return "\n".join(rendered)

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result = await asyncio.to_thread(read, params["path"], params.get("start_line"), params.get("end_line"))
        return AgentToolResult([TextBlock(result)])

    return AgentTool(
        Tool("read", "Read a UTF-8 text file from the workspace", {"path": str, "start_line": int, "end_line": int}, required=("path",)),
        execute,
        execution_mode=ToolExecutionMode.PARALLEL,
    )
