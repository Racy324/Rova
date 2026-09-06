from __future__ import annotations

import asyncio

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..workspace import CodingToolError, Workspace


def create_edit_tool(workspace: Workspace) -> AgentTool:
    def edit(path: str, old_text: str, new_text: str, replace_all: bool) -> str:
        if not old_text:
            raise CodingToolError("old_text must not be empty")
        resolved = workspace.resolve(path)
        text = workspace.read_text(resolved)
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise CodingToolError("old_text was not found")
        if occurrences > 1 and not replace_all:
            raise CodingToolError(f"old_text occurs {occurrences} times; set replace_all to true")
        replacements = occurrences if replace_all else 1
        updated = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        workspace.write_text(resolved, updated)
        return f"{workspace.display_path(resolved)}: replacements={replacements}"

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result = await asyncio.to_thread(
            edit,
            params["path"],
            params["old_text"],
            params["new_text"],
            params.get("replace_all", False),
        )
        return AgentToolResult([TextBlock(result)])

    return AgentTool(
        Tool(
            "edit",
            "Replace exact UTF-8 text in a workspace file",
            {"path": str, "old_text": str, "new_text": str, "replace_all": bool},
            required=("path", "old_text", "new_text"),
        ),
        execute,
        execution_mode=ToolExecutionMode.SEQUENTIAL,
    )
