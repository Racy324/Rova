from __future__ import annotations

import asyncio

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..environment import WorkspaceFileSystem, workspace_filesystem
from ..workspace import CodingToolError, Workspace


def create_edit_tool(workspace: Workspace | WorkspaceFileSystem) -> AgentTool:
    filesystem = workspace_filesystem(workspace)

    async def edit(path: str, old_text: str, new_text: str, replace_all: bool) -> str:
        if not old_text:
            raise CodingToolError("old_text must not be empty")
        resolved = filesystem.resolve(path)
        text = await filesystem.read_text(path)
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise CodingToolError("old_text was not found")
        if occurrences > 1 and not replace_all:
            raise CodingToolError(f"old_text occurs {occurrences} times; set replace_all to true")
        replacements = occurrences if replace_all else 1
        updated = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        await filesystem.write_text(path, updated)
        return f"{filesystem.display_path(resolved)}: replacements={replacements}"

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result = await edit(
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
