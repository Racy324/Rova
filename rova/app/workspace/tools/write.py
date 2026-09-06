from __future__ import annotations

import asyncio

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..workspace import Workspace


def create_write_tool(workspace: Workspace) -> AgentTool:
    def write(path: str, content: str) -> str:
        resolved = workspace.resolve(path)
        workspace.write_text(resolved, content)
        return f"{workspace.display_path(resolved)}: written"

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result = await asyncio.to_thread(write, params["path"], params["content"])
        return AgentToolResult([TextBlock(result)])

    return AgentTool(
        Tool("write", "Create or replace a UTF-8 workspace file", {"path": str, "content": str}),
        execute,
        execution_mode=ToolExecutionMode.SEQUENTIAL,
    )
