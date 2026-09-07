from __future__ import annotations

import asyncio

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..environment import WorkspaceFileSystem, workspace_filesystem
from ..workspace import CodingToolError, Workspace


def create_list_dir_tool(workspace: Workspace | WorkspaceFileSystem) -> AgentTool:
    filesystem = workspace_filesystem(workspace)

    def list_dir(path: str) -> str:
        resolved = filesystem.resolve(path)
        if not resolved.exists():
            raise CodingToolError("directory not found")
        if not resolved.is_dir():
            raise CodingToolError("not a directory")
        try:
            entries = sorted(resolved.iterdir(), key=lambda entry: entry.name)
        except OSError as error:
            raise CodingToolError(f"unable to list directory: {error}") from error
        rendered = []
        for entry in entries:
            kind = "symlink" if entry.is_symlink() else "directory" if entry.is_dir() else "file"
            rendered.append(f"{entry.name} [{kind}]")
        return "\n".join(rendered) if rendered else "no entries"

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result = await asyncio.to_thread(list_dir, params.get("path", "."))
        return AgentToolResult([TextBlock(result)])

    return AgentTool(
        Tool("list_dir", "List one workspace directory level", {"path": str}, required=()),
        execute,
        execution_mode=ToolExecutionMode.PARALLEL,
    )
