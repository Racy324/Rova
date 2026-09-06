from __future__ import annotations

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

from ..terminal import TerminalBackend, TerminalExecutionResult
from ..workspace import CodingToolError


DEFAULT_TIMEOUT_SECONDS = 30


def create_shell_tool(backend: TerminalBackend) -> AgentTool:
    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        command = params["command"]
        timeout_seconds = params.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        if not command:
            raise CodingToolError("command must not be empty")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise CodingToolError("timeout_seconds must be at least 1")
        result = await backend.execute(command, timeout_seconds=timeout_seconds)
        return AgentToolResult([TextBlock(_render(result))], _metadata(result))

    return AgentTool(
        Tool(
            "shell",
            backend.environment.tool_description,
            {"command": str, "timeout_seconds": int},
            required=("command",),
        ),
        execute,
        execution_mode=ToolExecutionMode.SEQUENTIAL,
    )


def _metadata(result: TerminalExecutionResult) -> dict:
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "outcome": "command_timeout" if result.timed_out else "success" if result.exit_code == 0 else "command_nonzero_exit",
    }


def _render(result: TerminalExecutionResult) -> str:
    rendered_exit_code = "null" if result.exit_code is None else str(result.exit_code)
    return (
        f"command: {result.command}\n"
        f"exit_code: {rendered_exit_code}\n"
        f"timed_out: {str(result.timed_out).lower()}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
