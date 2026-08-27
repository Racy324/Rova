from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Mapping

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult

from ..workspace import CodingToolError, Workspace


DEFAULT_TIMEOUT_SECONDS = 30


def create_shell_tool(workspace: Workspace, *, environment: Mapping[str, str] | None = None) -> AgentTool:
    child_environment, blocked_values = _shell_environment(os.environ if environment is None else environment)

    def shell(command: str, timeout_seconds: int) -> tuple[str, dict]:
        if not command:
            raise CodingToolError("command must not be empty")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise CodingToolError("timeout_seconds must be at least 1")
        try:
            completed = subprocess.run(
                command,
                cwd=workspace.root,
                shell=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                env=child_environment,
            )
            return (
                _render(command, completed.returncode, completed.stdout, completed.stderr, timed_out=False, blocked_values=blocked_values),
                _metadata(command, completed.returncode, timed_out=False, blocked_values=blocked_values),
            )
        except subprocess.TimeoutExpired as error:
            return (
                _render(command, None, _as_text(error.stdout), _as_text(error.stderr), timed_out=True, blocked_values=blocked_values),
                _metadata(command, None, timed_out=True, blocked_values=blocked_values),
            )
        except OSError as error:
            raise CodingToolError(f"unable to execute command: {error}") from error

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result, metadata = await asyncio.to_thread(shell, params["command"], params.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        return AgentToolResult([TextBlock(result)], metadata)

    return AgentTool(
        Tool("shell", "Run a workspace shell command; policy protection is deferred to Phase 4.2", {"command": str, "timeout_seconds": int}, required=("command",)),
        execute,
    )


def _shell_environment(environment: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    blocked_values = {
        value for name, value in environment.items() if _is_blocked_environment_variable(name) and value
    }
    child_environment = {
        name: value for name, value in environment.items() if not _is_blocked_environment_variable(name)
    }
    return child_environment, tuple(sorted(blocked_values, key=len, reverse=True))


def _is_blocked_environment_variable(name: str) -> bool:
    normalized_name = name.upper()
    return normalized_name == "OPENAI_API_KEY" or normalized_name.startswith("ROVA_") or normalized_name.endswith("_API_KEY")


def _metadata(
    command: str, exit_code: int | None, *, timed_out: bool, blocked_values: tuple[str, ...]
) -> dict:
    return {
        "command": _redact(command, blocked_values),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "outcome": "command_timeout" if timed_out else "success" if exit_code == 0 else "command_nonzero_exit",
    }


def _render(
    command: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    *,
    timed_out: bool,
    blocked_values: tuple[str, ...],
) -> str:
    rendered_exit_code = "null" if exit_code is None else str(exit_code)
    return (
        f"command: {_redact(command, blocked_values)}\n"
        f"exit_code: {rendered_exit_code}\n"
        f"timed_out: {str(timed_out).lower()}\n"
        f"stdout:\n{_redact(stdout, blocked_values)}\n"
        f"stderr:\n{_redact(stderr, blocked_values)}"
    )


def _redact(text: str, blocked_values: tuple[str, ...]) -> str:
    for value in blocked_values:
        text = text.replace(value, "[REDACTED]")
    return text


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
