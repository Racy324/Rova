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

    async def shell(command: str, timeout_seconds: int) -> tuple[str, dict]:
        if not command:
            raise CodingToolError("command must not be empty")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise CodingToolError("timeout_seconds must be at least 1")
        try:
            process = await _start_shell_process(command, workspace.root, child_environment)
            communicate = asyncio.create_task(process.communicate())
            try:
                done, _ = await asyncio.wait((communicate,), timeout=timeout_seconds)
                if done:
                    stdout, stderr = communicate.result()
                else:
                    await _terminate_process_tree(process)
                    stdout, stderr = await communicate
                    return (
                        _render(command, None, _as_text(stdout), _as_text(stderr), timed_out=True, blocked_values=blocked_values),
                        _metadata(command, None, timed_out=True, blocked_values=blocked_values),
                    )
            except asyncio.CancelledError:
                await _terminate_process_tree(process)
                stdout, stderr = await communicate
                raise
            return (
                _render(command, process.returncode, _as_text(stdout), _as_text(stderr), timed_out=False, blocked_values=blocked_values),
                _metadata(command, process.returncode, timed_out=False, blocked_values=blocked_values),
            )
        except OSError as error:
            raise CodingToolError(f"unable to execute command: {error}") from error

    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        result, metadata = await shell(params["command"], params.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        return AgentToolResult([TextBlock(result)], metadata)

    return AgentTool(
        Tool(
            "shell",
            "Run a local host shell command with the workspace as its working directory. "
            "Shell commands require approval and are not filesystem sandboxed.",
            {"command": str, "timeout_seconds": int},
            required=("command",),
        ),
        execute,
    )


async def _start_shell_process(command: str, cwd, environment: Mapping[str, str]) -> asyncio.subprocess.Process:
    options = {
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": environment,
        **_subprocess_group_options(),
    }
    return await asyncio.create_subprocess_shell(command, **options)


def _subprocess_group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        try:
            terminator = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(process.pid), "/T", "/F",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.wait_for(terminator.wait(), timeout=5)
        except (OSError, TimeoutError):
            process.kill()
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


def _shell_environment(environment: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    blocked_values = {
        value for name, value in environment.items() if _is_blocked_environment_variable(name) and value
    }
    child_environment = {
        name: value for name, value in environment.items() if not _is_blocked_environment_variable(name)
    }
    if os.name == "nt" and not any(name.upper() == "PATH" for name in child_environment):
        inherited_path = os.environ.get("PATH")
        if inherited_path:
            child_environment["PATH"] = inherited_path
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
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    return text.replace("\r\n", "\n")
