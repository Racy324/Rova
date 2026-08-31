from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Protocol
from uuid import uuid4

from .workspace import CodingToolError, Workspace


@dataclass(frozen=True)
class TerminalExecutionResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


@dataclass(frozen=True)
class TerminalEnvironment:
    kind: str
    executor: str
    cwd: str
    is_filesystem_sandboxed: bool
    tool_description: str = "Run a command through the configured terminal backend."
    approval_summary: str = "cwd:\n<unknown>"


class TerminalBackend(Protocol):
    @property
    def environment(self) -> TerminalEnvironment: ...

    async def execute(self, command: str, *, timeout_seconds: int) -> TerminalExecutionResult: ...

    async def close(self) -> None: ...

    def render_skill_directory(self, directory: Path) -> str: ...


class LocalTerminalBackend:
    """Run one approved shell command on the host in the selected Workspace."""

    def __init__(self, workspace: Workspace, *, environment: Mapping[str, str] | None = None) -> None:
        self._workspace = workspace
        self._child_environment, self._blocked_values = _shell_environment(
            os.environ if environment is None else environment
        )

    @property
    def environment(self) -> TerminalEnvironment:
        return TerminalEnvironment(
            kind="local",
            executor=_local_shell_executor(),
            cwd=str(self._workspace.root),
            is_filesystem_sandboxed=False,
            tool_description=(
                "Run a local host shell command with the workspace as its working directory. "
                "Shell commands require approval and are not filesystem sandboxed."
            ),
            approval_summary=f"cwd:\n{self._workspace.root}",
        )

    async def execute(self, command: str, *, timeout_seconds: int) -> TerminalExecutionResult:
        try:
            process = await _start_local_process(command, self._workspace.root, self._child_environment)
            communicate = asyncio.create_task(process.communicate())
            try:
                done, _ = await asyncio.wait((communicate,), timeout=timeout_seconds)
                if done:
                    stdout, stderr = communicate.result()
                else:
                    await _terminate_local_process_tree(process)
                    stdout, stderr = await communicate
                    return self._result(command, stdout, stderr, None, timed_out=True)
            except asyncio.CancelledError:
                await _terminate_local_process_tree(process)
                await communicate
                raise
            return self._result(command, stdout, stderr, process.returncode, timed_out=False)
        except OSError as error:
            raise CodingToolError(f"unable to execute command: {error}") from error

    async def close(self) -> None:
        """Local commands own no backend-scoped resource."""
        return None

    def render_skill_directory(self, directory: Path) -> str:
        return str(directory.resolve())

    def _result(
        self,
        command: str,
        stdout: str | bytes | None,
        stderr: str | bytes | None,
        exit_code: int | None,
        *,
        timed_out: bool,
    ) -> TerminalExecutionResult:
        return TerminalExecutionResult(
            command=_redact(command, self._blocked_values),
            stdout=_redact(_as_text(stdout), self._blocked_values),
            stderr=_redact(_as_text(stderr), self._blocked_values),
            exit_code=exit_code,
            timed_out=timed_out,
        )


class DockerTerminalBackend:
    """Run approved commands in one lazy Docker container bound to Workspace."""

    _CONTAINER_WORKSPACE = "/workspace"
    _CONTAINER_SKILLS = "/opt/rova/skills"

    def __init__(
        self,
        workspace: Workspace,
        *,
        image: str,
        skill_root: Path | None = None,
        docker_executable: str = "docker",
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not image.strip():
            raise ValueError("docker image is required")
        self._workspace = workspace
        self._image = image
        self._skill_root = Path(skill_root).resolve() if skill_root is not None and Path(skill_root).is_dir() else None
        self._docker_executable = docker_executable
        self._child_environment, self._blocked_values = _shell_environment(
            os.environ if environment is None else environment
        )
        self._container_name: str | None = None

    @property
    def environment(self) -> TerminalEnvironment:
        return TerminalEnvironment(
            kind="docker",
            executor="docker (/bin/sh)",
            cwd=self._CONTAINER_WORKSPACE,
            is_filesystem_sandboxed=True,
            tool_description=(
                "Run a Docker container shell command with the workspace bind-mounted at /workspace. "
                "Shell commands require approval."
            ),
            approval_summary=(
                f"backend:\ndocker\nimage:\n{self._image}\n"
                f"cwd:\n{self._CONTAINER_WORKSPACE}\n"
                f"workspace bind mount:\n{self._workspace.root} -> {self._CONTAINER_WORKSPACE}"
            ),
        )

    async def execute(self, command: str, *, timeout_seconds: int) -> TerminalExecutionResult:
        try:
            container_name = await self._ensure_container()
            process = await self._start_exec_process(container_name, command)
            communicate = asyncio.create_task(process.communicate())
            try:
                done, _ = await asyncio.wait((communicate,), timeout=timeout_seconds)
                if done:
                    stdout, stderr = communicate.result()
                else:
                    await self._remove_container(container_name)
                    stdout, stderr = await communicate
                    return self._result(command, stdout, stderr, None, timed_out=True)
            except asyncio.CancelledError:
                await self._remove_container(container_name)
                await communicate
                raise
            return self._result(command, stdout, stderr, process.returncode, timed_out=False)
        except OSError as error:
            raise CodingToolError(f"unable to execute Docker command: {error}") from error

    async def close(self) -> None:
        if self._container_name is not None:
            await self._remove_container(self._container_name)

    def render_skill_directory(self, directory: Path) -> str:
        if self._skill_root is None:
            raise CodingToolError("Docker Skill mount is unavailable")
        try:
            relative = Path(directory).resolve().relative_to(self._skill_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise CodingToolError("Skill path is outside the Docker Skill mount") from error
        return f"{self._CONTAINER_SKILLS}/{relative.as_posix()}"

    async def _ensure_container(self) -> str:
        if self._container_name is not None:
            return self._container_name
        container_name = f"rova-{uuid4().hex}"
        process = await asyncio.create_subprocess_exec(
            *self._create_argv(container_name),
            stdin=subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_environment,
            **_subprocess_group_options(),
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = _as_text(stderr).strip() or _as_text(stdout).strip() or "unknown Docker error"
            raise CodingToolError(f"unable to create Docker container: {detail}")
        self._container_name = container_name
        return container_name

    async def _start_exec_process(self, container_name: str, command: str) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *self._exec_argv(container_name, command),
            stdin=subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_environment,
            **_subprocess_group_options(),
        )

    def _create_argv(self, container_name: str) -> tuple[str, ...]:
        argv = [
            self._docker_executable,
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--workdir",
            self._CONTAINER_WORKSPACE,
            "--mount",
            f"type=bind,source={self._workspace.root},target={self._CONTAINER_WORKSPACE}",
        ]
        if self._skill_root is not None:
            argv.extend([
                "--mount",
                f"type=bind,source={self._skill_root},target={self._CONTAINER_SKILLS},readonly",
            ])
        argv.extend([self._image, "/bin/sh", "-lc", "while :; do sleep 3600; done"])
        return tuple(argv)

    def _exec_argv(self, container_name: str, command: str) -> tuple[str, ...]:
        return (
            self._docker_executable,
            "exec",
            "--workdir",
            self._CONTAINER_WORKSPACE,
            container_name,
            "/bin/sh",
            "-lc",
            command,
        )

    async def _remove_container(self, container_name: str) -> None:
        if self._container_name == container_name:
            self._container_name = None
        try:
            cleanup = await asyncio.create_subprocess_exec(
                self._docker_executable,
                "rm",
                "--force",
                container_name,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._child_environment,
            )
            await asyncio.wait_for(cleanup.wait(), timeout=5)
        except (OSError, TimeoutError):
            return

    def _result(
        self,
        command: str,
        stdout: str | bytes | None,
        stderr: str | bytes | None,
        exit_code: int | None,
        *,
        timed_out: bool,
    ) -> TerminalExecutionResult:
        return TerminalExecutionResult(
            command=_redact(command, self._blocked_values),
            stdout=_redact(_as_text(stdout), self._blocked_values),
            stderr=_redact(_as_text(stderr), self._blocked_values),
            exit_code=exit_code,
            timed_out=timed_out,
        )


async def _start_local_process(
    command: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        **_subprocess_group_options(),
    )


def _subprocess_group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _terminate_local_process_tree(process: asyncio.subprocess.Process) -> None:
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


def _local_shell_executor() -> str:
    if os.name == "nt":
        return Path(os.environ.get("COMSPEC", "cmd.exe")).name
    return "/bin/sh"


def _redact(text: str, blocked_values: tuple[str, ...]) -> str:
    for value in blocked_values:
        text = text.replace(value, "[REDACTED]")
    return text


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    return text.replace("\r\n", "\n")
