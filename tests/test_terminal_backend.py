from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from rova.app.workspace import Workspace
from rova.app.workspace.approval import AlwaysApprove, AlwaysDeny
from rova.app.workspace.controlled_tool import build_controlled_coding_tools
from rova.app.workspace.policy import DefaultCodingToolPolicy
from rova.app.workspace import terminal
from rova.app.workspace.terminal import DockerTerminalBackend, LocalTerminalBackend, TerminalEnvironment, TerminalExecutionResult
from rova.app.workspace.tools import create_shell_tool
from rova.agent_core.tools import ToolRegistry, ToolRuntime
from rova.ai.messages import ToolCall


@pytest.mark.asyncio
async def test_local_terminal_backend_executes_in_workspace_and_describes_host_execution(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    backend = LocalTerminalBackend(Workspace(workspace_root))

    result = await backend.execute(
        f'"{sys.executable}" -c "import os; print(os.getcwd())"',
        timeout_seconds=5,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.replace("\\", "/").strip() == workspace_root.as_posix()
    assert result.stderr == ""
    assert backend.environment.kind == "local"
    assert backend.environment.cwd == str(workspace_root.resolve())
    assert backend.environment.is_filesystem_sandboxed is False


@pytest.mark.asyncio
async def test_shell_tool_delegates_execution_and_renders_backend_result() -> None:
    class RecordingBackend:
        environment = TerminalEnvironment(
            kind="recording",
            executor="recording-shell",
            cwd="/recording",
            is_filesystem_sandboxed=False,
        )

        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def execute(self, command: str, *, timeout_seconds: int) -> TerminalExecutionResult:
            self.calls.append((command, timeout_seconds))
            return TerminalExecutionResult(command, "observed stdout", "observed stderr", 3, False)

        def render_skill_directory(self, directory: Path) -> str:
            return str(directory)

    backend = RecordingBackend()
    tool = create_shell_tool(backend)

    result = await tool.execute("shell-call", {"command": "run task", "timeout_seconds": 12})

    assert backend.calls == [("run task", 12)]
    assert result.content[0].text == (
        "command: run task\n"
        "exit_code: 3\n"
        "timed_out: false\n"
        "stdout:\nobserved stdout\n"
        "stderr:\nobserved stderr"
    )
    assert result.metadata == {
        "command": "run task",
        "exit_code": 3,
        "timed_out": False,
        "outcome": "command_nonzero_exit",
    }


def test_controlled_tool_builder_uses_the_injected_terminal_backend(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    backend = LocalTerminalBackend(Workspace(workspace_root))

    tools = build_controlled_coding_tools(
        Workspace(workspace_root),
        DefaultCodingToolPolicy(),
        AlwaysApprove(),
        terminal_backend=backend,
    )

    shell = next(tool for tool in tools if tool.tool.name == "shell")
    assert shell.inner.tool.description == (
        "Run an approved shell command in the current logical workspace. "
        "Runtime facts describe the selected execution environment."
    )


@pytest.mark.asyncio
async def test_shell_approval_happens_before_docker_backend_execution(tmp_path: Path) -> None:
    class RecordingBackend:
        environment = TerminalEnvironment("docker", "docker (/bin/sh)", "/workspace", True)

        def __init__(self) -> None:
            self.execute_calls = 0

        async def execute(self, command: str, *, timeout_seconds: int) -> TerminalExecutionResult:
            self.execute_calls += 1
            return TerminalExecutionResult(command, "", "", 0, False)

        async def close(self) -> None:
            return None

        def render_skill_directory(self, directory: Path) -> str:
            return str(directory)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    backend = RecordingBackend()
    tools = build_controlled_coding_tools(
        Workspace(workspace_root),
        DefaultCodingToolPolicy(),
        AlwaysDeny(),
        terminal_backend=backend,
    )

    result = await ToolRuntime(ToolRegistry(tools)).execute(ToolCall("denied-shell", "shell", {"command": "true"}))

    assert result.is_error is True
    assert backend.execute_calls == 0


class _CompletedProcess:
    returncode = 0

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    async def wait(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_docker_terminal_backend_is_lazy_then_reuses_one_container_for_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def start_process(*args: object, **kwargs: object) -> _CompletedProcess:
        calls.append((args, kwargs))
        if args[1:3] == ("run", "-d"):
            return _CompletedProcess(b"container-id\n")
        return _CompletedProcess(b"container stdout\n")

    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", start_process)
    workspace_root = tmp_path / "workspace"
    skill_root = tmp_path / "skills"
    (skill_root / "paper-card").mkdir(parents=True)
    workspace_root.mkdir()
    backend = DockerTerminalBackend(Workspace(workspace_root), image="rova-test:latest", skill_root=skill_root)

    assert calls == []
    first = await backend.execute("printf 'first'", timeout_seconds=5)
    second = await backend.execute("printf 'second'", timeout_seconds=5)

    assert first.stdout == "container stdout\n"
    assert second.exit_code == 0
    assert [argv[1] for argv, _ in calls] == ["run", "exec", "exec"]
    create_argv, create_options = calls[0]
    assert create_argv[0:3] == ("docker", "run", "-d")
    assert "--rm" in create_argv
    container_name = create_argv[create_argv.index("--name") + 1]
    assert "--network" in create_argv and create_argv[create_argv.index("--network") + 1] == "none"
    mounts = [create_argv[index + 1] for index, item in enumerate(create_argv[:-1]) if item == "--mount"]
    assert f"type=bind,source={workspace_root.resolve()},target=/workspace" in mounts
    assert f"type=bind,source={skill_root.resolve()},target=/opt/rova/skills,readonly" in mounts
    assert create_options["env"] is not None
    for exec_argv, _ in (calls[1], calls[2]):
        assert exec_argv[0:5] == ("docker", "exec", "--workdir", "/workspace", container_name)
        assert exec_argv[-3:] == ("/bin/sh", "-lc", exec_argv[-1])
        assert "--mount" not in exec_argv
    assert backend.environment.kind == "docker"
    assert backend.environment.cwd == "/workspace"
    assert backend.render_skill_directory(skill_root / "paper-card") == "/opt/rova/skills/paper-card"


@pytest.mark.asyncio
async def test_docker_terminal_backend_close_cleans_up_once_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, ...]] = []

    async def start_process(*args: object, **kwargs: object) -> _CompletedProcess:
        calls.append(args)
        return _CompletedProcess(b"container-id\n" if args[1:3] == ("run", "-d") else b"")

    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", start_process)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    backend = DockerTerminalBackend(Workspace(workspace_root), image="rova-test:latest")

    await backend.execute("true", timeout_seconds=5)
    container_name = calls[0][calls[0].index("--name") + 1]
    await backend.close()
    await backend.close()

    assert calls[-1] == ("docker", "rm", "--force", container_name)
    assert sum(call[1:3] == ("rm", "--force") for call in calls) == 1


@pytest.mark.asyncio
async def test_docker_terminal_backend_timeout_resets_container_before_next_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DelayedProcess:
        returncode = None

        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            await self.release.wait()
            return b"", b""

    delayed = DelayedProcess()
    calls: list[tuple[object, ...]] = []

    async def start_process(*args: object, **kwargs: object) -> object:
        calls.append(args)
        if args[1:3] == ("run", "-d"):
            return _CompletedProcess(b"container-id\n")
        if args[1:3] == ("rm", "--force"):
            delayed.release.set()
            return _CompletedProcess()
        if len([call for call in calls if call[1] == "exec"]) == 1:
            return delayed
        return _CompletedProcess(b"next\n")

    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", start_process)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    backend = DockerTerminalBackend(Workspace(workspace_root), image="rova-test:latest")

    timed_out = await backend.execute("sleep 60", timeout_seconds=0)
    next_result = await backend.execute("printf next", timeout_seconds=5)

    assert timed_out.timed_out is True
    assert next_result.stdout == "next\n"
    assert [call[1] for call in calls] == ["run", "exec", "rm", "run", "exec"]


@pytest.mark.asyncio
async def test_docker_terminal_backend_cancellation_resets_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingProcess:
        returncode = None

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            self.started.set()
            await self.release.wait()
            return b"", b""

    blocking = BlockingProcess()
    calls: list[tuple[object, ...]] = []

    async def start_process(*args: object, **kwargs: object) -> object:
        calls.append(args)
        if args[1:3] == ("run", "-d"):
            return _CompletedProcess(b"container-id\n")
        if args[1:3] == ("rm", "--force"):
            blocking.release.set()
            return _CompletedProcess()
        return blocking

    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", start_process)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    backend = DockerTerminalBackend(Workspace(workspace_root), image="rova-test:latest")
    task = asyncio.create_task(backend.execute("sleep 60", timeout_seconds=60))
    await blocking.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [call[1] for call in calls] == ["run", "exec", "rm"]
