from pathlib import Path
import shutil

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall
from rova.ai.models import Model
from rova.app import runtime as runtime_module
from rova.app.runtime import build_rova_runtime
from rova.app.workspace.environment import (
    ExecutionEnvironmentDescriptor,
    LocalWorkspaceFileSystem,
)
from rova.app.workspace.terminal import LocalTerminalBackend
from rova.app.workspace.workspace import Workspace


class _VisionClient:
    def __init__(self) -> None:
        self.image_data_url: str | None = None

    async def analyze(self, *, question: str, image_data_url: str) -> str:
        self.image_data_url = image_data_url
        return "sandbox image"


class _LocalDockerSandboxEnvironment:
    """Test double: Docker logical paths with a local executor over the Sandbox tree."""

    def __init__(self, *, host_workspace: Workspace, sandbox_workspace: Workspace, image: str, **_kwargs) -> None:
        self.filesystem = LocalWorkspaceFileSystem(sandbox_workspace)
        self.terminal = LocalTerminalBackend(sandbox_workspace)
        self.descriptor = ExecutionEnvironmentDescriptor(
            kind="docker_sandbox",
            logical_workspace="/workspace",
            host_workspace=str(host_workspace.root),
            host_workspace_isolated=True,
        )

    async def close(self) -> None:
        await self.terminal.close()


async def _stream_done(_model, _context, _options):
    yield StreamDone(AssistantMessage([TextBlock("done")]))


@pytest.mark.asyncio
async def test_sandbox_runtime_routes_file_and_shell_tools_to_the_same_sandbox_workspace(monkeypatch, tmp_path: Path) -> None:
    host_root = tmp_path / "host-project"
    (host_root / "src").mkdir(parents=True)
    host_file = host_root / "src" / "a.txt"
    host_file.write_text("host baseline", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "DockerSandboxEnvironment", _LocalDockerSandboxEnvironment, raising=False)

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=_stream_done,
        workspace_root=host_root,
        terminal_backend="docker",
        docker_image="rova-test:latest",
        isolated_sandbox=True,
        permission_mode="full",
        sandbox_root=tmp_path / "rova-data" / "sandboxes",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert runtime.execution_environment is not None
    assert runtime.execution_environment.descriptor.kind == "docker_sandbox"
    assert runtime.execution_environment.descriptor.logical_workspace == "/workspace"
    assert runtime.execution_environment.descriptor.host_workspace_isolated is True

    await runtime.agent.tool_runtime.execute(ToolCall("write", "write", {"path": "src/a.txt", "content": "file tool change"}))
    shell_read = await runtime.agent.tool_runtime.execute(ToolCall("shell-read", "shell", {"command": "type src\\a.txt"}))
    assert "file tool change" in shell_read.text

    await runtime.agent.tool_runtime.execute(ToolCall("shell-write", "shell", {"command": "echo shell change> src\\a.txt"}))
    file_read = await runtime.agent.tool_runtime.execute(ToolCall("read", "read", {"path": "src/a.txt"}))
    assert "shell change" in file_read.text

    host_file.write_text("host-only change", encoding="utf-8")
    search = await runtime.agent.tool_runtime.execute(ToolCall("search", "search", {"query": "host-only change"}))
    assert search.text == "no results"
    assert host_file.read_text(encoding="utf-8") == "host-only change"

    git_status = await runtime.agent.tool_runtime.execute(ToolCall("status", "shell", {"command": "git status --short"}))
    assert "src/a.txt" in git_status.text.replace("\\", "/")

    await runtime.close()


@pytest.mark.asyncio
async def test_local_runtime_keeps_file_search_and_shell_on_the_host_workspace(tmp_path: Path) -> None:
    host_root = tmp_path / "host-project"
    (host_root / "src").mkdir(parents=True)
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=_stream_done,
        workspace_root=host_root,
        permission_mode="full",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.agent.tool_runtime.execute(ToolCall("write", "write", {"path": "src/a.txt", "content": "from file"}))
    await runtime.agent.tool_runtime.execute(ToolCall("shell", "shell", {"command": "echo from shell> src\\a.txt"}))
    read = await runtime.agent.tool_runtime.execute(ToolCall("read", "read", {"path": "src/a.txt"}))
    search = await runtime.agent.tool_runtime.execute(ToolCall("search", "search", {"query": "from shell"}))

    assert "from shell" in read.text
    assert "src/a.txt:1: from shell" in search.text
    assert (host_root / "src" / "a.txt").read_text(encoding="utf-8").strip() == "from shell"
    await runtime.close()


@pytest.mark.asyncio
async def test_sandbox_runtime_vision_reads_the_sandbox_copy_not_the_host_copy(monkeypatch, tmp_path: Path) -> None:
    host_root = tmp_path / "host-project"
    host_root.mkdir()
    (host_root / "diagram.png").write_bytes(b"host-image")
    client = _VisionClient()
    monkeypatch.setattr(runtime_module, "DockerSandboxEnvironment", _LocalDockerSandboxEnvironment, raising=False)
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=_stream_done,
        workspace_root=host_root,
        vision_client=client,
        terminal_backend="docker",
        docker_image="rova-test:latest",
        isolated_sandbox=True,
        permission_mode="full",
        sandbox_root=tmp_path / "rova-data" / "sandboxes",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )
    assert runtime.execution_environment is not None
    runtime.execution_environment.filesystem.resolve("diagram.png").write_bytes(b"sandbox-image")

    result = await runtime.agent.tool_runtime.execute(
        ToolCall("vision", "vision_analyze", {"image_path": "diagram.png", "question": "inspect"})
    )

    assert result.text == "sandbox image"
    assert client.image_data_url is not None and "c2FuZGJveC1pbWFnZQ==" in client.image_data_url
    assert (host_root / "diagram.png").read_bytes() == b"host-image"
    await runtime.close()


def test_sandbox_runtime_uses_only_the_sandbox_as_the_docker_mount_source(tmp_path: Path) -> None:
    host_root = tmp_path / "host-project"
    host_root.mkdir()
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=_stream_done,
        workspace_root=host_root,
        terminal_backend="docker",
        docker_image="rova-test:latest",
        isolated_sandbox=True,
        sandbox_root=tmp_path / "rova-data" / "sandboxes",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )
    assert runtime.execution_environment is not None
    argv = runtime.terminal_backend._create_argv("rova-test")  # type: ignore[union-attr]
    sandbox_root = runtime.execution_environment.filesystem.resolve(".")

    assert f"type=bind,source={sandbox_root},target=/workspace" in argv
    assert f"type=bind,source={host_root.resolve()},target=/workspace" not in argv


@pytest.mark.asyncio
async def test_sandbox_runtime_keeps_coding_tool_schemas_and_runtime_facts_logical(tmp_path: Path) -> None:
    host_root = tmp_path / "host-project"
    host_root.mkdir()
    captured_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        captured_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    local = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=host_root,
        session_root=tmp_path / "local-sessions", artifact_root=tmp_path / "local-artifacts",
    )
    sandbox = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=host_root,
        terminal_backend="docker", docker_image="rova-test:latest", isolated_sandbox=True,
        sandbox_root=tmp_path / "rova-data" / "sandboxes",
        session_root=tmp_path / "sandbox-sessions", artifact_root=tmp_path / "sandbox-artifacts",
    )

    assert local.agent.registry.schemas == sandbox.agent.registry.schemas
    await sandbox.session.prompt("show the workspace")
    assert captured_system_prompts
    prompt = captured_system_prompts[-1]
    assert "/workspace" in prompt
    assert str(sandbox.execution_environment.filesystem.resolve(".")) not in prompt  # type: ignore[union-attr]
    assert str(host_root.resolve()) not in prompt
    await local.close()
    await sandbox.close()


def test_corrupt_resumed_sandbox_fails_closed_without_a_host_workspace_fallback(monkeypatch, tmp_path: Path) -> None:
    host_root = tmp_path / "host-project"
    host_root.mkdir()
    monkeypatch.setattr(runtime_module, "DockerSandboxEnvironment", _LocalDockerSandboxEnvironment, raising=False)
    initial = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=_stream_done,
        workspace_root=host_root,
        terminal_backend="docker",
        docker_image="rova-test:latest",
        isolated_sandbox=True,
        sandbox_root=tmp_path / "rova-data" / "sandboxes",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )
    assert initial.execution_environment is not None
    shutil.rmtree(initial.execution_environment.filesystem.resolve("."), onerror=_clear_readonly)

    with pytest.raises(Exception, match="Sandbox is unavailable"):
        build_rova_runtime(
            model=Model(provider="mock"),
            stream_fn=_stream_done,
            workspace_root=host_root,
            terminal_backend="docker",
            docker_image="rova-test:latest",
            isolated_sandbox=True,
            sandbox_root=tmp_path / "rova-data" / "sandboxes",
            session_id=initial.session.session_id,
            session_root=tmp_path / "sessions",
            artifact_root=tmp_path / "artifacts",
        )


def _clear_readonly(function, path, _exc_info) -> None:
    Path(path).chmod(0o700)
    function(path)
