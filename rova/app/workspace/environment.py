from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from pathlib import Path

from .terminal import DockerTerminalBackend, LocalTerminalBackend, TerminalBackend
from .workspace import Workspace


@dataclass(frozen=True)
class ExecutionEnvironmentDescriptor:
    """Stable facts describing the logical workspace exposed to Coding Tools."""

    kind: str
    logical_workspace: str
    host_workspace: str
    host_workspace_isolated: bool
    resume_note: str | None = None


class WorkspaceFileSystem(Protocol):
    """Controlled file operations for one logical Workspace."""

    @property
    def display_root(self) -> str: ...

    def resolve(self, path: str) -> Path: ...

    def display_path(self, path: Path) -> str: ...

    async def read_text(self, path: str) -> str: ...

    async def write_text(self, path: str, content: str) -> None: ...

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes: ...


class ExecutionEnvironment(Protocol):
    """One matching file and terminal view used by Coding Tools."""

    @property
    def filesystem(self) -> WorkspaceFileSystem: ...

    @property
    def terminal(self) -> TerminalBackend: ...

    @property
    def descriptor(self) -> ExecutionEnvironmentDescriptor: ...

    async def close(self) -> None: ...


class LocalWorkspaceFileSystem:
    """Async facade over the existing Host filesystem Workspace boundary."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def display_root(self) -> str:
        return str(self._workspace.root)

    def resolve(self, path: str) -> Path:
        return self._workspace.resolve(path)

    def display_path(self, path: Path) -> str:
        return self._workspace.display_path(path)

    async def read_text(self, path: str) -> str:
        return await asyncio.to_thread(self._read_text, path)

    async def write_text(self, path: str, content: str) -> None:
        await asyncio.to_thread(self._write_text, path, content)

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes:
        return await asyncio.to_thread(self._read_bytes, path, max_bytes)

    def _read_text(self, path: str) -> str:
        return self._workspace.read_text(self._workspace.resolve(path))

    def _write_text(self, path: str, content: str) -> None:
        self._workspace.write_text(self._workspace.resolve(path), content)

    def _read_bytes(self, path: str, max_bytes: int) -> bytes:
        return self._workspace.resolve(path).read_bytes()[: max_bytes + 1]


def workspace_filesystem(value: Workspace | WorkspaceFileSystem) -> WorkspaceFileSystem:
    """Accept the legacy Workspace boundary only at direct Tool factory call sites."""
    if isinstance(value, Workspace):
        return LocalWorkspaceFileSystem(value)
    return value


class LocalExecutionEnvironment:
    """Trusted environment where the logical Workspace is the Host Workspace."""

    def __init__(self, workspace: Workspace, *, terminal: TerminalBackend | None = None) -> None:
        self._workspace = workspace
        self._filesystem = LocalWorkspaceFileSystem(workspace)
        self._terminal = terminal or LocalTerminalBackend(workspace)
        self._descriptor = ExecutionEnvironmentDescriptor(
            kind="local",
            logical_workspace=str(workspace.root),
            host_workspace=str(workspace.root),
            host_workspace_isolated=False,
        )

    @property
    def filesystem(self) -> WorkspaceFileSystem:
        return self._filesystem

    @property
    def terminal(self) -> TerminalBackend:
        return self._terminal

    @property
    def descriptor(self) -> ExecutionEnvironmentDescriptor:
        return self._descriptor

    async def close(self) -> None:
        await self._terminal.close()


class DockerSandboxEnvironment:
    """Docker terminal plus the Rova-owned Workspace that it mounts."""

    def __init__(
        self,
        *,
        host_workspace: Workspace,
        sandbox_workspace: Workspace,
        image: str,
        skill_root: Path | None = None,
        docker_executable: str = "docker",
        resumed: bool = False,
    ) -> None:
        self._filesystem = LocalWorkspaceFileSystem(sandbox_workspace)
        self._terminal = DockerTerminalBackend(
            sandbox_workspace,
            image=image,
            skill_root=skill_root,
            docker_executable=docker_executable,
        )
        self._descriptor = ExecutionEnvironmentDescriptor(
            kind="docker_sandbox",
            logical_workspace=str(sandbox_workspace.root),
            host_workspace=str(host_workspace.root),
            host_workspace_isolated=True,
            resume_note=(
                "Sandbox workspace was resumed; a fresh execution container was created. "
                "Workspace files persisted; container-local packages, processes and temporary state may have been lost."
                if resumed
                else "Sandbox workspace files persist independently from this Runtime; container-local packages, "
                "processes and temporary state persist only while the current execution container exists."
            ),
        )

    @property
    def filesystem(self) -> WorkspaceFileSystem:
        return self._filesystem

    @property
    def terminal(self) -> TerminalBackend:
        return self._terminal

    @property
    def descriptor(self) -> ExecutionEnvironmentDescriptor:
        return self._descriptor

    async def close(self) -> None:
        await self._terminal.close()
