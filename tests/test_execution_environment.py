from __future__ import annotations

from pathlib import Path

import pytest

from rova.app.workspace.workspace import WorkspaceError


@pytest.mark.asyncio
async def test_local_environment_uses_one_workspace_for_files_and_shell(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    (workspace_root / "src").mkdir(parents=True)

    from rova.app.workspace.environment import LocalExecutionEnvironment
    from rova.app.workspace.workspace import Workspace

    environment = LocalExecutionEnvironment(Workspace(workspace_root))

    await environment.filesystem.write_text("src/example.txt", "local environment")

    assert await environment.filesystem.read_text("src/example.txt") == "local environment"
    assert environment.terminal.environment.cwd == str(workspace_root.resolve())
    assert environment.descriptor.kind == "local"
    assert environment.descriptor.host_workspace_isolated is False


@pytest.mark.asyncio
async def test_local_environment_preserves_workspace_escape_rejection(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    from rova.app.workspace.environment import LocalExecutionEnvironment
    from rova.app.workspace.workspace import Workspace

    environment = LocalExecutionEnvironment(Workspace(workspace_root))

    with pytest.raises(WorkspaceError, match="path escapes workspace root"):
        await environment.filesystem.read_text("../outside.txt")
