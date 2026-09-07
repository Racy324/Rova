from __future__ import annotations

import subprocess
import os
from pathlib import Path

import pytest

from rova.app.workspace.workspace import Workspace


def test_baseline_import_creates_one_private_b0_from_actual_host_tree(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    (host_root / "tracked.txt").write_text("dirty host content", encoding="utf-8")
    (host_root / "ignored.generated").write_bytes(b"\x00binary\xff")
    (host_root / "empty-dir").mkdir()
    (host_root / ".git").mkdir()
    (host_root / ".git" / "config").write_text("host-secret-config", encoding="utf-8")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")

    imported = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)

    assert imported.baseline_commit_oid is not None
    assert (imported.sandbox_root / "tracked.txt").read_text(encoding="utf-8") == "dirty host content"
    assert (imported.sandbox_root / "ignored.generated").read_bytes() == b"\x00binary\xff"
    assert (imported.sandbox_root / "empty-dir").is_dir()
    assert (imported.sandbox_root / ".git").is_dir()
    assert "host-secret-config" not in (imported.sandbox_root / ".git" / "config").read_text(encoding="utf-8")
    assert _git(imported.sandbox_root, "rev-list", "--count", "HEAD") == "1"
    assert _git(imported.sandbox_root, "rev-parse", "HEAD") == imported.baseline_commit_oid
    assert _git(imported.sandbox_root, "show", f"{imported.baseline_commit_oid}:tracked.txt") == "dirty host content"
    assert _git(imported.sandbox_root, "remote") == ""


def test_docker_sandbox_environment_mounts_only_rova_owned_sandbox_root(tmp_path: Path) -> None:
    from rova.app.workspace.environment import DockerSandboxEnvironment
    from rova.app.workspace.sandbox import SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    (host_root / "src").mkdir()
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    imported = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)

    environment = DockerSandboxEnvironment(
        host_workspace=Workspace(host_root),
        sandbox_workspace=Workspace(imported.sandbox_root),
        image="rova-test:latest",
    )
    argv = environment.terminal._create_argv("rova-test-container")

    assert f"type=bind,source={imported.sandbox_root},target=/workspace" in argv
    assert f"type=bind,source={host_root.resolve()},target=/workspace" not in argv
    assert environment.descriptor.kind == "docker_sandbox"
    assert environment.descriptor.host_workspace_isolated is True
    assert environment.descriptor.resume_note is not None


def test_baseline_import_excludes_embedded_rova_data_root(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxStore

    host_root = tmp_path / "host-project"
    data_root = host_root / ".rova"
    data_root.mkdir(parents=True)
    (host_root / "source.py").write_text("source", encoding="utf-8")
    (data_root / "local-state.json").write_text("must not enter sandbox", encoding="utf-8")
    store = SandboxStore(data_root / "sandboxes")

    imported = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)

    assert (imported.sandbox_root / "source.py").read_text(encoding="utf-8") == "source"
    assert not (imported.sandbox_root / ".rova").exists()


def test_baseline_import_rejects_a_symlink_that_escapes_the_host_workspace(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxImportError, SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not workspace state", encoding="utf-8")
    try:
        os.symlink(outside, host_root / "outside-link")
    except OSError:
        pytest.skip("the current Windows account cannot create symlinks")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")

    with pytest.raises(SandboxImportError, match="symlink escapes Host Workspace"):
        store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)


def _git(cwd: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()
