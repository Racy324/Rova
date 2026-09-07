from __future__ import annotations

import shutil
import os
import stat
from pathlib import Path

import pytest

from rova.app.paths import RovaDataPaths
from rova.app.workspace.workspace import Workspace


def test_sandbox_store_persists_one_session_workspace_binding(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxState, SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    (host_root / "dirty.txt").write_text("user change", encoding="utf-8")
    store_root = tmp_path / "rova-data" / "sandboxes"
    store = SandboxStore(store_root)

    created = store.create_unbound(Workspace(host_root))
    imported = store.import_baseline(created.sandbox_id)
    bound = store.bind_session(imported.sandbox_id, "session-1")
    ready = store.mark_ready(bound.sandbox_id)

    resumed = SandboxStore(store_root).load_for_session("session-1", created.workspace_id)

    assert RovaDataPaths(tmp_path / "rova-data").sandboxes == store_root
    assert ready.state is SandboxState.READY
    assert resumed is not None
    assert resumed.sandbox_id == created.sandbox_id
    assert resumed.state is SandboxState.READY
    assert (host_root / "dirty.txt").read_text(encoding="utf-8") == "user change"


def test_sandbox_store_rejects_a_second_active_sandbox_for_one_host_workspace(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxConflictError, SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")

    store.create_unbound(Workspace(host_root))

    with pytest.raises(SandboxConflictError, match="active Sandbox already exists"):
        store.create_unbound(Workspace(host_root))


def test_missing_ready_sandbox_is_marked_abandoned_without_host_reimport(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxState, SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    (host_root / "source.py").write_text("host baseline", encoding="utf-8")
    store_root = tmp_path / "rova-data" / "sandboxes"
    store = SandboxStore(store_root)
    created = store.create_unbound(Workspace(host_root))
    sandbox = store.bind_session(store.import_baseline(created.sandbox_id).sandbox_id, "session-1")
    ready = store.mark_ready(sandbox.sandbox_id)
    _remove_tree(ready.sandbox_root)

    resumed = SandboxStore(store_root).load_for_session("session-1", ready.workspace_id)

    assert resumed is not None
    assert resumed.state is SandboxState.ABANDONED
    assert (host_root / "source.py").read_text(encoding="utf-8") == "host baseline"


def test_sandbox_cannot_become_ready_without_an_immutable_baseline(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxError, SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    bound = store.bind_session(store.create_unbound(Workspace(host_root)).sandbox_id, "session-1")

    with pytest.raises(SandboxError, match="immutable baseline"):
        store.mark_ready(bound.sandbox_id)


def _remove_tree(path: Path) -> None:
    def clear_readonly(function, failed_path, _exception) -> None:
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    shutil.rmtree(path, onerror=clear_readonly)
