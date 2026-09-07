from __future__ import annotations

import shutil
import os
import stat
from dataclasses import replace
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


def test_discard_requires_confirmation_and_never_mutates_host(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxStore

    host_root, store, ready = _ready_sandbox(tmp_path)
    (ready.sandbox_root / "source.py").write_text("sandbox change", encoding="utf-8")
    host_before = (host_root / "source.py").read_bytes()
    plans = []

    report = store.discard(ready.sandbox_id, confirm=lambda plan: plans.append(plan) or False)

    assert report.discarded is False
    assert report.confirmed is False
    assert plans[0].changed_path_count == 1
    assert ready.sandbox_root.is_dir()
    assert (host_root / "source.py").read_bytes() == host_before


def test_discard_persists_discarding_before_removing_workspace_and_keeps_tombstone(monkeypatch, tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxState, SandboxStore

    host_root, store, ready = _ready_sandbox(tmp_path)
    observed = []
    original = store._remove_disposable_workspace

    def observe(metadata):
        assert store._load_metadata(metadata.sandbox_id).state is SandboxState.DISCARDING
        observed.append(metadata.sandbox_id)
        original(metadata)

    monkeypatch.setattr(store, "_remove_disposable_workspace", observe)
    report = store.discard(ready.sandbox_id, confirm=lambda _plan: True)
    resumed = store.load_for_session("session-1", ready.workspace_id)

    assert report.discarded is True
    assert observed == [ready.sandbox_id]
    assert not ready.sandbox_root.exists()
    assert store._load_metadata(ready.sandbox_id).state is SandboxState.DISCARDED
    assert resumed is not None and resumed.state is SandboxState.DISCARDED
    assert (host_root / "source.py").read_text(encoding="utf-8") == "host baseline"


def test_discarding_reconciles_only_after_workspace_is_known_absent(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxState, SandboxStore

    _host_root, store, ready = _ready_sandbox(tmp_path)
    store._write_metadata(replace(ready, state=SandboxState.DISCARDING))

    still_discarding = store.load_for_session("session-1", ready.workspace_id)
    assert still_discarding is not None and still_discarding.state is SandboxState.DISCARDING

    _remove_tree(ready.sandbox_root)
    reconciled = store.load_for_session("session-1", ready.workspace_id)
    assert reconciled is not None and reconciled.state is SandboxState.DISCARDED


def test_discard_rejects_unfinished_apply_evidence(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ApplyState, SandboxApplyError, SandboxApplyService, SandboxStore

    _host_root, store, ready = _ready_sandbox(tmp_path)
    (ready.sandbox_root / "source.py").write_text("sandbox change", encoding="utf-8")
    service = SandboxApplyService(store)
    original_mutate = service._mutate_operation

    def fail_mutation(_operation, _host_root_arg):
        raise SandboxApplyError("simulated Apply interruption")

    service._mutate_operation = fail_mutation  # type: ignore[method-assign]
    interrupted = service.apply(ready.sandbox_id, confirm=lambda _plan: True)
    assert interrupted.state is ApplyState.RECOVERY_REQUIRED
    service._mutate_operation = original_mutate  # type: ignore[method-assign]

    with pytest.raises(SandboxApplyError, match="cannot discard|unfinished Apply"):
        store.discard(ready.sandbox_id, confirm=lambda _plan: True)
    assert ready.sandbox_root.is_dir()


def test_terminal_cleanup_is_idempotent_and_never_deletes_ready_workspace(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxApplyError, SandboxState, SandboxStore

    _host_root, store, ready = _ready_sandbox(tmp_path)
    with pytest.raises(SandboxApplyError, match="active Sandbox"):
        store.cleanup_terminal(ready.sandbox_id)

    store._write_metadata(replace(ready, state=SandboxState.FAILED))
    first = store.cleanup_terminal(ready.sandbox_id)
    second = store.cleanup_terminal(ready.sandbox_id)

    assert first.cleaned is True
    assert second.cleaned is True
    assert not ready.sandbox_root.exists()


def _ready_sandbox(tmp_path: Path):
    from rova.app.workspace.sandbox import SandboxStore

    host_root = tmp_path / "host-project"
    host_root.mkdir()
    (host_root / "source.py").write_text("host baseline", encoding="utf-8")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    created = store.create_unbound(Workspace(host_root))
    imported = store.import_baseline(created.sandbox_id)
    bound = store.bind_session(imported.sandbox_id, "session-1")
    return host_root, store, store.mark_ready(bound.sandbox_id)


def _remove_tree(path: Path) -> None:
    def clear_readonly(function, failed_path, _exception) -> None:
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    shutil.rmtree(path, onerror=clear_readonly)
