from __future__ import annotations

from pathlib import Path

from rova.app.workspace.sandbox import SandboxState, SandboxStore
from rova.app.workspace.sandbox_control import SandboxControl
from rova.app.workspace.workspace import Workspace


def _ready_control(tmp_path: Path) -> tuple[SandboxStore, SandboxControl, Path]:
    host_root = tmp_path / "host"
    host_root.mkdir()
    (host_root / "source.py").write_text("baseline", encoding="utf-8")
    store = SandboxStore(tmp_path / "sandboxes")
    metadata = store.create_unbound(Workspace(host_root))
    metadata = store.import_baseline(metadata.sandbox_id)
    store.bind_session(metadata.sandbox_id, "session-1")
    store.mark_ready(metadata.sandbox_id)
    return store, SandboxControl(store, Workspace(host_root), "session-1", metadata.sandbox_id), host_root


def test_status_is_desensitized_and_reports_a_ready_isolated_environment(tmp_path: Path) -> None:
    _store, control, _host_root = _ready_control(tmp_path)

    status = control.status()

    assert status.environment_kind == "sandbox"
    assert status.sandbox_state is SandboxState.READY
    assert status.host_isolation_active is True
    assert len(status.sandbox_id) == 8
    assert "sandbox_root" not in repr(status)


def test_diff_is_read_only_and_reports_changed_paths(tmp_path: Path) -> None:
    _store, control, host_root = _ready_control(tmp_path)
    control._store.load_for_execution(control._sandbox_id).sandbox_root.joinpath("source.py").write_text("sandbox", encoding="utf-8")

    changed = control.diff()

    assert [item.path for item in changed.changes] == ["source.py"]
    assert (host_root / "source.py").read_text(encoding="utf-8") == "baseline"


def test_discard_uses_the_domain_service_and_never_mutates_the_host(tmp_path: Path) -> None:
    _store, control, host_root = _ready_control(tmp_path)
    control._store.load_for_execution(control._sandbox_id).sandbox_root.joinpath("source.py").write_text("sandbox", encoding="utf-8")

    report = control.discard(confirm=lambda _plan: True)

    assert report.discarded is True
    assert report.state is SandboxState.DISCARDED
    assert (host_root / "source.py").read_text(encoding="utf-8") == "baseline"


def test_explicit_new_sandbox_replaces_only_a_terminal_session_binding(tmp_path: Path) -> None:
    store, control, host_root = _ready_control(tmp_path)
    old_id = control._sandbox_id
    store.discard(old_id, confirm=lambda _plan: True)
    (host_root / "source.py").write_text("host after discard", encoding="utf-8")

    metadata = store.create_new_for_terminal_session(Workspace(host_root), "session-1")

    assert metadata.sandbox_id != old_id
    assert metadata.state is SandboxState.READY
    assert metadata.sandbox_root.joinpath("source.py").read_text(encoding="utf-8") == "host after discard"
    assert store.load_for_session("session-1", metadata.workspace_id).sandbox_id == metadata.sandbox_id  # type: ignore[union-attr]
