import json
from pathlib import Path

import pytest

from rova.app.workspace.sandbox import SandboxStore
from rova.app.workspace.workspace import Workspace


def _ready_sandbox(tmp_path: Path, files: dict[str, bytes]):
    host_root = tmp_path / "host"
    host_root.mkdir()
    for relative, content in files.items():
        target = host_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    imported = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    store.bind_session(imported.sandbox_id, "session1")
    metadata = store.mark_ready(imported.sandbox_id)
    return host_root, store, metadata


def test_apply_requires_independent_confirmation_before_any_host_mutation(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ApplyState, SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"src/a.txt": b"baseline"})
    (metadata.sandbox_root / "src" / "a.txt").write_bytes(b"sandbox")
    service = SandboxApplyService(store)
    seen = []

    report = service.apply(metadata.sandbox_id, confirm=lambda plan: seen.append(plan) or False)

    assert report.applied is False
    assert report.confirmed is False
    assert report.operation_id is None
    assert len(seen) == 1
    assert host_root.joinpath("src/a.txt").read_bytes() == b"baseline"
    assert store._load_metadata(metadata.sandbox_id).state.value == "ready"
    assert not (metadata.sandbox_root.parent / "apply").exists()

    applied = service.apply(metadata.sandbox_id, confirm=lambda _plan: True)
    assert applied.applied is True
    assert applied.state is ApplyState.APPLIED
    assert host_root.joinpath("src/a.txt").read_bytes() == b"sandbox"
    assert store._load_metadata(metadata.sandbox_id).state.value == "applied"


def test_apply_rejects_any_changed_path_host_drift_without_partial_mutation(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"a.txt": b"A", "b.txt": b"B"})
    (metadata.sandbox_root / "a.txt").write_bytes(b"sandbox A")
    (metadata.sandbox_root / "b.txt").write_bytes(b"sandbox B")
    host_root.joinpath("b.txt").write_bytes(b"host B")

    report = SandboxApplyService(store).apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is False
    assert report.confirmed is False
    assert [conflict.path for conflict in report.conflicts] == ["b.txt"]
    assert host_root.joinpath("a.txt").read_bytes() == b"A"
    assert host_root.joinpath("b.txt").read_bytes() == b"host B"
    assert not (metadata.sandbox_root.parent / "apply").exists()


def test_apply_detects_added_deleted_and_ancestor_descendant_conflicts(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"delete.txt": b"A", "folder/base.txt": b"B"})
    (metadata.sandbox_root / "delete.txt").unlink()
    (metadata.sandbox_root / "new.txt").write_bytes(b"new")
    (metadata.sandbox_root / "folder").unlink(missing_ok=True) if (metadata.sandbox_root / "folder").is_file() else None
    for child in (metadata.sandbox_root / "folder").iterdir():
        child.unlink()
    (metadata.sandbox_root / "folder").rmdir()
    (metadata.sandbox_root / "folder").write_bytes(b"replacement")
    host_root.joinpath("delete.txt").unlink()
    host_root.joinpath("new.txt").write_bytes(b"host collision")
    host_root.joinpath("folder/user.txt").write_bytes(b"host nested drift")

    report = SandboxApplyService(store).apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is False
    assert {conflict.path for conflict in report.conflicts} >= {"delete.txt", "new.txt", "folder"}
    assert host_root.joinpath("folder/user.txt").read_bytes() == b"host nested drift"


def test_apply_persists_backup_and_manifest_before_first_host_mutation(monkeypatch, tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"a.txt": b"A"})
    (metadata.sandbox_root / "a.txt").write_bytes(b"B")
    service = SandboxApplyService(store)
    observed: list[str] = []
    original = service._mutate_operation

    def observe(operation, host_root_arg):
        operation_id = service._active_operation_id
        assert operation_id is not None
        manifest_path = metadata.sandbox_root.parent / "apply" / f"{operation_id}.json"
        backup_path = metadata.sandbox_root.parent / "apply-backups" / operation_id / "manifest.json"
        assert manifest_path.is_file()
        assert backup_path.is_file()
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["state"] == "applying"
        observed.append(operation.path)
        return original(operation, host_root_arg)

    monkeypatch.setattr(service, "_mutate_operation", observe)
    report = service.apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is True
    assert observed == ["a.txt"]
    assert host_root.joinpath("a.txt").read_bytes() == b"B"


def test_immediate_revalidation_stops_partial_apply_and_explicit_preimage_restore_repairs_it(monkeypatch, tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ApplyState, SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"a.txt": b"A", "b.txt": b"B"})
    (metadata.sandbox_root / "a.txt").write_bytes(b"sandbox A")
    (metadata.sandbox_root / "b.txt").write_bytes(b"sandbox B")
    service = SandboxApplyService(store)
    original = service._revalidate_operation
    calls = 0

    def drift_before_second(operation, host_root_arg):
        nonlocal calls
        calls += 1
        if calls == 2:
            host_root.joinpath("b.txt").write_bytes(b"late host change")
        return original(operation, host_root_arg)

    monkeypatch.setattr(service, "_revalidate_operation", drift_before_second)
    report = service.apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is False
    assert report.state is ApplyState.RECOVERY_REQUIRED
    assert report.operation_id is not None
    assert host_root.joinpath("a.txt").read_bytes() == b"sandbox A"
    assert host_root.joinpath("b.txt").read_bytes() == b"late host change"

    restored = service.restore_preimages(metadata.sandbox_id, report.operation_id, confirm=lambda _report: True)
    assert restored.state is ApplyState.RECOVERY_REQUIRED
    assert restored.conflicts[0].path == "b.txt"
    assert host_root.joinpath("a.txt").read_bytes() == b"sandbox A"
    assert host_root.joinpath("b.txt").read_bytes() == b"late host change"


def test_apply_only_touches_changed_paths_and_applied_sandbox_cannot_be_applied_again(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxApplyError, SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"changed.txt": b"A", "untouched.txt": b"U"})
    (metadata.sandbox_root / "changed.txt").write_bytes(b"B")
    host_root.joinpath("user-note.txt").write_bytes(b"host-only")
    service = SandboxApplyService(store)

    report = service.apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is True
    assert host_root.joinpath("changed.txt").read_bytes() == b"B"
    assert host_root.joinpath("untouched.txt").read_bytes() == b"U"
    assert host_root.joinpath("user-note.txt").read_bytes() == b"host-only"
    with pytest.raises(SandboxApplyError, match="applied"):
        service.build_plan(metadata.sandbox_id)


def test_apply_orders_file_to_directory_replacement_before_descendant_creation(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"output": b"old-file"})
    (metadata.sandbox_root / "output").unlink()
    (metadata.sandbox_root / "output").mkdir()
    (metadata.sandbox_root / "output" / "result.txt").write_bytes(b"new")

    report = SandboxApplyService(store).apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is True
    assert host_root.joinpath("output/result.txt").read_bytes() == b"new"


def test_apply_rejects_mode_drift_on_posix(tmp_path: Path) -> None:
    if __import__("os").name == "nt":
        pytest.skip("Windows does not expose POSIX executable mode in FileState")
    from rova.app.workspace.sandbox import SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"script.sh": b"echo baseline\n"})
    sandbox_file = metadata.sandbox_root / "script.sh"
    sandbox_file.chmod(0o755)
    host_root.joinpath("script.sh").chmod(0o755)

    report = SandboxApplyService(store).apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is False
    assert report.conflicts[0].path == "script.sh"


def test_apply_replaces_symlink_without_following_it(tmp_path: Path) -> None:
    import os

    from rova.app.workspace.sandbox import SandboxApplyService

    host_root = tmp_path / "host"
    host_root.mkdir()
    try:
        os.symlink("one.txt", host_root / "link")
    except OSError:
        pytest.skip("symlink creation is unavailable for this test user")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    imported = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    store.bind_session(imported.sandbox_id, "session1")
    metadata = store.mark_ready(imported.sandbox_id)
    (metadata.sandbox_root / "link").unlink()
    os.symlink("two.txt", metadata.sandbox_root / "link")

    report = SandboxApplyService(store).apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert report.applied is True
    assert os.readlink(host_root / "link") == "two.txt"


def test_restore_preimages_recovers_a_completed_mutation_even_if_progress_was_not_durable(monkeypatch, tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ApplyState, SandboxApplyError, SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"a.txt": b"A", "b.txt": b"B"})
    (metadata.sandbox_root / "a.txt").write_bytes(b"sandbox A")
    (metadata.sandbox_root / "b.txt").write_bytes(b"sandbox B")
    service = SandboxApplyService(store)
    original_mutate = service._mutate_operation

    def fail_after_first_mutation(operation, host_root_arg):
        if operation.path == "b.txt":
            raise SandboxApplyError("simulated crash after first Host mutation")
        return original_mutate(operation, host_root_arg)

    monkeypatch.setattr(service, "_mutate_operation", fail_after_first_mutation)
    monkeypatch.setattr(service, "_mark_operation_completed", lambda *_args: None)
    interrupted = service.apply(metadata.sandbox_id, confirm=lambda _plan: True)

    assert interrupted.state is ApplyState.RECOVERY_REQUIRED
    assert host_root.joinpath("a.txt").read_bytes() == b"sandbox A"

    restored = service.restore_preimages(metadata.sandbox_id, interrupted.operation_id, confirm=lambda _report: True)

    assert restored.state is ApplyState.RESTORED
    assert host_root.joinpath("a.txt").read_bytes() == b"A"


def test_restore_preimages_refuses_to_overwrite_post_apply_host_drift(monkeypatch, tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ApplyState, SandboxApplyError, SandboxApplyService

    host_root, store, metadata = _ready_sandbox(tmp_path, {"a.txt": b"A", "b.txt": b"B"})
    (metadata.sandbox_root / "a.txt").write_bytes(b"sandbox A")
    (metadata.sandbox_root / "b.txt").write_bytes(b"sandbox B")
    service = SandboxApplyService(store)
    original_mutate = service._mutate_operation

    def fail_after_first_mutation(operation, host_root_arg):
        if operation.path == "b.txt":
            raise SandboxApplyError("simulated interrupted Apply")
        return original_mutate(operation, host_root_arg)

    monkeypatch.setattr(service, "_mutate_operation", fail_after_first_mutation)
    interrupted = service.apply(metadata.sandbox_id, confirm=lambda _plan: True)
    host_root.joinpath("a.txt").write_bytes(b"user changed Host")

    restored = service.restore_preimages(metadata.sandbox_id, interrupted.operation_id, confirm=lambda _report: True)

    assert restored.state is ApplyState.RECOVERY_REQUIRED
    assert restored.conflicts[0].path == "a.txt"
    assert host_root.joinpath("a.txt").read_bytes() == b"user changed Host"
