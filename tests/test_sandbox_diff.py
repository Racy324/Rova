import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from rova.app.workspace.sandbox import SandboxStore
from rova.app.workspace.workspace import Workspace


def _imported_sandbox(tmp_path: Path):
    host_root = tmp_path / "host"
    host_root.mkdir()
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    return host_root, store, metadata


def test_private_b0_preserves_imported_regular_file_bytes_despite_attributes(tmp_path: Path) -> None:
    host_root = tmp_path / "host"
    host_root.mkdir()
    raw = b"first\r\nsecond\r\n"
    (host_root / "source.txt").write_bytes(raw)
    (host_root / ".gitattributes").write_text("*.txt text eol=lf\n", encoding="utf-8")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")

    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)

    assert metadata.baseline_commit_oid is not None
    assert _git_bytes(metadata.sandbox_root, "show", f"{metadata.baseline_commit_oid}:source.txt") == raw


def test_changed_set_is_structured_deterministic_and_never_reads_host_changes(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ChangeKind, SandboxDiffService

    host_root, store, metadata = _imported_sandbox(tmp_path)
    (host_root / "base.txt").write_text("baseline", encoding="utf-8")
    # Re-import after establishing the actual baseline content.
    (host_root / "gone.txt").write_text("remove me", encoding="utf-8")
    store = SandboxStore(tmp_path / "rova-data-2" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    sandbox = metadata.sandbox_root
    (sandbox / "base.txt").write_text("sandbox change", encoding="utf-8")
    (sandbox / "added.bin").write_bytes(b"\x00\xff\x01")
    (sandbox / "gone.txt").unlink()
    host_root.joinpath("base.txt").write_text("host-only change", encoding="utf-8")
    host_root.joinpath("host-only.txt").write_text("not part of sandbox", encoding="utf-8")

    changed_set = SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)

    assert changed_set.sandbox_id == metadata.sandbox_id
    assert changed_set.baseline_oid == metadata.baseline_commit_oid
    assert [change.path for change in changed_set.changes] == sorted(change.path for change in changed_set.changes)
    assert [change.kind for change in changed_set.changes] == [ChangeKind.ADDED, ChangeKind.MODIFIED, ChangeKind.DELETED]
    added, modified, deleted = changed_set.changes
    assert added.path == "added.bin"
    assert added.current is not None and added.current.binary is True
    assert modified.path == "base.txt"
    assert modified.baseline is not None and modified.current is not None
    assert modified.baseline.content_digest != modified.current.content_digest
    assert deleted.path == "gone.txt" and deleted.current is None
    assert host_root.joinpath("base.txt").read_text(encoding="utf-8") == "host-only change"


def test_changed_set_records_deleted_files_empty_directories_and_ignores_private_git(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ChangeKind, SandboxDiffService

    host_root = tmp_path / "host"
    (host_root / "empty-baseline").mkdir(parents=True)
    (host_root / "delete.txt").write_text("delete", encoding="utf-8")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    (metadata.sandbox_root / "empty-baseline").rmdir()
    (metadata.sandbox_root / "delete.txt").unlink()
    (metadata.sandbox_root / "new-empty").mkdir()
    (metadata.sandbox_root / ".git" / "internal-marker").write_text("ignore", encoding="utf-8")

    changed_set = SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)

    assert [(change.path, change.kind) for change in changed_set.changes] == [
        ("delete.txt", ChangeKind.DELETED),
        ("empty-baseline", ChangeKind.DELETED),
        ("new-empty", ChangeKind.ADDED),
    ]
    assert all(not change.path.startswith(".git/") for change in changed_set.changes)


def test_changed_set_reports_mode_changes_without_content_changes_when_supported(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows does not provide POSIX executable-bit fidelity")
    from rova.app.workspace.sandbox import ChangeKind, SandboxDiffService

    host_root = tmp_path / "host"
    host_root.mkdir()
    path = host_root / "script.py"
    path.write_text("print('x')\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    sandbox_path = metadata.sandbox_root / "script.py"
    sandbox_path.chmod(sandbox_path.stat().st_mode & ~stat.S_IXUSR)

    changed_set = SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)

    assert [(change.path, change.kind) for change in changed_set.changes] == [("script.py", ChangeKind.MODE_CHANGED)]


def test_changed_set_fails_closed_when_current_tree_exceeds_limits(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxDiffError, SandboxDiffLimits, SandboxDiffService

    host_root, store, metadata = _imported_sandbox(tmp_path)
    (metadata.sandbox_root / "large.bin").write_bytes(b"x" * 32)

    with pytest.raises(SandboxDiffError, match="resource limit"):
        SandboxDiffService(store, limits=SandboxDiffLimits(max_files=10, max_total_bytes=16, max_single_file_bytes=16)).compute_changed_set(metadata.sandbox_id)

    assert not any(host_root.iterdir())


def test_changed_set_ignores_mtime_only_changes_and_reports_no_changes(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxDiffService

    host_root = tmp_path / "host"
    host_root.mkdir()
    (host_root / "stable.txt").write_bytes(b"same bytes")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    os.utime(metadata.sandbox_root / "stable.txt", None)

    changed_set = SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)

    assert changed_set.changes == ()
    assert changed_set.summary.added == changed_set.summary.modified == changed_set.summary.deleted == 0


def test_changed_set_represents_file_directory_type_replacement_and_nested_empty_directory(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ChangeKind, FileStateKind, SandboxDiffService

    host_root = tmp_path / "host"
    host_root.mkdir()
    (host_root / "replace-me").write_text("file", encoding="utf-8")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    target = metadata.sandbox_root / "replace-me"
    target.unlink()
    (target / "nested-empty").mkdir(parents=True)

    changed_set = SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)

    assert [(change.path, change.kind) for change in changed_set.changes] == [
        ("replace-me", ChangeKind.TYPE_CHANGED),
        ("replace-me/nested-empty", ChangeKind.ADDED),
    ]
    replacement = changed_set.changes[0]
    assert replacement.baseline is not None and replacement.baseline.kind is FileStateKind.REGULAR
    assert replacement.current is not None and replacement.current.kind is FileStateKind.DIRECTORY


def test_changed_set_handles_internal_symlinks_without_following_them(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import ChangeKind, SandboxDiffService

    host_root = tmp_path / "host"
    host_root.mkdir()
    (host_root / "target-a.txt").write_text("a", encoding="utf-8")
    (host_root / "target-b.txt").write_text("b", encoding="utf-8")
    try:
        os.symlink("target-a.txt", host_root / "link")
    except OSError:
        pytest.skip("the current Windows account cannot create symlinks")
    store = SandboxStore(tmp_path / "rova-data" / "sandboxes")
    metadata = store.import_baseline(store.create_unbound(Workspace(host_root)).sandbox_id)
    assert SandboxDiffService(store).compute_changed_set(metadata.sandbox_id).changes == ()
    (metadata.sandbox_root / "link").unlink()
    os.symlink("target-b.txt", metadata.sandbox_root / "link")

    changed_set = SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)

    assert [(change.path, change.kind) for change in changed_set.changes] == [("link", ChangeKind.SYMLINK_CHANGED)]


def test_changed_set_fails_closed_when_private_b0_is_missing(tmp_path: Path) -> None:
    from rova.app.workspace.sandbox import SandboxDiffError, SandboxDiffService

    _host_root, store, metadata = _imported_sandbox(tmp_path)
    shutil.rmtree(metadata.sandbox_root / ".git", onerror=_clear_readonly)

    with pytest.raises(SandboxDiffError, match="immutable baseline"):
        SandboxDiffService(store).compute_changed_set(metadata.sandbox_id)


def _git_bytes(cwd: Path, *arguments: str) -> bytes:
    process = subprocess.run(["git", "-C", str(cwd), *arguments], check=True, capture_output=True)
    return process.stdout


def _clear_readonly(function, path, _exc_info) -> None:
    Path(path).chmod(0o700)
    function(path)
