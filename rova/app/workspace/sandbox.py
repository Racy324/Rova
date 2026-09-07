from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from .workspace import Workspace


class SandboxError(RuntimeError):
    """A durable Sandbox lifecycle failure."""


class SandboxConflictError(SandboxError):
    """A Host Workspace is already owned by an active Sandbox."""


class SandboxImportError(SandboxError):
    """The Host Workspace could not be imported as a complete Sandbox baseline."""


class SandboxDiffError(SandboxError):
    """A ChangedSet could not be computed completely and safely."""


class SandboxState(str, Enum):
    CREATING = "creating"
    READY = "ready"
    APPLYING = "applying"
    APPLIED = "applied"
    DISCARDING = "discarding"
    DISCARDED = "discarded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class FileStateKind(str, Enum):
    REGULAR = "regular"
    SYMLINK = "symlink"
    DIRECTORY = "directory"


class ChangeKind(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    MODE_CHANGED = "mode_changed"
    SYMLINK_CHANGED = "symlink_changed"
    TYPE_CHANGED = "type_changed"


@dataclass(frozen=True)
class FileState:
    kind: FileStateKind
    content_digest: str | None
    size_bytes: int | None
    executable: bool | None
    symlink_target: str | None
    binary: bool = False


@dataclass(frozen=True)
class PathChange:
    path: str
    kind: ChangeKind
    baseline: FileState | None
    current: FileState | None


@dataclass(frozen=True)
class ChangedSetSummary:
    added: int = 0
    modified: int = 0
    deleted: int = 0
    mode_changed: int = 0
    symlink_changed: int = 0
    type_changed: int = 0


@dataclass(frozen=True)
class ChangedSet:
    sandbox_id: str
    baseline_oid: str
    generated_at: str
    changes: tuple[PathChange, ...]
    summary: ChangedSetSummary


@dataclass(frozen=True)
class SandboxDiffLimits:
    max_files: int = 100_000
    max_total_bytes: int = 1_073_741_824
    max_single_file_bytes: int = 268_435_456

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (
            self.max_files,
            self.max_total_bytes,
            self.max_single_file_bytes,
        )):
            raise ValueError("Sandbox diff limits must be positive integers")


@dataclass(frozen=True)
class SandboxMetadata:
    version: int
    sandbox_id: str
    session_id: str | None
    workspace_id: str
    host_root: Path
    sandbox_root: Path
    backend: str
    state: SandboxState
    baseline_commit_oid: str | None
    baseline_fingerprint: str | None
    created_at: str
    updated_at: str
    active_apply_id: str | None = None


class SandboxStore:
    """Durable Session-to-Sandbox metadata, independent from conversation JSONL."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create_unbound(self, host_workspace: Workspace) -> SandboxMetadata:
        workspace_id = workspace_identity(host_workspace.root)
        sandbox_id = uuid4().hex
        self._reserve_workspace(workspace_id, sandbox_id)
        sandbox_dir = self.root / sandbox_id
        sandbox_root = sandbox_dir / "workspace"
        now = _utc_now()
        metadata = SandboxMetadata(
            version=1,
            sandbox_id=sandbox_id,
            session_id=None,
            workspace_id=workspace_id,
            host_root=host_workspace.root,
            sandbox_root=sandbox_root,
            backend="docker_sandbox",
            state=SandboxState.CREATING,
            baseline_commit_oid=None,
            baseline_fingerprint=None,
            created_at=now,
            updated_at=now,
        )
        try:
            sandbox_root.mkdir(parents=True, exist_ok=False)
            self._write_metadata(metadata)
        except Exception:
            self._release_workspace_reservation(workspace_id, sandbox_id)
            raise
        return metadata

    def bind_session(self, sandbox_id: str, session_id: str) -> SandboxMetadata:
        if not isinstance(session_id, str) or not session_id:
            raise SandboxError("session_id must be a non-empty string")
        metadata = self._load_metadata(sandbox_id)
        if metadata.session_id not in {None, session_id}:
            raise SandboxConflictError("Sandbox is already bound to a different Session")
        pointer = self._session_pointer_path(session_id)
        if pointer.exists():
            existing = self._read_json(pointer)
            if existing.get("sandbox_id") != sandbox_id:
                raise SandboxConflictError("Session already has an active Sandbox")
        bound = replace(metadata, session_id=session_id, updated_at=_utc_now())
        self._write_metadata(bound)
        self._write_json(pointer, {"version": 1, "sandbox_id": sandbox_id, "workspace_id": bound.workspace_id})
        return bound

    def mark_ready(self, sandbox_id: str) -> SandboxMetadata:
        metadata = self._load_metadata(sandbox_id)
        if metadata.session_id is None:
            raise SandboxError("Sandbox must be bound to a Session before it becomes ready")
        if metadata.state is not SandboxState.CREATING:
            raise SandboxError(f"Sandbox cannot become ready from state {metadata.state.value}")
        if metadata.baseline_commit_oid is None or not _private_commit_exists(metadata.sandbox_root, metadata.baseline_commit_oid):
            raise SandboxError("Sandbox cannot become ready without an immutable baseline")
        ready = replace(metadata, state=SandboxState.READY, updated_at=_utc_now())
        self._write_metadata(ready)
        return ready

    def import_baseline(self, sandbox_id: str) -> SandboxMetadata:
        """Copy the actual Host tree and create the immutable private Git B0."""
        metadata = self._load_metadata(sandbox_id)
        if metadata.state is not SandboxState.CREATING:
            raise SandboxImportError(f"Sandbox cannot import a baseline from state {metadata.state.value}")
        if metadata.baseline_commit_oid is not None:
            raise SandboxImportError("Sandbox already has an immutable baseline")
        staging = self.root / ".staging" / uuid4().hex
        try:
            manifest = _copy_workspace_tree(
                metadata.host_root,
                staging,
                excluded_roots=self._embedded_owned_roots(metadata.host_root),
            )
            baseline_commit_oid = _initialize_private_baseline(staging)
            baseline_fingerprint = _manifest_fingerprint(manifest)
            if metadata.sandbox_root.exists():
                metadata.sandbox_root.rmdir()
            staging.replace(metadata.sandbox_root)
            self._write_json(
                metadata.sandbox_root.parent / "baseline.manifest.json",
                {"version": 1, "entries": manifest, "fingerprint": baseline_fingerprint},
            )
            imported = replace(
                metadata,
                baseline_commit_oid=baseline_commit_oid,
                baseline_fingerprint=baseline_fingerprint,
                updated_at=_utc_now(),
            )
            self._write_metadata(imported)
            return imported
        except SandboxImportError:
            self._mark_import_failed(metadata)
            raise
        except (OSError, subprocess.SubprocessError) as error:
            self._mark_import_failed(metadata)
            raise SandboxImportError(f"could not import Sandbox baseline: {error}") from error
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def load_for_session(self, session_id: str, workspace_id: str) -> SandboxMetadata | None:
        pointer = self._session_pointer_path(session_id)
        if not pointer.exists():
            return None
        value = self._read_json(pointer)
        if value.get("workspace_id") != workspace_id:
            raise SandboxConflictError("Session Sandbox belongs to a different Host Workspace")
        sandbox_id = value.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise SandboxError("Sandbox pointer is invalid")
        metadata = self._load_metadata(sandbox_id)
        if metadata.session_id != session_id or metadata.workspace_id != workspace_id:
            raise SandboxError("Sandbox metadata does not match the Session pointer")
        if metadata.state is SandboxState.READY and (
            not metadata.sandbox_root.is_dir()
            or metadata.baseline_commit_oid is None
            or not _private_commit_exists(metadata.sandbox_root, metadata.baseline_commit_oid)
        ):
            metadata = replace(metadata, state=SandboxState.ABANDONED, updated_at=_utc_now())
            self._write_metadata(metadata)
        return metadata

    def _mark_import_failed(self, metadata: SandboxMetadata) -> None:
        if metadata.state is SandboxState.CREATING:
            self._write_metadata(replace(metadata, state=SandboxState.FAILED, updated_at=_utc_now()))

    def _embedded_owned_roots(self, host_root: Path) -> tuple[Path, ...]:
        data_root = self.root.parent.resolve()
        if data_root == host_root:
            raise SandboxImportError("Host Workspace cannot be the Rova data root")
        try:
            data_root.relative_to(host_root)
        except ValueError:
            return ()
        return (data_root,)

    def _reserve_workspace(self, workspace_id: str, sandbox_id: str) -> None:
        path = self.root / "by-workspace" / f"{workspace_id}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(sandbox_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise SandboxConflictError("an active Sandbox already exists for this Host Workspace") from error

    def _release_workspace_reservation(self, workspace_id: str, sandbox_id: str) -> None:
        path = self.root / "by-workspace" / f"{workspace_id}.lock"
        try:
            if path.exists() and path.read_text(encoding="utf-8").strip() == sandbox_id:
                path.unlink()
        except OSError:
            return

    def _load_metadata(self, sandbox_id: str) -> SandboxMetadata:
        value = self._read_json(self.root / sandbox_id / "state.json")
        try:
            return SandboxMetadata(
                version=int(value["version"]),
                sandbox_id=_required_string(value, "sandbox_id"),
                session_id=_optional_string(value, "session_id"),
                workspace_id=_required_string(value, "workspace_id"),
                host_root=Path(_required_string(value, "host_root")),
                sandbox_root=Path(_required_string(value, "sandbox_root")),
                backend=_required_string(value, "backend"),
                state=SandboxState(_required_string(value, "state")),
                baseline_commit_oid=_optional_string(value, "baseline_commit_oid"),
                baseline_fingerprint=_optional_string(value, "baseline_fingerprint"),
                created_at=_required_string(value, "created_at"),
                updated_at=_required_string(value, "updated_at"),
                active_apply_id=_optional_string(value, "active_apply_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SandboxError(f"Sandbox metadata is invalid: {error}") from error

    def _write_metadata(self, metadata: SandboxMetadata) -> None:
        self._write_json(
            self.root / metadata.sandbox_id / "state.json",
            {
                "version": metadata.version,
                "sandbox_id": metadata.sandbox_id,
                "session_id": metadata.session_id,
                "workspace_id": metadata.workspace_id,
                "host_root": str(metadata.host_root),
                "sandbox_root": str(metadata.sandbox_root),
                "backend": metadata.backend,
                "state": metadata.state.value,
                "baseline_commit_oid": metadata.baseline_commit_oid,
                "baseline_fingerprint": metadata.baseline_fingerprint,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "active_apply_id": metadata.active_apply_id,
            },
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SandboxError(f"could not read Sandbox metadata: {error}") from error
        if not isinstance(value, dict):
            raise SandboxError("Sandbox metadata must be an object")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise SandboxError(f"could not persist Sandbox metadata: {error}") from error

    def _session_pointer_path(self, session_id: str) -> Path:
        return self.root / "by-session" / f"{session_id}.json"


class SandboxDiffService:
    """Compute the Sandbox-owned B0 -> current-tree domain ChangedSet."""

    def __init__(self, store: SandboxStore, *, limits: SandboxDiffLimits | None = None) -> None:
        self._store = store
        self._limits = limits or SandboxDiffLimits()

    def compute_changed_set(self, sandbox_id: str) -> ChangedSet:
        metadata = self._store._load_metadata(sandbox_id)
        if metadata.baseline_commit_oid is None or not _private_commit_exists(metadata.sandbox_root, metadata.baseline_commit_oid):
            raise SandboxDiffError("Sandbox immutable baseline is unavailable")
        if not metadata.sandbox_root.is_dir():
            raise SandboxDiffError("Sandbox workspace is unavailable")
        baseline = _scan_baseline_tree(metadata)
        current = _scan_workspace_tree(metadata.sandbox_root, self._limits)
        changes = tuple(
            change
            for path in sorted(set(baseline) | set(current))
            if (change := _path_change(path, baseline.get(path), current.get(path))) is not None
        )
        return ChangedSet(
            sandbox_id=metadata.sandbox_id,
            baseline_oid=metadata.baseline_commit_oid,
            generated_at=_utc_now(),
            changes=changes,
            summary=_summarize_changes(changes),
        )


def workspace_identity(root: Path) -> str:
    normalized = str(Path(root).resolve())
    if os.name == "nt":
        normalized = os.path.normcase(normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _required_string(value: dict, key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: dict, key: str) -> str | None:
    result = value.get(key)
    if result is not None and not isinstance(result, str):
        raise ValueError(f"{key} must be text or null")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_workspace_tree(
    source_root: Path,
    destination_root: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> list[dict[str, str | int]]:
    """Copy the complete managed source tree without following links or `.git`."""
    destination_root.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, str | int]] = []

    def copy_directory(source: Path, destination: Path, relative: Path) -> None:
        for entry in sorted(source.iterdir(), key=lambda value: value.name):
            if entry.name == ".git":
                continue
            if _is_within_any(entry, excluded_roots):
                continue
            target = destination / entry.name
            entry_relative = relative / entry.name
            entry_stat = entry.lstat()
            mode = stat.S_IMODE(entry_stat.st_mode)
            if stat.S_ISLNK(entry_stat.st_mode):
                link_target = os.readlink(entry)
                _validate_internal_symlink(entry, source_root)
                try:
                    os.symlink(link_target, target, target_is_directory=entry.is_dir())
                except OSError as error:
                    raise SandboxImportError(f"could not recreate symlink {entry_relative.as_posix()}: {error}") from error
                manifest.append({"path": entry_relative.as_posix(), "kind": "symlink", "mode": mode, "target": link_target})
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                target.mkdir()
                copy_directory(entry, target, entry_relative)
                _copy_mode(target, mode)
                manifest.append({"path": entry_relative.as_posix(), "kind": "directory", "mode": mode})
                continue
            if stat.S_ISREG(entry_stat.st_mode):
                shutil.copyfile(entry, target)
                _copy_mode(target, mode)
                manifest.append(
                    {
                        "path": entry_relative.as_posix(),
                        "kind": "file",
                        "mode": mode,
                        "sha256": _file_sha256(entry),
                    }
                )
                continue
            raise SandboxImportError(f"unsupported workspace entry: {entry_relative.as_posix()}")

    copy_directory(source_root, destination_root, Path())
    return sorted(manifest, key=lambda entry: str(entry["path"]))


def _is_within_any(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _validate_internal_symlink(path: Path, source_root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(source_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise SandboxImportError(f"symlink escapes Host Workspace: {path}") from error


def _copy_mode(path: Path, mode: int) -> None:
    if os.name != "nt":
        os.chmod(path, mode)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(entries: list[dict[str, str | int]]) -> str:
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _initialize_private_baseline(workspace_root: Path) -> str:
    """Create B0 with Git plumbing so attributes and filters cannot alter bytes."""
    _run_git(workspace_root, "init", "--quiet", "--initial-branch=rova-baseline")
    index_entries: list[tuple[int, str, str]] = []

    def add_directory(directory: Path) -> None:
        for entry in sorted(directory.iterdir(), key=lambda value: value.name):
            if entry.name == ".git":
                continue
            relative = entry.relative_to(workspace_root).as_posix()
            entry_stat = entry.lstat()
            if stat.S_ISLNK(entry_stat.st_mode):
                raw = os.fsencode(os.readlink(entry))
                mode = 0o120000
            elif stat.S_ISDIR(entry_stat.st_mode):
                add_directory(entry)
                continue
            elif stat.S_ISREG(entry_stat.st_mode):
                raw = entry.read_bytes()
                mode = 0o100755 if entry_stat.st_mode & stat.S_IXUSR else 0o100644
            else:
                raise SandboxImportError(f"unsupported workspace entry: {relative}")
            oid = _run_git_bytes(workspace_root, "hash-object", "-w", "--stdin", input_bytes=raw).decode("ascii").strip()
            if not oid:
                raise SandboxImportError("private Sandbox Git did not return a blob id")
            index_entries.append((mode, oid, relative))

    add_directory(workspace_root)
    _run_git_bytes(
        workspace_root,
        "update-index",
        "-z",
        "--index-info",
        input_bytes=b"".join(
            f"{mode:o} {oid}\t{path}".encode("utf-8") + b"\0" for mode, oid, path in index_entries
        ),
    )
    tree_oid = _run_git(workspace_root, "write-tree").strip()
    baseline_commit_oid = _run_git(
        workspace_root,
        "-c",
        "user.name=Rova Sandbox",
        "-c",
        "user.email=rova-sandbox@local.invalid",
        "commit-tree",
        tree_oid,
        "-m",
        "Rova Sandbox Baseline",
    ).strip()
    _run_git(workspace_root, "update-ref", "refs/heads/rova-baseline", baseline_commit_oid)
    if _run_git(workspace_root, "rev-list", "--count", "HEAD").strip() != "1":
        raise SandboxImportError("private Sandbox Git must contain exactly one baseline commit")
    return baseline_commit_oid


def _private_commit_exists(workspace_root: Path, commit_oid: str) -> bool:
    if not workspace_root.is_dir() or not (workspace_root / ".git").is_dir():
        return False
    try:
        _run_git(workspace_root, "cat-file", "-e", f"{commit_oid}^{{commit}}")
    except SandboxImportError:
        return False
    return True


def _run_git(workspace_root: Path, *arguments: str) -> str:
    output = _run_git_bytes(workspace_root, *arguments)
    try:
        return output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SandboxImportError("private Sandbox Git returned invalid text output") from error


def _run_git_bytes(workspace_root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    environment = _private_git_environment()
    try:
        process = subprocess.run(
            ["git", "-C", str(workspace_root), *arguments],
            check=False,
            capture_output=True,
            input=input_bytes,
            env=environment,
        )
    except OSError as error:
        raise SandboxImportError(f"private Sandbox Git is unavailable: {error}") from error
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip() or process.stdout.decode("utf-8", errors="replace").strip() or "unknown Git error"
        raise SandboxImportError(f"private Sandbox Git command failed: {detail}")
    return process.stdout


def _private_git_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }


def _scan_baseline_tree(metadata: SandboxMetadata) -> dict[str, FileState]:
    try:
        manifest_value = SandboxStore._read_json(metadata.sandbox_root.parent / "baseline.manifest.json")
        manifest_entries = manifest_value["entries"]
    except (KeyError, TypeError, SandboxError) as error:
        raise SandboxDiffError("Sandbox baseline manifest is unavailable") from error
    if not isinstance(manifest_entries, list):
        raise SandboxDiffError("Sandbox baseline manifest is invalid")
    states: dict[str, FileState] = {}
    for entry in manifest_entries:
        if not isinstance(entry, dict):
            raise SandboxDiffError("Sandbox baseline manifest is invalid")
        path = _logical_path(entry.get("path"))
        if entry.get("kind") == "directory":
            states[path] = FileState(FileStateKind.DIRECTORY, None, None, None, None)
    output = _run_git_bytes(metadata.sandbox_root, "ls-tree", "-r", "-z", metadata.baseline_commit_oid or "")
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            raw_mode, object_type, raw_oid = header.split(b" ", 2)
            mode = int(raw_mode, 8)
            oid = raw_oid.decode("ascii")
            path = _logical_path(os.fsdecode(raw_path))
        except (ValueError, UnicodeDecodeError) as error:
            raise SandboxDiffError("private Sandbox Git tree is invalid") from error
        if object_type != b"blob":
            raise SandboxDiffError("private Sandbox Git baseline contains an unsupported tree entry")
        content = _run_git_bytes(metadata.sandbox_root, "cat-file", "blob", oid)
        if mode == 0o120000:
            states[path] = FileState(
                FileStateKind.SYMLINK,
                _sha256(content),
                len(content),
                None,
                os.fsdecode(content),
            )
        elif mode in {0o100644, 0o100755}:
            states[path] = FileState(
                FileStateKind.REGULAR,
                _sha256(content),
                len(content),
                None if os.name == "nt" else mode == 0o100755,
                None,
                b"\0" in content,
            )
        else:
            raise SandboxDiffError("private Sandbox Git baseline contains an unsupported file mode")
    return states


def _scan_workspace_tree(root: Path, limits: SandboxDiffLimits) -> dict[str, FileState]:
    states: dict[str, FileState] = {}
    file_count = 0
    total_bytes = 0

    def register(path: str, state: FileState) -> None:
        nonlocal file_count, total_bytes
        file_count += 1
        if file_count > limits.max_files:
            raise SandboxDiffError("Sandbox ChangedSet resource limit exceeded: max_files")
        if state.size_bytes is not None:
            if state.size_bytes > limits.max_single_file_bytes:
                raise SandboxDiffError("Sandbox ChangedSet resource limit exceeded: max_single_file_bytes")
            total_bytes += state.size_bytes
            if total_bytes > limits.max_total_bytes:
                raise SandboxDiffError("Sandbox ChangedSet resource limit exceeded: max_total_bytes")
        states[path] = state

    def visit(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError as error:
            raise SandboxDiffError(f"could not scan Sandbox workspace: {error}") from error
        for entry in entries:
            if entry.name == ".git":
                continue
            entry_relative = relative / entry.name
            path = _logical_path(entry_relative.as_posix())
            try:
                entry_stat = entry.lstat()
            except OSError as error:
                raise SandboxDiffError(f"could not inspect Sandbox entry: {path}") from error
            if stat.S_ISLNK(entry_stat.st_mode):
                target = os.readlink(entry)
                target_bytes = os.fsencode(target)
                register(path, FileState(FileStateKind.SYMLINK, _sha256(target_bytes), len(target_bytes), None, target))
            elif stat.S_ISDIR(entry_stat.st_mode):
                register(path, FileState(FileStateKind.DIRECTORY, None, None, None, None))
                visit(entry, entry_relative)
            elif stat.S_ISREG(entry_stat.st_mode):
                digest, size_bytes, binary = _regular_file_facts(entry, limits.max_single_file_bytes)
                register(
                    path,
                    FileState(
                        FileStateKind.REGULAR,
                        digest,
                        size_bytes,
                        None if os.name == "nt" else bool(entry_stat.st_mode & stat.S_IXUSR),
                        None,
                        binary,
                    ),
                )
            else:
                raise SandboxDiffError(f"unsupported Sandbox entry: {path}")

    visit(root, Path())
    return states


def _regular_file_facts(path: Path, max_size: int) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    size_bytes = 0
    binary = False
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size_bytes += len(chunk)
                if size_bytes > max_size:
                    raise SandboxDiffError("Sandbox ChangedSet resource limit exceeded: max_single_file_bytes")
                digest.update(chunk)
                binary = binary or b"\0" in chunk
    except SandboxDiffError:
        raise
    except OSError as error:
        raise SandboxDiffError(f"could not read Sandbox file: {path.name}") from error
    return digest.hexdigest(), size_bytes, binary


def _path_change(path: str, baseline: FileState | None, current: FileState | None) -> PathChange | None:
    if baseline is None and current is not None:
        return PathChange(path, ChangeKind.ADDED, None, current)
    if baseline is not None and current is None:
        return PathChange(path, ChangeKind.DELETED, baseline, None)
    assert baseline is not None and current is not None
    if baseline.kind is not current.kind:
        return PathChange(path, ChangeKind.TYPE_CHANGED, baseline, current)
    if baseline.kind is FileStateKind.SYMLINK and baseline.symlink_target != current.symlink_target:
        return PathChange(path, ChangeKind.SYMLINK_CHANGED, baseline, current)
    if baseline.kind is FileStateKind.REGULAR:
        if baseline.content_digest != current.content_digest:
            return PathChange(path, ChangeKind.MODIFIED, baseline, current)
        if baseline.executable is not None and current.executable is not None and baseline.executable != current.executable:
            return PathChange(path, ChangeKind.MODE_CHANGED, baseline, current)
    return None


def _summarize_changes(changes: tuple[PathChange, ...]) -> ChangedSetSummary:
    counts = {kind: 0 for kind in ChangeKind}
    for change in changes:
        counts[change.kind] += 1
    return ChangedSetSummary(
        added=counts[ChangeKind.ADDED],
        modified=counts[ChangeKind.MODIFIED],
        deleted=counts[ChangeKind.DELETED],
        mode_changed=counts[ChangeKind.MODE_CHANGED],
        symlink_changed=counts[ChangeKind.SYMLINK_CHANGED],
        type_changed=counts[ChangeKind.TYPE_CHANGED],
    )


def _logical_path(value: object) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise SandboxDiffError("Sandbox path is invalid")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SandboxDiffError("Sandbox path is invalid")
    return "/".join(parts)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
