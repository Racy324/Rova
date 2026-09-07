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


class SandboxState(str, Enum):
    CREATING = "creating"
    READY = "ready"
    APPLYING = "applying"
    APPLIED = "applied"
    DISCARDING = "discarding"
    DISCARDED = "discarded"
    FAILED = "failed"
    ABANDONED = "abandoned"


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
    _run_git(workspace_root, "init", "--quiet", "--initial-branch=rova-baseline")
    _run_git(workspace_root, "add", "--all", "--force")
    _run_git(
        workspace_root,
        "-c",
        "user.name=Rova Sandbox",
        "-c",
        "user.email=rova-sandbox@local.invalid",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "--allow-empty",
        "-m",
        "Rova Sandbox Baseline",
    )
    baseline_commit_oid = _run_git(workspace_root, "rev-parse", "HEAD").strip()
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
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        process = subprocess.run(
            ["git", "-C", str(workspace_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise SandboxImportError(f"private Sandbox Git is unavailable: {error}") from error
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown Git error"
        raise SandboxImportError(f"private Sandbox Git command failed: {detail}")
    return process.stdout
