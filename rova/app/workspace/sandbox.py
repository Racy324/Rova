from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable
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


class SandboxApplyError(SandboxError):
    """A Host Apply or its durable repair record could not be completed safely."""


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


class ApplyAction(str, Enum):
    CREATE = "create"
    WRITE = "write"
    DELETE = "delete"
    SET_MODE = "set_mode"
    CREATE_SYMLINK = "create_symlink"
    CREATE_DIRECTORY = "create_directory"
    DELETE_DIRECTORY = "delete_directory"
    REPLACE_TYPE = "replace_type"


class ApplyState(str, Enum):
    APPLYING = "applying"
    APPLIED = "applied"
    RECOVERY_REQUIRED = "recovery_required"
    RESTORING = "restoring"
    RESTORED = "restored"


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
class HostConflict:
    path: str
    reason: str
    baseline: FileState | None
    host_current: FileState | None


@dataclass(frozen=True)
class ApplyOperation:
    path: str
    action: ApplyAction
    expected_baseline: FileState | None
    desired: FileState | None


@dataclass(frozen=True)
class ApplyPlan:
    sandbox_id: str
    baseline_oid: str
    generated_at: str
    changed_set: ChangedSet
    operations: tuple[ApplyOperation, ...]
    conflicts: tuple[HostConflict, ...]


@dataclass(frozen=True)
class ApplyReport:
    sandbox_id: str
    state: ApplyState | None
    applied: bool
    confirmed: bool
    operation_id: str | None
    conflicts: tuple[HostConflict, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class DiscardPlan:
    sandbox_id: str
    changed_set: ChangedSet

    @property
    def changed_path_count(self) -> int:
        return len(self.changed_set.changes)


@dataclass(frozen=True)
class DiscardReport:
    sandbox_id: str
    state: SandboxState
    discarded: bool
    confirmed: bool
    plan: DiscardPlan | None = None
    error: str | None = None


@dataclass(frozen=True)
class SandboxCleanupReport:
    sandbox_id: str
    state: SandboxState
    cleaned: bool


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

    def create_new_for_terminal_session(self, host_workspace: Workspace, session_id: str) -> SandboxMetadata:
        """Explicitly create a new B0 after this Session's prior Sandbox ended.

        This is intentionally not used by Runtime startup: replacing a baseline
        must be a user control-plane decision.
        """
        workspace_id = workspace_identity(host_workspace.root)
        pointer = self._session_pointer_path(session_id)
        if not pointer.exists():
            raise SandboxConflictError("Session has no terminal Sandbox to replace")
        previous_value = self._read_json(pointer)
        previous_id = previous_value.get("sandbox_id")
        if not isinstance(previous_id, str) or previous_value.get("workspace_id") != workspace_id:
            raise SandboxConflictError("Session Sandbox belongs to a different Host Workspace")
        previous = self.load_for_execution(previous_id)
        if previous.state not in {SandboxState.APPLIED, SandboxState.DISCARDED, SandboxState.FAILED, SandboxState.ABANDONED}:
            raise SandboxConflictError("Apply or Discard the active Sandbox before creating a new one")
        self._release_workspace_reservation(workspace_id, previous_id)
        created = self.create_unbound(host_workspace)
        try:
            imported = self.import_baseline(created.sandbox_id)
            self.bind_session(imported.sandbox_id, session_id)
            return self.mark_ready(imported.sandbox_id)
        except Exception:
            self._release_workspace_reservation(workspace_id, created.sandbox_id)
            raise

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
                old_id = existing.get("sandbox_id")
                if not isinstance(old_id, str):
                    raise SandboxConflictError("Session Sandbox pointer is invalid")
                old = self.load_for_execution(old_id)
                if old.state not in {SandboxState.APPLIED, SandboxState.DISCARDED, SandboxState.FAILED, SandboxState.ABANDONED}:
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
        if metadata.state is SandboxState.DISCARDING:
            if not metadata.sandbox_root.exists():
                metadata = replace(metadata, state=SandboxState.DISCARDED, updated_at=_utc_now())
                self._write_metadata(metadata)
            return metadata
        return self.load_for_execution(sandbox_id)

    def load_for_execution(self, sandbox_id: str) -> SandboxMetadata:
        """Resolve durable Sandbox availability without exposing physical paths to callers."""
        metadata = self._load_metadata(sandbox_id)
        if metadata.state is SandboxState.READY and (
            not metadata.sandbox_root.is_dir()
            or metadata.baseline_commit_oid is None
            or not _private_commit_exists(metadata.sandbox_root, metadata.baseline_commit_oid)
        ):
            metadata = replace(metadata, state=SandboxState.ABANDONED, updated_at=_utc_now())
            self._write_metadata(metadata)
        return metadata

    def plan_discard(self, sandbox_id: str) -> DiscardPlan:
        metadata = self._load_metadata(sandbox_id)
        if metadata.state is not SandboxState.READY:
            raise SandboxApplyError(f"Sandbox cannot discard from state {metadata.state.value}")
        if _has_unfinished_apply(metadata):
            raise SandboxApplyError("Sandbox has an unfinished Apply; resolve it before discard")
        return DiscardPlan(metadata.sandbox_id, SandboxDiffService(self).compute_changed_set(sandbox_id))

    def discard(self, sandbox_id: str, *, confirm: Callable[[DiscardPlan], bool]) -> DiscardReport:
        plan = self.plan_discard(sandbox_id)
        if not confirm(plan):
            return DiscardReport(sandbox_id, SandboxState.READY, False, False, plan)
        metadata = self._load_metadata(sandbox_id)
        discarding = replace(metadata, state=SandboxState.DISCARDING, updated_at=_utc_now())
        self._write_metadata(discarding)
        try:
            self._remove_disposable_workspace(discarding)
        except SandboxError as error:
            return DiscardReport(sandbox_id, SandboxState.DISCARDING, False, True, plan, str(error))
        if discarding.sandbox_root.exists():
            return DiscardReport(sandbox_id, SandboxState.DISCARDING, False, True, plan, "Sandbox workspace cleanup is incomplete")
        discarded = replace(discarding, state=SandboxState.DISCARDED, updated_at=_utc_now())
        self._write_metadata(discarded)
        return DiscardReport(sandbox_id, SandboxState.DISCARDED, True, True, plan)

    def cleanup_terminal(self, sandbox_id: str) -> SandboxCleanupReport:
        metadata = self._load_metadata(sandbox_id)
        if metadata.state not in {SandboxState.APPLIED, SandboxState.DISCARDED, SandboxState.FAILED, SandboxState.ABANDONED}:
            raise SandboxApplyError("cleanup refuses an active Sandbox")
        if _has_unfinished_apply(metadata):
            raise SandboxApplyError("cleanup refuses Sandbox with unfinished Apply evidence")
        self._remove_disposable_workspace(metadata)
        return SandboxCleanupReport(metadata.sandbox_id, metadata.state, not metadata.sandbox_root.exists())

    def _remove_disposable_workspace(self, metadata: SandboxMetadata) -> None:
        expected_root = self.root / metadata.sandbox_id / "workspace"
        try:
            if metadata.sandbox_root.resolve(strict=False) != expected_root.resolve(strict=False):
                raise SandboxError("Sandbox workspace metadata is not owned by this store")
            workspace_stat = metadata.sandbox_root.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise SandboxError(f"could not inspect Sandbox workspace for cleanup: {error}") from error
        if stat.S_ISLNK(workspace_stat.st_mode) or not stat.S_ISDIR(workspace_stat.st_mode):
            raise SandboxError("Sandbox workspace is not a removable owned directory")

        def clear_readonly(function, failed_path, _exception) -> None:
            try:
                os.chmod(failed_path, stat.S_IWRITE)
                function(failed_path)
            except OSError as error:
                raise SandboxError(f"could not remove Sandbox workspace: {error}") from error

        try:
            shutil.rmtree(metadata.sandbox_root, onerror=clear_readonly)
        except OSError as error:
            raise SandboxError(f"could not remove Sandbox workspace: {error}") from error

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


class SandboxApplyService:
    """The only Phase-6 boundary permitted to mutate a Sandbox Host Workspace.

    This service is deliberately not an Agent tool.  It treats B0, the Sandbox
    ChangedSet and the current Host state as three independent authorities and
    fails the whole Apply before any Host write if one changed path has drifted.
    """

    def __init__(self, store: SandboxStore, *, limits: SandboxDiffLimits | None = None) -> None:
        self._store = store
        self._limits = limits or SandboxDiffLimits()
        self._diff = SandboxDiffService(store, limits=self._limits)
        self._active_operation_id: str | None = None
        self._active_metadata: SandboxMetadata | None = None

    def build_plan(self, sandbox_id: str) -> ApplyPlan:
        metadata = self._store._load_metadata(sandbox_id)
        if metadata.state is not SandboxState.READY:
            raise SandboxApplyError(f"Sandbox cannot Apply from state {metadata.state.value}")
        if _has_unfinished_apply(metadata):
            raise SandboxApplyError("Sandbox has an unfinished Apply; restore its preimages before another Apply")
        changed_set = self._diff.compute_changed_set(sandbox_id)
        baseline = _scan_baseline_tree(metadata)
        operations = _ordered_apply_operations(changed_set.changes)
        conflicts = self._find_host_conflicts(metadata, baseline, operations)
        return ApplyPlan(
            sandbox_id=metadata.sandbox_id,
            baseline_oid=changed_set.baseline_oid,
            generated_at=_utc_now(),
            changed_set=changed_set,
            operations=operations,
            conflicts=conflicts,
        )

    def apply(self, sandbox_id: str, *, confirm: Callable[[ApplyPlan], bool]) -> ApplyReport:
        plan = self.build_plan(sandbox_id)
        if plan.conflicts:
            return ApplyReport(sandbox_id, None, False, False, None, plan.conflicts)
        if not confirm(plan):
            return ApplyReport(sandbox_id, None, False, False, None)

        metadata = self._store._load_metadata(sandbox_id)
        # Confirmation cannot make a stale plan safe.  Re-check before durable
        # preparation, then re-check each exact operation before mutation.
        baseline = _scan_baseline_tree(metadata)
        conflicts = self._find_host_conflicts(metadata, baseline, plan.operations)
        if conflicts:
            return ApplyReport(sandbox_id, None, False, True, None, conflicts)

        operation_id = uuid4().hex
        self._active_operation_id = operation_id
        self._active_metadata = metadata
        try:
            backup_ref = self._create_preimage_backup(metadata, operation_id, plan.operations)
            manifest = self._new_manifest(operation_id, plan, backup_ref)
            self._write_manifest(metadata, operation_id, manifest)
            self._store._write_metadata(
                replace(metadata, state=SandboxState.APPLYING, active_apply_id=operation_id, updated_at=_utc_now())
            )

            for index, operation in enumerate(plan.operations):
                conflict = self._revalidate_operation(operation, metadata.host_root)
                if conflict is not None:
                    self._set_manifest_state(metadata, operation_id, ApplyState.RECOVERY_REQUIRED, index, str(conflict))
                    return ApplyReport(sandbox_id, ApplyState.RECOVERY_REQUIRED, False, True, operation_id, (conflict,))
                try:
                    self._mutate_operation(operation, metadata.host_root)
                except (OSError, SandboxError) as error:
                    self._set_manifest_state(metadata, operation_id, ApplyState.RECOVERY_REQUIRED, index, str(error))
                    return ApplyReport(sandbox_id, ApplyState.RECOVERY_REQUIRED, False, True, operation_id, error=str(error))
                self._mark_operation_completed(metadata, operation_id, index)

            self._set_manifest_state(metadata, operation_id, ApplyState.APPLIED, None, None)
            self._store._write_metadata(
                replace(metadata, state=SandboxState.APPLIED, active_apply_id=operation_id, updated_at=_utc_now())
            )
            return ApplyReport(sandbox_id, ApplyState.APPLIED, True, True, operation_id)
        finally:
            self._active_operation_id = None
            self._active_metadata = None

    def restore_preimages(
        self,
        sandbox_id: str,
        operation_id: str,
        *,
        confirm: Callable[[ApplyReport], bool],
    ) -> ApplyReport:
        metadata = self._store._load_metadata(sandbox_id)
        manifest = self._read_manifest(metadata, operation_id)
        state = _manifest_state(manifest)
        if state is ApplyState.RESTORED:
            return ApplyReport(sandbox_id, ApplyState.RESTORED, False, True, operation_id)
        if state not in {ApplyState.APPLYING, ApplyState.RECOVERY_REQUIRED, ApplyState.RESTORING}:
            raise SandboxApplyError("Apply preimages are only available for an unfinished Apply")
        report = ApplyReport(sandbox_id, state, False, False, operation_id)
        if not confirm(report):
            return report

        try:
            preimages = self._read_backup(metadata, operation_id)
            restore_entries, conflicts = self._validate_restore_preimages(manifest, preimages, metadata.host_root)
            if conflicts:
                self._set_manifest_state(metadata, operation_id, ApplyState.RECOVERY_REQUIRED, None, "Host changed after Apply")
                return ApplyReport(sandbox_id, ApplyState.RECOVERY_REQUIRED, False, True, operation_id, conflicts)
            self._set_manifest_state(metadata, operation_id, ApplyState.RESTORING, None, None)
            for preimage in _ordered_preimages_for_restore(restore_entries):
                self._restore_preimage(preimage, metadata.host_root, metadata, operation_id)
        except (OSError, SandboxError) as error:
            self._set_manifest_state(metadata, operation_id, ApplyState.RECOVERY_REQUIRED, None, str(error))
            return ApplyReport(sandbox_id, ApplyState.RECOVERY_REQUIRED, False, True, operation_id, error=str(error))

        self._set_manifest_state(metadata, operation_id, ApplyState.RESTORED, None, None)
        self._store._write_metadata(
            replace(metadata, state=SandboxState.READY, active_apply_id=None, updated_at=_utc_now())
        )
        return ApplyReport(sandbox_id, ApplyState.RESTORED, False, True, operation_id)

    def _find_host_conflicts(
        self,
        metadata: SandboxMetadata,
        baseline: dict[str, FileState],
        operations: tuple[ApplyOperation, ...],
    ) -> tuple[HostConflict, ...]:
        conflicts: dict[str, HostConflict] = {}
        for operation in operations:
            ancestor = _first_unsafe_host_ancestor(
                metadata.host_root,
                baseline,
                operation.path,
                operations,
                self._limits,
            )
            if ancestor is not None:
                conflicts.setdefault(ancestor.path, ancestor)
                continue
            host_state = _file_state_at(metadata.host_root, operation.path, self._limits)
            if host_state != operation.expected_baseline:
                conflicts[operation.path] = HostConflict(
                    operation.path,
                    "host_state_differs_from_baseline",
                    operation.expected_baseline,
                    host_state,
                )
            if _operation_replaces_or_deletes_directory(operation):
                conflict = _subtree_conflict(metadata.host_root, baseline, operation.path, self._limits)
                if conflict is not None:
                    conflicts.setdefault(conflict.path, conflict)
        return tuple(conflicts[path] for path in sorted(conflicts))

    def _create_preimage_backup(
        self,
        metadata: SandboxMetadata,
        operation_id: str,
        operations: tuple[ApplyOperation, ...],
    ) -> str:
        backup_root = metadata.sandbox_root.parent / "apply-backups" / operation_id
        if backup_root.exists():
            raise SandboxApplyError("Apply backup id already exists")
        entries: list[dict[str, object]] = []
        for index, operation in enumerate(operations):
            state = _file_state_at(metadata.host_root, operation.path, self._limits)
            if state != operation.expected_baseline:
                raise SandboxApplyError(f"Host drifted before Apply backup: {operation.path}")
            entry: dict[str, object] = {"path": operation.path, "state": _file_state_to_json(state)}
            if state is not None and state.kind is FileStateKind.REGULAR:
                content = _read_regular_bytes_at(metadata.host_root, operation.path, self._limits)
                relative = f"files/{index:08d}.bin"
                _write_durable_bytes(backup_root / relative, content)
                entry["content_ref"] = relative
            entries.append(entry)
        SandboxStore._write_json(
            backup_root / "manifest.json",
            {"version": 1, "operation_id": operation_id, "created_at": _utc_now(), "entries": entries},
        )
        return str(backup_root.relative_to(metadata.sandbox_root.parent).as_posix())

    def _new_manifest(self, operation_id: str, plan: ApplyPlan, backup_ref: str) -> dict[str, object]:
        return {
            "version": 1,
            "operation_id": operation_id,
            "sandbox_id": plan.sandbox_id,
            "baseline_oid": plan.baseline_oid,
            "state": ApplyState.APPLYING.value,
            "started_at": _utc_now(),
            "backup_ref": backup_ref,
            "operations": [
                {
                    "path": operation.path,
                    "action": operation.action.value,
                    "expected_baseline": _file_state_to_json(operation.expected_baseline),
                    "desired": _file_state_to_json(operation.desired),
                    "state": "pending",
                }
                for operation in plan.operations
            ],
        }

    def _write_manifest(self, metadata: SandboxMetadata, operation_id: str, manifest: dict[str, object]) -> None:
        SandboxStore._write_json(_manifest_path(metadata, operation_id), manifest)

    def _read_manifest(self, metadata: SandboxMetadata, operation_id: str) -> dict:
        value = SandboxStore._read_json(_manifest_path(metadata, operation_id))
        if value.get("operation_id") != operation_id or value.get("sandbox_id") != metadata.sandbox_id:
            raise SandboxApplyError("Apply manifest does not belong to this Sandbox")
        return value

    def _set_manifest_state(
        self,
        metadata: SandboxMetadata,
        operation_id: str,
        state: ApplyState,
        failed_index: int | None,
        error: str | None,
    ) -> None:
        manifest = self._read_manifest(metadata, operation_id)
        manifest["state"] = state.value
        manifest["updated_at"] = _utc_now()
        if failed_index is not None:
            operations = manifest.get("operations")
            if isinstance(operations, list) and 0 <= failed_index < len(operations) and isinstance(operations[failed_index], dict):
                operations[failed_index]["state"] = "failed"
        if error is not None:
            manifest["error"] = error
        self._write_manifest(metadata, operation_id, manifest)

    def _mark_operation_completed(self, metadata: SandboxMetadata, operation_id: str, index: int) -> None:
        manifest = self._read_manifest(metadata, operation_id)
        operations = manifest.get("operations")
        if not isinstance(operations, list) or not isinstance(operations[index], dict):
            raise SandboxApplyError("Apply manifest operations are invalid")
        operations[index]["state"] = "completed"
        operations[index]["completed_at"] = _utc_now()
        self._write_manifest(metadata, operation_id, manifest)

    def _revalidate_operation(self, operation: ApplyOperation, host_root: Path) -> HostConflict | None:
        try:
            _ensure_safe_parent(host_root, _host_target(host_root, operation.path))
            current = _file_state_at(host_root, operation.path, self._limits)
        except SandboxApplyError:
            return HostConflict(operation.path, "host_parent_is_not_safe", operation.expected_baseline, None)
        if current != operation.expected_baseline:
            return HostConflict(operation.path, "host_state_changed_before_mutation", operation.expected_baseline, current)
        return None

    def _mutate_operation(self, operation: ApplyOperation, host_root: Path) -> None:
        target = _host_target(host_root, operation.path)
        _ensure_safe_parent(host_root, target)
        if operation.action is ApplyAction.CREATE_DIRECTORY:
            target.mkdir()
            return
        if operation.action is ApplyAction.DELETE_DIRECTORY:
            _remove_existing_path(target, directory_only=True)
            return
        if operation.action is ApplyAction.DELETE:
            _remove_existing_path(target, directory_only=False)
            return
        if operation.action is ApplyAction.SET_MODE:
            if operation.desired is None or operation.desired.kind is not FileStateKind.REGULAR:
                raise SandboxApplyError("mode operation has no regular-file destination")
            _copy_mode(target, 0o755 if operation.desired.executable else 0o644)
            return
        if operation.action is ApplyAction.REPLACE_TYPE:
            _remove_existing_path(target, directory_only=False)
        elif operation.action is ApplyAction.CREATE_SYMLINK and operation.expected_baseline is not None:
            _remove_existing_path(target, directory_only=False)
        if operation.desired is None:
            raise SandboxApplyError("Apply operation has no destination state")
        metadata = self._active_metadata
        if metadata is None:
            raise SandboxApplyError("Apply mutation is not active")
        if operation.desired.kind is FileStateKind.DIRECTORY:
            target.mkdir()
        elif operation.desired.kind is FileStateKind.SYMLINK:
            _create_symlink_from_sandbox(metadata.sandbox_root, operation.path, target, operation.desired)
        else:
            content = _read_regular_bytes_at(metadata.sandbox_root, operation.path, self._limits)
            _write_regular_atomically(target, content, operation.desired)

    def _read_backup(self, metadata: SandboxMetadata, operation_id: str) -> list[dict]:
        manifest = SandboxStore._read_json(metadata.sandbox_root.parent / "apply-backups" / operation_id / "manifest.json")
        if manifest.get("operation_id") != operation_id:
            raise SandboxApplyError("Apply preimage manifest is invalid")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise SandboxApplyError("Apply preimage entries are invalid")
        return [entry for entry in entries if isinstance(entry, dict)]

    def _validate_restore_preimages(
        self,
        manifest: dict,
        preimages: list[dict],
        host_root: Path,
    ) -> tuple[list[dict], tuple[HostConflict, ...]]:
        operations = manifest.get("operations")
        if not isinstance(operations, list):
            raise SandboxApplyError("Apply manifest operations are invalid")
        desired_by_path: dict[str, FileState | None] = {}
        for operation in operations:
            if not isinstance(operation, dict):
                raise SandboxApplyError("Apply manifest operations are invalid")
            desired_by_path[_logical_path(operation.get("path"))] = _file_state_from_json(operation.get("desired"))

        restore_entries: list[dict] = []
        conflicts: list[HostConflict] = []
        for preimage in preimages:
            path = _logical_path(preimage.get("path"))
            if path not in desired_by_path:
                raise SandboxApplyError("Apply backup does not match its manifest")
            before_apply = _file_state_from_json(preimage.get("state"))
            desired = desired_by_path[path]
            current = _file_state_at(host_root, path, self._limits)
            if current == before_apply:
                continue
            if current == desired:
                restore_entries.append(preimage)
                continue
            conflicts.append(
                HostConflict(path, "host_state_changed_after_apply", before_apply, current)
            )
        return restore_entries, tuple(conflicts)

    def _restore_preimage(
        self,
        preimage: dict,
        host_root: Path,
        metadata: SandboxMetadata,
        operation_id: str,
    ) -> None:
        path = _logical_path(preimage.get("path"))
        state = _file_state_from_json(preimage.get("state"))
        target = _host_target(host_root, path)
        _ensure_safe_parent(host_root, target)
        if state is None:
            _remove_existing_path(target, directory_only=False)
            return
        if state.kind is FileStateKind.DIRECTORY:
            current = _file_state_at(host_root, path, self._limits)
            if current is None:
                target.mkdir()
            elif current.kind is not FileStateKind.DIRECTORY:
                _remove_existing_path(target, directory_only=False)
                target.mkdir()
            return
        _remove_existing_path(target, directory_only=False)
        if state.kind is FileStateKind.SYMLINK:
            if state.symlink_target is None:
                raise SandboxApplyError("symlink preimage has no target")
            os.symlink(state.symlink_target, target)
            return
        content_ref = preimage.get("content_ref")
        if not isinstance(content_ref, str):
            raise SandboxApplyError("regular-file preimage has no content")
        content_path = metadata.sandbox_root.parent / "apply-backups" / operation_id / content_ref
        _ensure_within(content_path, metadata.sandbox_root.parent / "apply-backups" / operation_id)
        _write_regular_atomically(target, content_path.read_bytes(), state)


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


def _ordered_apply_operations(changes: tuple[PathChange, ...]) -> tuple[ApplyOperation, ...]:
    operations = tuple(_apply_operation_for(change) for change in changes)

    def sort_key(operation: ApplyOperation) -> tuple[int, int, str]:
        depth = operation.path.count("/") + 1
        desired_directory = operation.desired is not None and operation.desired.kind is FileStateKind.DIRECTORY
        replaces_directory = _operation_replaces_or_deletes_directory(operation)
        if desired_directory:
            return (0, depth, operation.path)
        if operation.action is ApplyAction.DELETE and not replaces_directory:
            return (2, -depth, operation.path)
        if replaces_directory:
            return (3, -depth, operation.path)
        if operation.action is ApplyAction.DELETE_DIRECTORY:
            return (4, -depth, operation.path)
        return (1, depth, operation.path)

    return tuple(sorted(operations, key=sort_key))


def _apply_operation_for(change: PathChange) -> ApplyOperation:
    if change.kind is ChangeKind.ADDED:
        assert change.current is not None
        action = {
            FileStateKind.DIRECTORY: ApplyAction.CREATE_DIRECTORY,
            FileStateKind.SYMLINK: ApplyAction.CREATE_SYMLINK,
            FileStateKind.REGULAR: ApplyAction.CREATE,
        }[change.current.kind]
    elif change.kind is ChangeKind.DELETED:
        assert change.baseline is not None
        action = ApplyAction.DELETE_DIRECTORY if change.baseline.kind is FileStateKind.DIRECTORY else ApplyAction.DELETE
    elif change.kind is ChangeKind.MODE_CHANGED:
        action = ApplyAction.SET_MODE
    elif change.kind is ChangeKind.SYMLINK_CHANGED:
        action = ApplyAction.CREATE_SYMLINK
    elif change.kind is ChangeKind.TYPE_CHANGED:
        action = ApplyAction.REPLACE_TYPE
    else:
        action = ApplyAction.WRITE
    return ApplyOperation(change.path, action, change.baseline, change.current)


def _operation_replaces_or_deletes_directory(operation: ApplyOperation) -> bool:
    return (
        operation.expected_baseline is not None
        and operation.expected_baseline.kind is FileStateKind.DIRECTORY
        and (operation.desired is None or operation.desired.kind is not FileStateKind.DIRECTORY)
    )


def _file_state_to_json(state_value: FileState | None) -> dict[str, object] | None:
    if state_value is None:
        return None
    return {
        "kind": state_value.kind.value,
        "content_digest": state_value.content_digest,
        "size_bytes": state_value.size_bytes,
        "executable": state_value.executable,
        "symlink_target": state_value.symlink_target,
        "binary": state_value.binary,
    }


def _file_state_from_json(value: object) -> FileState | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SandboxApplyError("stored FileState is invalid")
    try:
        return FileState(
            kind=FileStateKind(value["kind"]),
            content_digest=value.get("content_digest"),
            size_bytes=value.get("size_bytes"),
            executable=value.get("executable"),
            symlink_target=value.get("symlink_target"),
            binary=bool(value.get("binary", False)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SandboxApplyError("stored FileState is invalid") from error


def _manifest_path(metadata: SandboxMetadata, operation_id: str) -> Path:
    if not isinstance(operation_id, str) or not operation_id or any(char not in "0123456789abcdef" for char in operation_id):
        raise SandboxApplyError("Apply operation id is invalid")
    return metadata.sandbox_root.parent / "apply" / f"{operation_id}.json"


def _manifest_state(manifest: dict) -> ApplyState:
    try:
        return ApplyState(manifest["state"])
    except (KeyError, TypeError, ValueError) as error:
        raise SandboxApplyError("Apply manifest state is invalid") from error


def _has_unfinished_apply(metadata: SandboxMetadata) -> bool:
    apply_root = metadata.sandbox_root.parent / "apply"
    if not apply_root.is_dir():
        return False
    for manifest_path in apply_root.glob("*.json"):
        manifest = SandboxStore._read_json(manifest_path)
        if manifest.get("sandbox_id") != metadata.sandbox_id:
            continue
        if _manifest_state(manifest) in {ApplyState.APPLYING, ApplyState.RECOVERY_REQUIRED, ApplyState.RESTORING}:
            return True
    return False


def _host_target(root: Path, path: str) -> Path:
    logical = _logical_path(path)
    target = Path(root).joinpath(*logical.split("/"))
    _ensure_within(target, Path(root))
    return target


def _ensure_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SandboxApplyError("path escapes its managed root") from error


def _file_state_at(root: Path, path: str, limits: SandboxDiffLimits) -> FileState | None:
    target = _host_target(root, path)
    try:
        entry_stat = _lstat_without_following_parents(root, target)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SandboxApplyError(f"could not inspect managed path {path}: {error}") from error
    if stat.S_ISLNK(entry_stat.st_mode):
        target_value = os.readlink(target)
        target_bytes = os.fsencode(target_value)
        return FileState(FileStateKind.SYMLINK, _sha256(target_bytes), len(target_bytes), None, target_value)
    if stat.S_ISDIR(entry_stat.st_mode):
        return FileState(FileStateKind.DIRECTORY, None, None, None, None)
    if stat.S_ISREG(entry_stat.st_mode):
        digest, size_bytes, binary = _regular_file_facts(target, limits.max_single_file_bytes)
        return FileState(
            FileStateKind.REGULAR,
            digest,
            size_bytes,
            None if os.name == "nt" else bool(entry_stat.st_mode & stat.S_IXUSR),
            None,
            binary,
        )
    raise SandboxApplyError(f"unsupported managed path type: {path}")


def _read_regular_bytes_at(root: Path, path: str, limits: SandboxDiffLimits) -> bytes:
    target = _host_target(root, path)
    try:
        before = _lstat_without_following_parents(root, target)
    except OSError as error:
        raise SandboxApplyError(f"could not inspect regular file {path}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise SandboxApplyError(f"managed path is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
        try:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_single_file_bytes:
                    raise SandboxApplyError("regular file exceeds Sandbox Apply resource limit")
                chunks.append(chunk)
            content = b"".join(chunks)
        finally:
            os.close(descriptor)
        after = _lstat_without_following_parents(root, target)
    except SandboxApplyError:
        raise
    except OSError as error:
        raise SandboxApplyError(f"could not read regular file {path}: {error}") from error
    if not stat.S_ISREG(after.st_mode) or after.st_size != before.st_size:
        raise SandboxApplyError(f"regular file changed while it was read: {path}")
    return content


def _lstat_without_following_parents(root: Path, target: Path) -> os.stat_result:
    """Inspect a logical child without traversing an intermediate symlink."""
    root = Path(root)
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise SandboxApplyError(f"managed root is unavailable: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise SandboxApplyError("managed root is not a safe directory")
    parent = root
    for part in target.relative_to(root).parts[:-1]:
        parent = parent / part
        try:
            parent_stat = parent.lstat()
        except FileNotFoundError:
            raise
        except OSError as error:
            raise SandboxApplyError(f"could not inspect managed parent: {error}") from error
        if stat.S_ISLNK(parent_stat.st_mode):
            raise SandboxApplyError("managed parent is a symlink")
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise FileNotFoundError(str(parent))
    return target.lstat()


def _first_unsafe_host_ancestor(
    host_root: Path,
    baseline: dict[str, FileState],
    path: str,
    operations: tuple[ApplyOperation, ...],
    limits: SandboxDiffLimits,
) -> HostConflict | None:
    planned_directories = {
        operation.path
        for operation in operations
        if operation.desired is not None and operation.desired.kind is FileStateKind.DIRECTORY
    }
    parts = path.split("/")[:-1]
    for index in range(1, len(parts) + 1):
        ancestor_path = "/".join(parts[:index])
        expected = baseline.get(ancestor_path)
        current = _file_state_at(host_root, ancestor_path, limits)
        if ancestor_path in planned_directories:
            # This ancestor is itself a validated operation.  Its B0 state may
            # be missing or non-directory; the ordered plan creates the
            # directory before any descendant operation.
            continue
        if expected is not None and expected.kind is FileStateKind.DIRECTORY:
            if current != expected:
                return HostConflict(ancestor_path, "host_ancestor_differs_from_baseline", expected, current)
        elif current is not None and current.kind is not FileStateKind.DIRECTORY:
            return HostConflict(ancestor_path, "host_ancestor_is_not_directory", expected, current)
    return None


def _subtree_conflict(
    host_root: Path,
    baseline: dict[str, FileState],
    root_path: str,
    limits: SandboxDiffLimits,
) -> HostConflict | None:
    host_root_state = _file_state_at(host_root, root_path, limits)
    expected_root = baseline.get(root_path)
    if host_root_state != expected_root:
        return HostConflict(root_path, "host_subtree_root_differs_from_baseline", expected_root, host_root_state)
    if host_root_state is None or host_root_state.kind is not FileStateKind.DIRECTORY:
        return None
    expected = {
        path: state for path, state in baseline.items() if path == root_path or path.startswith(root_path + "/")
    }
    current = _scan_subtree(host_root, root_path, limits)
    if current != expected:
        return HostConflict(root_path, "host_subtree_differs_from_baseline", expected_root, host_root_state)
    return None


def _scan_subtree(root: Path, root_path: str, limits: SandboxDiffLimits) -> dict[str, FileState]:
    states: dict[str, FileState] = {}
    file_count = 0
    total_bytes = 0

    def visit(path: str) -> None:
        nonlocal file_count, total_bytes
        state_value = _file_state_at(root, path, limits)
        if state_value is None:
            return
        file_count += 1
        if file_count > limits.max_files:
            raise SandboxApplyError("Host drift scan exceeded max_files")
        if state_value.size_bytes is not None:
            total_bytes += state_value.size_bytes
            if total_bytes > limits.max_total_bytes:
                raise SandboxApplyError("Host drift scan exceeded max_total_bytes")
        states[path] = state_value
        if state_value.kind is FileStateKind.DIRECTORY:
            directory = _host_target(root, path)
            try:
                entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
            except OSError as error:
                raise SandboxApplyError(f"could not scan Host subtree {path}: {error}") from error
            for entry in entries:
                visit(f"{path}/{entry.name}")

    visit(root_path)
    return states


def _ensure_safe_parent(root: Path, target: Path) -> None:
    root = Path(root)
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise SandboxApplyError(f"Host Workspace is unavailable: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise SandboxApplyError("Host Workspace root is not a safe directory")
    relative_parts = target.relative_to(root).parts[:-1]
    parent = root
    for part in relative_parts:
        parent = parent / part
        try:
            parent_stat = parent.lstat()
        except FileNotFoundError as error:
            raise SandboxApplyError(f"Host parent directory is missing: {parent}") from error
        except OSError as error:
            raise SandboxApplyError(f"could not inspect Host parent directory: {error}") from error
        if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
            raise SandboxApplyError("Host parent path is not a safe directory")


def _remove_existing_path(target: Path, *, directory_only: bool) -> None:
    try:
        state_value = target.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(state_value.st_mode) and not stat.S_ISLNK(state_value.st_mode):
        target.rmdir()  # Intentionally never recursive: unknown Host content must survive.
        return
    if directory_only:
        raise SandboxApplyError("expected an empty Host directory")
    if stat.S_ISLNK(state_value.st_mode) or stat.S_ISREG(state_value.st_mode):
        target.unlink()
        return
    raise SandboxApplyError("unsupported Host path type for removal")


def _write_durable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise SandboxApplyError(f"could not persist Apply preimage: {error}") from error


def _write_regular_atomically(target: Path, content: bytes, state_value: FileState) -> None:
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if state_value.executable is not None:
            _copy_mode(temporary, 0o755 if state_value.executable else 0o644)
        temporary.replace(target)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise SandboxApplyError(f"could not atomically write Host file: {error}") from error


def _create_symlink_from_sandbox(sandbox_root: Path, path: str, target: Path, desired: FileState) -> None:
    source = _host_target(sandbox_root, path)
    try:
        source_stat = source.lstat()
    except OSError as error:
        raise SandboxApplyError(f"could not inspect Sandbox symlink: {error}") from error
    if not stat.S_ISLNK(source_stat.st_mode):
        raise SandboxApplyError("Sandbox desired symlink changed after confirmation")
    link_target = os.readlink(source)
    if desired.symlink_target != link_target:
        raise SandboxApplyError("Sandbox desired symlink changed after confirmation")
    os.symlink(link_target, target)


def _ordered_preimages_for_restore(entries: list[dict]) -> list[dict]:
    def key(entry: dict) -> tuple[int, int, str]:
        path = _logical_path(entry.get("path"))
        state_value = _file_state_from_json(entry.get("state"))
        depth = path.count("/") + 1
        if state_value is not None and state_value.kind is FileStateKind.DIRECTORY:
            return (0, depth, path)
        if state_value is None:
            return (2, -depth, path)
        return (1, depth, path)
    return sorted(entries, key=key)


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
