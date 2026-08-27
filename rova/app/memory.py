from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .file_lock import FileLock, FileLockError
from .paths import RovaDataPaths


class MemoryDocumentAction(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class MemoryStoreError(RuntimeError):
    """Expected local-memory storage failure; callers may degrade without stopping a run."""


@dataclass(frozen=True)
class MemorySnapshot:
    user_markdown: str = ""
    memory_markdown: str = ""

@dataclass(frozen=True)
class MemoryDocumentUpdate:
    action: MemoryDocumentAction
    markdown: str = ""


@dataclass(frozen=True)
class MemoryUpdate:
    user: MemoryDocumentUpdate
    memory: MemoryDocumentUpdate


@dataclass(frozen=True)
class MemoryApplyResult:
    snapshot: MemorySnapshot
    changed_documents: tuple[str, ...]


MemoryUpdateFactory = Callable[[MemorySnapshot], Awaitable[MemoryUpdate]]


class MemoryStore(Protocol):
    def load_snapshot(self) -> MemorySnapshot: ...

    async def update(self, factory: MemoryUpdateFactory, *, max_chars: int) -> MemoryApplyResult: ...


class FileMemoryStore:
    """Small local Markdown store for cross-session user memory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = RovaDataPaths.resolve().memory if root is None else Path(root)

    def load_snapshot(self) -> MemorySnapshot:
        self._ensure_root()
        try:
            with FileLock(self.root / ".memory.lock"):
                return self._load_snapshot_unlocked()
        except FileLockError as error:
            raise MemoryStoreError("could not acquire memory file lock") from error

    async def update(self, factory: MemoryUpdateFactory, *, max_chars: int) -> MemoryApplyResult:
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            raise ValueError("maximum length must be a positive integer")
        self._ensure_root()
        lock = FileLock(self.root / ".memory.lock")
        try:
            await asyncio.to_thread(lock.acquire)
        except FileLockError as error:
            raise MemoryStoreError("could not acquire memory file lock") from error
        try:
            current = self._load_snapshot_unlocked()
            update = await factory(current)
            if not isinstance(update, MemoryUpdate):
                raise TypeError("memory update factory must return MemoryUpdate")
            target, changed = _apply_update(current, update)
            _validate_snapshot_length(target, max_chars, changed)
            self._write_snapshot_unlocked(current, target, changed)
            return MemoryApplyResult(target, changed)
        finally:
            await asyncio.to_thread(lock.release)

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MemoryStoreError("could not create memory directory") from error
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def _load_snapshot_unlocked(self) -> MemorySnapshot:
        return MemorySnapshot(
            user_markdown=_read_markdown(self.root / "USER.md"),
            memory_markdown=_read_markdown(self.root / "MEMORY.md"),
        )

    def _write_snapshot_unlocked(
        self,
        current: MemorySnapshot,
        target: MemorySnapshot,
        changed_documents: tuple[str, ...],
    ) -> None:
        pending: list[tuple[Path, str]] = []
        if "USER.md" in changed_documents:
            pending.append((self.root / "USER.md", target.user_markdown))
        if "MEMORY.md" in changed_documents:
            pending.append((self.root / "MEMORY.md", target.memory_markdown))
        temporary_paths: list[tuple[Path, Path, str]] = []
        try:
            for path, content in pending:
                if content:
                    temporary = self.root / f".{path.name}.{uuid4().hex}.tmp"
                    _write_temporary(temporary, content)
                    temporary_paths.append((temporary, path, content))
            for temporary, path, _content in temporary_paths:
                os.replace(temporary, path)
            for path, content in pending:
                if not content:
                    path.unlink(missing_ok=True)
        except OSError as error:
            raise MemoryStoreError("failed to persist memory") from error
        finally:
            for temporary, _path, _content in temporary_paths:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def _apply_update(current: MemorySnapshot, update: MemoryUpdate) -> tuple[MemorySnapshot, tuple[str, ...]]:
    user = _apply_document_update(current.user_markdown, update.user)
    memory = _apply_document_update(current.memory_markdown, update.memory)
    target = MemorySnapshot(user, memory)
    changed: list[str] = []
    if target.user_markdown != current.user_markdown:
        changed.append("USER.md")
    if target.memory_markdown != current.memory_markdown:
        changed.append("MEMORY.md")
    return target, tuple(changed)


def _apply_document_update(current: str, update: MemoryDocumentUpdate) -> str:
    if update.action is MemoryDocumentAction.NOOP:
        return current
    if update.action is MemoryDocumentAction.DELETE:
        return ""
    if update.action not in {MemoryDocumentAction.ADD, MemoryDocumentAction.UPDATE}:
        raise ValueError(f"unsupported memory document action: {update.action!r}")
    replacement = _normalize_markdown(update.markdown)
    if not replacement:
        raise ValueError("ADD or UPDATE memory action requires Markdown content")
    return replacement


def _validate_snapshot_length(
    snapshot: MemorySnapshot,
    max_chars: int,
    changed_documents: tuple[str, ...],
) -> None:
    for filename, content in (("USER.md", snapshot.user_markdown), ("MEMORY.md", snapshot.memory_markdown)):
        if filename not in changed_documents:
            continue
        if len(content) > max_chars:
            raise ValueError(f"{filename} exceeds maximum length of {max_chars} characters")


def _read_markdown(path: Path) -> str:
    try:
        return _normalize_markdown(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return ""


def _normalize_markdown(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _write_temporary(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content + "\n")
        handle.flush()
        os.fsync(handle.fileno())
