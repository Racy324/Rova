from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rova.ai.messages import Message
from rova.app.paths import RovaDataPaths

from .context_builder import SessionProjectionError, build_session_projection
from .entries import CompactionEntry, MessageEntry, SessionEntry
from .serialization import MessageSerializationError, message_from_dict, message_to_dict


class SessionStoreError(RuntimeError):
    pass


class SessionCorruptionError(SessionStoreError):
    def __init__(self, session_id: str, line_number: int, reason: str) -> None:
        super().__init__(f"session {session_id!r} is corrupt at line {line_number}: {reason}")
        self.session_id = session_id
        self.line_number = line_number


@dataclass(frozen=True)
class SessionSummary:
    """Read-only metadata suitable for a local session picker."""

    session_id: str
    created_at: str
    updated_at: str
    first_user_preview: str | None


@dataclass
class DurableSession:
    store: JsonlSessionStore
    session_id: str
    entries: list[SessionEntry] = field(default_factory=list)
    by_id: dict[str, SessionEntry] = field(default_factory=dict)
    leaf_id: str | None = None

    @property
    def messages(self) -> list[Message]:
        """All physical entries in append order; intended for audit, not branch context."""
        return self.physical_messages

    @property
    def physical_messages(self) -> list[Message]:
        """Physical MessageEntry values only; this is not a logical projection."""
        return [entry.message for entry in self.entries if isinstance(entry, MessageEntry)]

    @property
    def entry_ids(self) -> list[str]:
        return [entry.entry_id for entry in self.entries]

    def append(self, message: Message) -> str:
        parent_id = self.leaf_id
        entry_id = self.store.append_message(self.session_id, parent_id, message)
        entry = MessageEntry(entry_id, parent_id, message)
        self.entries.append(entry)
        self.by_id[entry_id] = entry
        self.leaf_id = entry_id
        return entry_id

    def append_compaction(self, summary: str, first_kept_entry_id: str | None) -> CompactionEntry:
        parent_id = self.leaf_id
        if parent_id is None:
            raise SessionStoreError("compaction parent_id must reference the current leaf")
        if not isinstance(summary, str) or not summary:
            raise SessionStoreError("compaction summary must be a non-empty string")
        self._validate_first_kept(parent_id, first_kept_entry_id)
        entry_id = self.store.append_compaction(self.session_id, parent_id, summary, first_kept_entry_id)
        entry = CompactionEntry(entry_id, parent_id, summary, first_kept_entry_id)
        self.entries.append(entry)
        self.by_id[entry_id] = entry
        self.leaf_id = entry_id
        return entry

    def branch(self, entry_id: str) -> None:
        self.path_to_leaf(entry_id)
        self.leaf_id = entry_id

    def path_to_leaf(self, leaf_id: str | None = None) -> list[SessionEntry]:
        target_id = self.leaf_id if leaf_id is None else leaf_id
        if target_id is None:
            return []
        path: list[SessionEntry] = []
        while target_id is not None:
            entry = self.by_id.get(target_id)
            if entry is None:
                raise SessionStoreError(f"unknown session entry: {target_id}")
            path.append(entry)
            target_id = entry.parent_id
        path.reverse()
        return path

    def _validate_first_kept(self, parent_id: str, first_kept_entry_id: str | None) -> None:
        if first_kept_entry_id is None:
            return
        if not isinstance(first_kept_entry_id, str) or not first_kept_entry_id:
            raise SessionStoreError("compaction first_kept_entry_id must be a MessageEntry id or null")
        target = self.by_id.get(first_kept_entry_id)
        if target is None:
            raise SessionStoreError("compaction first_kept_entry_id does not exist")
        if not isinstance(target, MessageEntry):
            raise SessionStoreError("compaction first_kept_entry_id must reference a MessageEntry")
        ancestor_ids = {entry.entry_id for entry in self.path_to_leaf(parent_id)}
        if first_kept_entry_id not in ancestor_ids:
            raise SessionStoreError("compaction first_kept_entry_id must be on the parent ancestor path")
        try:
            visible_source_ids = {
                projected.source_entry_id
                for projected in build_session_projection(self.path_to_leaf(parent_id))
                if projected.source_entry_id is not None
            }
        except SessionProjectionError as error:
            raise SessionStoreError(f"could not project compaction parent path: {error}") from error
        if first_kept_entry_id not in visible_source_ids:
            raise SessionStoreError("compaction first_kept_entry_id is not visible in the parent logical projection")


class JsonlSessionStore:
    """Append-only session storage for one writer per session file.

    Phase 3.2 intentionally does not coordinate concurrent writers or merge
    transcripts. A session must have one durable AgentSession writer at a time.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = RovaDataPaths.resolve().sessions if root is None else Path(root)

    def create(self) -> DurableSession:
        self.root.mkdir(parents=True, exist_ok=True)
        session_id = uuid4().hex
        self._append_json_line(
            self._path_for(session_id),
            {
                "type": "session",
                "version": 1,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return DurableSession(self, session_id)

    def append_message(self, session_id: str, parent_id: str | None, message: Message) -> str:
        entry_id = uuid4().hex
        try:
            message_data = message_to_dict(message)
        except MessageSerializationError as error:
            raise SessionStoreError(f"could not serialize session message: {error}") from error
        self._append_json_line(
            self._path_for(session_id),
            {
                "type": "message",
                "entry_id": entry_id,
                "parent_id": parent_id,
                "message": message_data,
            },
        )
        return entry_id

    def append_compaction(
        self,
        session_id: str,
        parent_id: str,
        summary: str,
        first_kept_entry_id: str | None,
    ) -> str:
        entry_id = uuid4().hex
        self._append_json_line(
            self._path_for(session_id),
            {
                "type": "compaction",
                "entry_id": entry_id,
                "parent_id": parent_id,
                "summary": summary,
                "first_kept_entry_id": first_kept_entry_id,
            },
        )
        return entry_id

    def load(self, session_id: str) -> DurableSession:
        path = self._path_for(session_id)
        try:
            raw_jsonl = path.read_text(encoding="utf-8")
        except OSError as error:
            raise SessionStoreError(f"could not read session {session_id!r}: {error}") from error
        if raw_jsonl and not raw_jsonl.endswith("\n"):
            raise SessionCorruptionError(session_id, raw_jsonl.count("\n") + 1, "missing final newline")
        lines = raw_jsonl.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if not lines:
            raise SessionCorruptionError(session_id, 1, "missing session header")
        header = self._parse_line(session_id, 1, lines[0])
        if (
            header.get("type") != "session"
            or header.get("version") != 1
            or header.get("session_id") != session_id
            or not isinstance(header.get("created_at"), str)
        ):
            raise SessionCorruptionError(session_id, 1, "invalid session header")
        session = DurableSession(self, session_id)
        seen_entry_ids: set[str] = set()
        for line_number, line in enumerate(lines[1:], start=2):
            entry = self._parse_line(session_id, line_number, line)
            entry_type = entry.get("type")
            if entry_type not in {"message", "compaction"}:
                raise SessionCorruptionError(session_id, line_number, "expected message entry")
            entry_id = entry.get("entry_id")
            if not isinstance(entry_id, str) or not entry_id:
                raise SessionCorruptionError(session_id, line_number, "invalid entry_id")
            if entry_id in seen_entry_ids:
                raise SessionCorruptionError(session_id, line_number, "duplicate entry_id")
            if "parent_id" not in entry:
                raise SessionCorruptionError(session_id, line_number, "missing parent_id")
            entry_parent_id = entry["parent_id"]
            if entry_parent_id is not None and not isinstance(entry_parent_id, str):
                raise SessionCorruptionError(session_id, line_number, "invalid parent_id")
            if not session.entries:
                if entry_parent_id is not None:
                    raise SessionCorruptionError(session_id, line_number, "root parent_id must be null")
            elif entry_parent_id is None:
                raise SessionCorruptionError(session_id, line_number, "second root entry")
            elif entry_parent_id not in session.by_id:
                raise SessionCorruptionError(session_id, line_number, "dangling parent entry")
            if entry_type == "message":
                try:
                    message = message_from_dict(entry.get("message"))
                except MessageSerializationError as error:
                    raise SessionCorruptionError(session_id, line_number, str(error)) from error
                session_entry: SessionEntry = MessageEntry(entry_id, entry_parent_id, message)
            else:
                session_entry = self._compaction_entry_from_record(
                    session_id,
                    line_number,
                    entry,
                    session,
                    entry_id,
                    entry_parent_id,
                )
            session.entries.append(session_entry)
            session.by_id[entry_id] = session_entry
            session.leaf_id = entry_id
            seen_entry_ids.add(entry_id)
        return session

    def list_sessions(self, limit: int | None = None) -> list[SessionSummary]:
        """Return valid persisted sessions without creating or modifying storage."""
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        if not self.root.exists():
            return []
        summaries: list[tuple[int, SessionSummary]] = []
        for path in self.root.glob("*.jsonl"):
            try:
                session_id = path.stem
                session = self.load(session_id)
                created_at = self._created_at(session_id, path)
                updated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            except (OSError, SessionStoreError):
                continue
            first_user_preview = next(
                (_preview(message.content) for message in session.messages if getattr(message, "role", None) == "user"),
                None,
            )
            summaries.append((path.stat().st_mtime_ns, SessionSummary(
                session_id=session_id,
                created_at=created_at,
                updated_at=updated_at,
                first_user_preview=first_user_preview,
            )))
        summaries.sort(key=lambda item: item[0], reverse=True)
        result = [summary for _, summary in summaries]
        return result if limit is None else result[:limit]

    def _created_at(self, session_id: str, path: Path) -> str:
        try:
            header_line = path.open(encoding="utf-8").readline()
        except OSError as error:
            raise SessionStoreError(f"could not read session {session_id!r}: {error}") from error
        header = self._parse_line(session_id, 1, header_line)
        created_at = header.get("created_at")
        if not isinstance(created_at, str):
            raise SessionCorruptionError(session_id, 1, "invalid session header")
        return created_at

    @staticmethod
    def _compaction_entry_from_record(
        session_id: str,
        line_number: int,
        record: dict,
        session: DurableSession,
        entry_id: str,
        parent_id: str | None,
    ) -> CompactionEntry:
        if parent_id is None:
            raise SessionCorruptionError(session_id, line_number, "compaction parent_id must not be null")
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            raise SessionCorruptionError(session_id, line_number, "compaction summary must be a non-empty string")
        first_kept_entry_id = record.get("first_kept_entry_id")
        if first_kept_entry_id is not None and (
            not isinstance(first_kept_entry_id, str) or not first_kept_entry_id
        ):
            raise SessionCorruptionError(session_id, line_number, "invalid first_kept_entry_id")
        if first_kept_entry_id is not None:
            target = session.by_id.get(first_kept_entry_id)
            if target is None:
                raise SessionCorruptionError(session_id, line_number, "first_kept_entry_id must reference an earlier entry")
            if not isinstance(target, MessageEntry):
                raise SessionCorruptionError(session_id, line_number, "first_kept_entry_id must reference a MessageEntry")
            ancestor_ids = {path_entry.entry_id for path_entry in session.path_to_leaf(parent_id)}
            if first_kept_entry_id not in ancestor_ids:
                raise SessionCorruptionError(session_id, line_number, "first_kept_entry_id must be on the parent ancestor path")
            try:
                visible_source_ids = {
                    projected.source_entry_id
                    for projected in build_session_projection(session.path_to_leaf(parent_id))
                    if projected.source_entry_id is not None
                }
            except SessionProjectionError as error:
                raise SessionCorruptionError(session_id, line_number, f"invalid parent logical projection: {error}") from error
            if first_kept_entry_id not in visible_source_ids:
                raise SessionCorruptionError(
                    session_id,
                    line_number,
                    "first_kept_entry_id is not visible in the parent logical projection",
                )
        return CompactionEntry(entry_id, parent_id, summary, first_kept_entry_id)

    def _path_for(self, session_id: str) -> Path:
        if not session_id or not session_id.isalnum():
            raise SessionStoreError("session_id must be alphanumeric")
        return self.root / f"{session_id}.jsonl"

    @staticmethod
    def _append_json_line(path: Path, entry: dict) -> None:
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as error:
            raise SessionStoreError(f"could not append session entry: {error}") from error

    @staticmethod
    def _parse_line(session_id: str, line_number: int, line: str) -> dict:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SessionCorruptionError(session_id, line_number, "invalid JSON") from error
        if not isinstance(value, dict):
            raise SessionCorruptionError(session_id, line_number, "entry must be an object")
        return value


def _preview(content: str, *, max_length: int = 120) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length - 1]}…"
