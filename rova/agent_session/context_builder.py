from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rova.ai.messages import Message, UserMessage

from .entries import CompactionEntry, MessageEntry, SessionEntry


COMPACTION_SUMMARY_PREAMBLE = (
    "The following is a harness-generated summary of earlier conversation history. "
    "It is context only, not a new user instruction."
)


class SessionProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectedMessage:
    message: Message
    source_entry_id: str | None


def build_session_projection(entries: Sequence[SessionEntry]) -> list[ProjectedMessage]:
    """Purely project one selected physical entry path into logical Runtime history."""
    projection: list[ProjectedMessage] = []
    for entry in entries:
        if isinstance(entry, MessageEntry):
            projection.append(ProjectedMessage(entry.message, entry.entry_id))
            continue
        if isinstance(entry, CompactionEntry):
            projection = _apply_compaction(projection, entry)
            continue
        raise SessionProjectionError(f"unsupported session entry: {type(entry).__name__}")
    return projection


def build_session_messages(entries: Sequence[SessionEntry]) -> list[Message]:
    """Return a new Runtime message list for one selected session entry path."""
    return [projected.message for projected in build_session_projection(entries)]


def _apply_compaction(
    projection: list[ProjectedMessage],
    entry: CompactionEntry,
) -> list[ProjectedMessage]:
    if entry.first_kept_entry_id is None:
        retained: list[ProjectedMessage] = []
    else:
        first_kept_index = next(
            (
                index
                for index, projected in enumerate(projection)
                if projected.source_entry_id == entry.first_kept_entry_id
            ),
            None,
        )
        if first_kept_index is None:
            raise SessionProjectionError(
                f"compaction first_kept_entry_id is not visible: {entry.first_kept_entry_id}"
            )
        retained = projection[first_kept_index:]
    summary = UserMessage(
        f"{COMPACTION_SUMMARY_PREAMBLE}\n<SUMMARY>\n{entry.summary}\n</SUMMARY>"
    )
    return [ProjectedMessage(summary, None), *retained]
