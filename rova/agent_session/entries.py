from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from rova.ai.messages import Message


@dataclass
class MessageEntry:
    entry_id: str
    parent_id: str | None
    message: Message


@dataclass
class CompactionEntry:
    entry_id: str
    parent_id: str | None
    summary: str
    first_kept_entry_id: str | None


SessionEntry = Union[MessageEntry, CompactionEntry]
