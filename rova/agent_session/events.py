from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SessionMaintenanceEvent:
    """Structured, content-free observation emitted for session maintenance."""

    type: Literal[
        "compaction_started",
        "compaction_completed",
        "compaction_failed",
        "compaction_warning",
    ]
    trigger: Literal["automatic", "proactive", "overflow_recovery", "manual"]
    first_kept_entry_id: str | None
    pressure_before: int | None = None
    pressure_after: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    context_window: int | None = None
    reserve_tokens: int | None = None
    kept_recent_estimated_tokens: int | None = None
    summary_size_chars: int | None = None
