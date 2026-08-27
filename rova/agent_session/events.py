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
    trigger: Literal["automatic", "manual"]
    first_kept_entry_id: str | None
    pressure_before: int | None = None
    pressure_after: int | None = None
    error_type: str | None = None
    error_message: str | None = None
