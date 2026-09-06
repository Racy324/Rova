from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rova.agent_core.events import AgentEvent


class ToolExecutionJournal:
    """Append-only per-session execution observations, outside conversation history."""

    def __init__(self, root: Path, session_id: str) -> None:
        self.path = Path(root) / f"{session_id}.executions.jsonl"

    def append(self, event: AgentEvent) -> None:
        if event.type != "tool_execution_state":
            raise ValueError("execution journal only accepts tool_execution_state events")
        payload = {
            "batch_id": event.batch_id,
            "tool_call_id": event.tool_call_id,
            "call_index": event.call_index,
            "tool_name": event.tool_name,
            "batch_mode": event.batch_mode,
            "execution_mode": event.execution_mode,
            "state": event.execution_state,
            "outcome": event.outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if not all(payload[key] is not None for key in ("batch_id", "tool_call_id", "call_index", "tool_name", "batch_mode", "execution_mode", "state")):
            raise ValueError("execution journal event is missing required tool state metadata")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
