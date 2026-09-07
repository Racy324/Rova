from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rova.agent_core.events import AgentEvent


class ExecutionJournalError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletionReceipt:
    content: str
    is_error: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExecutionEnvironmentIdentity:
    """Durable logical identity of the environment used for one ToolCall."""

    kind: str
    sandbox_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("execution environment kind must be a non-empty string")
        if self.sandbox_id is not None and (not isinstance(self.sandbox_id, str) or not self.sandbox_id):
            raise ValueError("sandbox_id must be a non-empty string or None")
        if self.kind == "local" and self.sandbox_id is not None:
            raise ValueError("local execution environment cannot have a sandbox_id")


@dataclass(frozen=True)
class ExecutionEnvironmentStatus:
    """Current availability of a previously recorded execution environment."""

    identity: ExecutionEnvironmentIdentity
    available: bool
    state: str
    environment_lost: bool = False
    lifecycle_contradiction: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ValueError("execution environment availability must be a bool")
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("execution environment state must be a non-empty string")
        if self.environment_lost and self.available:
            raise ValueError("an available execution environment cannot be lost")
        if self.lifecycle_contradiction and self.available:
            raise ValueError("an available execution environment cannot have a lifecycle contradiction")


@dataclass(frozen=True)
class JournalRecord:
    assistant_entry_id: str | None
    batch_id: str
    tool_call_id: str
    call_index: int
    tool_name: str
    batch_mode: str
    execution_mode: str
    state: str
    outcome: str | None
    timestamp: str
    receipt: CompletionReceipt | None = None
    is_legacy: bool = False
    environment_kind: str | None = None
    sandbox_id: str | None = None


class ToolExecutionJournal:
    """Append-only per-session execution observations, outside conversation history."""

    def __init__(self, root: Path, session_id: str) -> None:
        self.path = Path(root) / f"{session_id}.executions.jsonl"

    def append(
        self,
        event: AgentEvent,
        *,
        assistant_entry_id: str | None = None,
        environment_identity: ExecutionEnvironmentIdentity | None = None,
    ) -> None:
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
        if assistant_entry_id is not None:
            if not isinstance(assistant_entry_id, str) or not assistant_entry_id:
                raise ValueError("assistant_entry_id must be a non-empty string or None")
            payload = {
                "version": 3 if environment_identity is not None else 2,
                "record_type": "state",
                "assistant_entry_id": assistant_entry_id,
                **payload,
            }
            if environment_identity is not None:
                payload["environment_kind"] = environment_identity.kind
                payload["sandbox_id"] = environment_identity.sandbox_id
            if event.execution_state == "completed":
                if event.result is None:
                    raise ValueError("completed journal record requires canonical result text")
                payload["completion_receipt"] = {
                    "content": event.result,
                    "is_error": event.is_error,
                    "metadata": _json_object(event.metadata or {}, "completion receipt metadata"),
                }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load(self) -> list[JournalRecord]:
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as error:
            raise ExecutionJournalError(f"could not read execution journal: {error}") from error
        if raw and not raw.endswith("\n"):
            raise ExecutionJournalError("execution journal has an incomplete final record")
        records: list[JournalRecord] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line:
                raise ExecutionJournalError(f"execution journal line {line_number} is empty")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExecutionJournalError(f"execution journal line {line_number} is invalid JSON") from error
            records.append(_record_from_dict(value, line_number))
        return records


def _record_from_dict(value: object, line_number: int) -> JournalRecord:
    if not isinstance(value, dict):
        raise ExecutionJournalError(f"execution journal line {line_number} must be an object")
    if value.get("version") is None:
        return _record_from_payload(value, line_number, assistant_entry_id=None, receipt=None, is_legacy=True)
    version = value.get("version")
    if version not in {2, 3} or value.get("record_type") != "state":
        raise ExecutionJournalError(f"execution journal line {line_number} has an unsupported version or record type")
    assistant_entry_id = _string(value.get("assistant_entry_id"), "assistant_entry_id", line_number)
    receipt_data = value.get("completion_receipt")
    receipt = _receipt_from_dict(receipt_data, line_number) if receipt_data is not None else None
    if value.get("state") == "completed" and receipt is None:
        raise ExecutionJournalError(f"execution journal line {line_number} completed record has no receipt")
    if value.get("state") != "completed" and receipt is not None:
        raise ExecutionJournalError(f"execution journal line {line_number} non-completed record has a receipt")
    environment_kind: str | None = None
    sandbox_id: str | None = None
    if version == 3:
        environment_kind = _string(value.get("environment_kind"), "environment_kind", line_number)
        sandbox_id = _optional_string(value.get("sandbox_id"), "sandbox_id", line_number)
        try:
            ExecutionEnvironmentIdentity(environment_kind, sandbox_id)
        except ValueError as error:
            raise ExecutionJournalError(f"execution journal line {line_number} has invalid environment identity: {error}") from error
    return _record_from_payload(
        value,
        line_number,
        assistant_entry_id=assistant_entry_id,
        receipt=receipt,
        is_legacy=False,
        environment_kind=environment_kind,
        sandbox_id=sandbox_id,
    )


def _record_from_payload(
    value: dict[str, Any],
    line_number: int,
    *,
    assistant_entry_id: str | None,
    receipt: CompletionReceipt | None,
    is_legacy: bool,
    environment_kind: str | None = None,
    sandbox_id: str | None = None,
) -> JournalRecord:
    state = _string(value.get("state"), "state", line_number)
    if state not in {"started", "executor_completed", "completed", "cancelled", "interrupted"}:
        raise ExecutionJournalError(f"execution journal line {line_number} has unknown state")
    outcome = value.get("outcome")
    if outcome is not None and not isinstance(outcome, str):
        raise ExecutionJournalError(f"execution journal line {line_number} has invalid outcome")
    call_index = value.get("call_index")
    if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
        raise ExecutionJournalError(f"execution journal line {line_number} has invalid call_index")
    return JournalRecord(
        assistant_entry_id=assistant_entry_id,
        batch_id=_string(value.get("batch_id"), "batch_id", line_number),
        tool_call_id=_string(value.get("tool_call_id"), "tool_call_id", line_number),
        call_index=call_index,
        tool_name=_string(value.get("tool_name"), "tool_name", line_number),
        batch_mode=_string(value.get("batch_mode"), "batch_mode", line_number),
        execution_mode=_string(value.get("execution_mode"), "execution_mode", line_number),
        state=state,
        outcome=outcome,
        timestamp=_string(value.get("timestamp"), "timestamp", line_number),
        receipt=receipt,
        is_legacy=is_legacy,
        environment_kind=environment_kind,
        sandbox_id=sandbox_id,
    )


def _receipt_from_dict(value: object, line_number: int) -> CompletionReceipt:
    if not isinstance(value, dict):
        raise ExecutionJournalError(f"execution journal line {line_number} receipt must be an object")
    content = value.get("content")
    if not isinstance(content, str):
        raise ExecutionJournalError(f"execution journal line {line_number} has invalid completion receipt content")
    is_error = value.get("is_error")
    if not isinstance(is_error, bool):
        raise ExecutionJournalError(f"execution journal line {line_number} receipt is_error must be a bool")
    return CompletionReceipt(content, is_error, _json_object(value.get("metadata"), "completion receipt metadata"))


def _string(value: object, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionJournalError(f"execution journal line {line_number} has invalid {name}")
    return value


def _optional_string(value: object, name: str, line_number: int) -> str | None:
    if value is None:
        return None
    return _string(value, name, line_number)


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExecutionJournalError(f"{name} must be a JSON object")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    return decoded
