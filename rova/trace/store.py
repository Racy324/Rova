from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, Usage

from .models import (
    CompactionStatus,
    CompactionTrace,
    CompactionTrigger,
    MemoryMaintenanceKind,
    MemoryMaintenanceStatus,
    MemoryMaintenanceTrace,
    RunStatus,
    RunTrace,
    StepTrace,
    StepUsageTrace,
    TerminationReason,
    ToolExecutionTrace,
    ToolCallTrace,
    ToolOutcome,
    ToolResultTrace,
    TraceError,
    TurnTrace,
)


SCHEMA_VERSION = 3


class TraceStoreError(RuntimeError):
    pass


class TraceCorruptionError(TraceStoreError):
    def __init__(self, line_number: int, reason: str) -> None:
        super().__init__(f"trace store is corrupt at line {line_number}: {reason}")


class TraceStore(Protocol):
    def append(self, trace: RunTrace) -> None: ...
    def load_all(self) -> list[RunTrace]: ...


class JsonlTraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, trace: RunTrace) -> None:
        if trace.status is RunStatus.RUNNING or trace.ended_at is None:
            raise TraceStoreError("only finalized RunTrace values can be persisted")
        existing = {item.run_id for item in self.load_all()} if self.path.exists() else set()
        if trace.run_id in existing:
            raise TraceStoreError(f"duplicate run_id: {trace.run_id}")
        try:
            record = {"schema_version": SCHEMA_VERSION, "trace": run_trace_to_dict(trace)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as error:
            raise TraceStoreError(f"could not append trace: {error}") from error

    def load_all(self) -> list[RunTrace]:
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as error:
            raise TraceStoreError(f"could not read trace store: {error}") from error
        if raw and not raw.endswith("\n"):
            raise TraceCorruptionError(raw.count("\n") + 1, "missing final newline")
        traces: list[RunTrace] = []
        seen: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceCorruptionError(line_number, "invalid JSON") from error
            if (
                not isinstance(record, dict)
                or set(record) != {"schema_version", "trace"}
                or record.get("schema_version") not in {1, 2, SCHEMA_VERSION}
            ):
                raise TraceCorruptionError(line_number, "unsupported schema_version")
            try:
                payload = record.get("trace")
                if record.get("schema_version") == 1 and isinstance(payload, dict):
                    payload = {**payload, "memory_events": []}
                trace = run_trace_from_dict(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise TraceCorruptionError(line_number, str(error)) from error
            if trace.run_id in seen:
                raise TraceCorruptionError(line_number, "duplicate run_id")
            seen.add(trace.run_id)
            traces.append(trace)
        return traces


def _legacy_trace_to_dict(trace: RunTrace) -> dict[str, object]:
    """Explicit v2 serializer; trace metadata stays local and provider-agnostic."""
    return {
        "run_id": trace.run_id,
        "started_at": _datetime_to_string(trace.started_at),
        "session_id": trace.session_id,
        "ended_at": _optional_datetime_to_string(trace.ended_at),
        "duration_ms": _optional_number(trace.duration_ms, "duration_ms"),
        "status": trace.status.value,
        "termination_reason": trace.termination_reason.value if trace.termination_reason else None,
        "turns": [_turn_to_dict(turn) for turn in trace.turns],
        "tool_executions": [_tool_to_dict(tool) for tool in trace.tool_executions],
        "compactions": [_compaction_to_dict(compaction) for compaction in trace.compactions],
        "memory_events": [_memory_event_to_dict(event) for event in trace.memory_events],
        "usage": _usage_to_dict(trace.usage),
        "final_message": _message_to_dict(trace.final_message),
        "error": _error_to_dict(trace.error),
    }


def trace_to_dict(trace: RunTrace) -> dict[str, object]:
    """Serialize only the canonical V3 run/step projection."""
    return {
        "run_id": trace.run_id,
        "session_id": trace.session_id,
        "input_entry_id": trace.input_entry_id,
        "input_message": trace.input_message,
        "started_at": _datetime_to_string(trace.started_at),
        "ended_at": _optional_datetime_to_string(trace.ended_at),
        "duration_ms": _optional_number(trace.duration_ms, "duration_ms"),
        "status": trace.status.value,
        "termination_reason": trace.termination_reason.value if trace.termination_reason else None,
        "steps": [_step_to_dict(step) for step in trace.steps],
        "compactions": [_compaction_to_dict(compaction) for compaction in trace.compactions],
        "actual_usage_complete": trace.actual_usage_complete,
        "actual_usage": _usage_to_dict(trace.actual_usage),
        "peak_estimated_context_tokens": trace.peak_estimated_context_tokens,
        "error": _error_to_dict(trace.error),
    }


def trace_from_dict(data: object) -> RunTrace:
    if not isinstance(data, dict) or "steps" not in data:
        return _legacy_trace_from_dict(data)
    item = _object(data, "trace", {
        "run_id", "session_id", "input_entry_id", "input_message", "started_at", "ended_at",
        "duration_ms", "status", "termination_reason", "steps", "compactions",
        "actual_usage_complete", "actual_usage", "peak_estimated_context_tokens", "error",
    })
    trace = RunTrace(
        run_id=_string(item["run_id"], "run_id"),
        started_at=_datetime(item["started_at"], "started_at"),
        session_id=_optional_string(item["session_id"], "session_id"),
        input_entry_id=_optional_string(item["input_entry_id"], "input_entry_id"),
        input_message=_string(item["input_message"], "input_message"),
        ended_at=_optional_datetime(item["ended_at"], "ended_at"),
        duration_ms=_optional_number(item["duration_ms"], "duration_ms"),
        status=RunStatus(_string(item["status"], "status")),
        termination_reason=_optional_enum(item["termination_reason"], TerminationReason, "termination_reason"),
        steps=[_step_from_dict(value) for value in _list(item["steps"], "steps")],
        compactions=[_compaction_from_dict(value) for value in _list(item["compactions"], "compactions")],
        actual_usage_complete=_bool(item["actual_usage_complete"], "actual_usage_complete"),
        actual_usage=_usage_from_dict(item["actual_usage"]),
        peak_estimated_context_tokens=_optional_integer(item["peak_estimated_context_tokens"], "peak_estimated_context_tokens"),
        error=_error_from_dict(item["error"]),
    )
    if trace.status is RunStatus.RUNNING or trace.ended_at is None:
        raise ValueError("persisted trace must be finalized")
    return trace


def run_trace_to_dict(trace: RunTrace) -> dict[str, object]:
    """Public explicit serializer for the versioned RunTrace schema."""
    return trace_to_dict(trace)


def _legacy_trace_from_dict(data: object) -> RunTrace:
    item = _object(data, "trace", {
        "run_id", "started_at", "session_id", "ended_at", "duration_ms", "status",
        "termination_reason", "turns", "tool_executions", "compactions", "memory_events", "usage",
        "final_message", "error",
    })
    trace = RunTrace(
        run_id=_string(item["run_id"], "run_id"),
        started_at=_datetime(item["started_at"], "started_at"),
        session_id=_optional_string(item["session_id"], "session_id"),
        ended_at=_optional_datetime(item["ended_at"], "ended_at"),
        duration_ms=_optional_number(item["duration_ms"], "duration_ms"),
        status=RunStatus(_string(item["status"], "status")),
        termination_reason=_optional_enum(item["termination_reason"], TerminationReason, "termination_reason"),
        turns=[_turn_from_dict(value) for value in _list(item["turns"], "turns")],
        tool_executions=[_tool_from_dict(value) for value in _list(item["tool_executions"], "tool_executions")],
        compactions=[_compaction_from_dict(value) for value in _list(item["compactions"], "compactions")],
        memory_events=[_memory_event_from_dict(value) for value in _list(item["memory_events"], "memory_events")],
        usage=_usage_from_dict(item["usage"]),
        final_message=_message_from_dict(item["final_message"]),
        error=_error_from_dict(item["error"]),
    )
    if trace.status is RunStatus.RUNNING or trace.ended_at is None:
        raise ValueError("persisted trace must be finalized")
    return trace


def run_trace_from_dict(data: object) -> RunTrace:
    """Public strict deserializer for the versioned RunTrace schema."""
    return trace_from_dict(data)


def _step_to_dict(step: StepTrace) -> dict[str, object]:
    return {
        "step_index": step.step_index,
        "started_at": _datetime_to_string(step.started_at),
        "ended_at": _optional_datetime_to_string(step.ended_at),
        "duration_ms": _optional_number(step.duration_ms, "duration_ms"),
        "assistant_message": _message_to_dict(step.assistant_message),
        "usage": {
            "estimated_input_tokens": step.usage.estimated_input_tokens,
            "actual_usage": _usage_to_dict(step.usage.actual_usage),
        },
        "tool_calls": [_tool_call_to_dict(tool_call) for tool_call in step.tool_calls],
    }


def _step_from_dict(value: object) -> StepTrace:
    item = _object(value, "step", {
        "step_index", "started_at", "ended_at", "duration_ms", "assistant_message", "usage", "tool_calls",
    })
    usage = _object(item["usage"], "step usage", {"estimated_input_tokens", "actual_usage"})
    return StepTrace(
        step_index=_integer(item["step_index"], "step_index"),
        started_at=_datetime(item["started_at"], "started_at"),
        ended_at=_optional_datetime(item["ended_at"], "ended_at"),
        duration_ms=_optional_number(item["duration_ms"], "duration_ms"),
        assistant_message=_message_from_dict(item["assistant_message"]),
        usage=StepUsageTrace(
            estimated_input_tokens=_optional_integer(usage["estimated_input_tokens"], "estimated_input_tokens"),
            actual_usage=_usage_from_dict(usage["actual_usage"]),
        ),
        tool_calls=[_tool_call_from_dict(item) for item in _list(item["tool_calls"], "tool_calls")],
    )


def _tool_call_to_dict(tool_call: ToolCallTrace) -> dict[str, object]:
    return {
        "tool_call_id": tool_call.tool_call_id,
        "tool_name": tool_call.tool_name,
        "arguments": _json_data(tool_call.arguments),
        "batch_id": tool_call.batch_id,
        "call_index": tool_call.call_index,
        "batch_mode": tool_call.batch_mode,
        "execution_mode": tool_call.execution_mode,
        "executed": tool_call.executed,
        "committed": tool_call.committed,
        "started_at": _optional_datetime_to_string(tool_call.started_at),
        "ended_at": _optional_datetime_to_string(tool_call.ended_at),
        "duration_ms": _optional_number(tool_call.duration_ms, "duration_ms"),
        "outcome": tool_call.outcome.value if tool_call.outcome else None,
        "failure_stage": tool_call.failure_stage,
        "policy_decision": tool_call.policy_decision,
        "policy_reason": tool_call.policy_reason,
        "approval_required": tool_call.approval_required,
        "approval_decision": tool_call.approval_decision,
        "exit_code": tool_call.exit_code,
        "timed_out": tool_call.timed_out,
        "result": _tool_result_to_dict(tool_call.result),
        "diagnostics": _json_data(tool_call.diagnostics),
    }


def _tool_call_from_dict(value: object) -> ToolCallTrace:
    item = _object(value, "tool call", {
        "tool_call_id", "tool_name", "arguments", "batch_id", "call_index", "batch_mode", "execution_mode",
        "executed", "committed", "started_at", "ended_at", "duration_ms", "outcome", "failure_stage",
        "policy_decision", "policy_reason", "approval_required", "approval_decision", "exit_code", "timed_out",
        "result", "diagnostics",
    })
    return ToolCallTrace(
        tool_call_id=_string(item["tool_call_id"], "tool_call_id"),
        tool_name=_string(item["tool_name"], "tool_name"),
        arguments=_json_object(item["arguments"], "arguments"),
        batch_id=_optional_string(item["batch_id"], "batch_id"),
        call_index=_integer(item["call_index"], "call_index"),
        batch_mode=_optional_string(item["batch_mode"], "batch_mode"),
        execution_mode=_optional_string(item["execution_mode"], "execution_mode"),
        executed=_bool(item["executed"], "executed"),
        committed=_bool(item["committed"], "committed"),
        started_at=_optional_datetime(item["started_at"], "started_at"),
        ended_at=_optional_datetime(item["ended_at"], "ended_at"),
        duration_ms=_optional_number(item["duration_ms"], "duration_ms"),
        outcome=_optional_enum(item["outcome"], ToolOutcome, "outcome"),
        failure_stage=_optional_string(item["failure_stage"], "failure_stage"),
        policy_decision=_optional_string(item["policy_decision"], "policy_decision"),
        policy_reason=_optional_string(item["policy_reason"], "policy_reason"),
        approval_required=_optional_bool(item["approval_required"], "approval_required"),
        approval_decision=_optional_string(item["approval_decision"], "approval_decision"),
        exit_code=_optional_integer(item["exit_code"], "exit_code"),
        timed_out=_optional_bool(item["timed_out"], "timed_out"),
        result=_tool_result_from_dict(item["result"]),
        diagnostics=_json_object(item["diagnostics"], "diagnostics"),
    )


def _tool_result_to_dict(result: ToolResultTrace | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "content": result.content,
        "externalized": result.externalized,
        "artifact_ref": result.artifact_ref,
        "original_size_chars": result.original_size_chars,
        "preview_truncated": result.preview_truncated,
        "preview_size_chars": result.preview_size_chars,
    }


def _tool_result_from_dict(value: object) -> ToolResultTrace | None:
    if value is None:
        return None
    item = _object(value, "tool result", {
        "content", "externalized", "artifact_ref", "original_size_chars", "preview_truncated", "preview_size_chars",
    })
    return ToolResultTrace(
        content=_string(item["content"], "content"),
        externalized=_bool(item["externalized"], "externalized"),
        artifact_ref=_optional_string(item["artifact_ref"], "artifact_ref"),
        original_size_chars=_integer(item["original_size_chars"], "original_size_chars"),
        preview_truncated=_bool(item["preview_truncated"], "preview_truncated"),
        preview_size_chars=_integer(item["preview_size_chars"], "preview_size_chars"),
    )


def _turn_to_dict(turn: TurnTrace) -> dict[str, object]:
    return {"turn_index": turn.turn_index, "started_at": _datetime_to_string(turn.started_at), "ended_at": _optional_datetime_to_string(turn.ended_at), "duration_ms": _optional_number(turn.duration_ms, "duration_ms"), "assistant_message": _message_to_dict(turn.assistant_message), "stop_reason": turn.stop_reason, "usage": _usage_to_dict(turn.usage), "tool_call_ids": list(turn.tool_call_ids)}


def _turn_from_dict(value: object) -> TurnTrace:
    item = _object(value, "turn", {"turn_index", "started_at", "ended_at", "duration_ms", "assistant_message", "stop_reason", "usage", "tool_call_ids"})
    return TurnTrace(_integer(item["turn_index"], "turn_index"), _datetime(item["started_at"], "started_at"), _optional_datetime(item["ended_at"], "ended_at"), _optional_number(item["duration_ms"], "duration_ms"), _message_from_dict(item["assistant_message"]), _optional_string(item["stop_reason"], "stop_reason"), _usage_from_dict(item["usage"]), [_string(entry, "tool_call_id") for entry in _list(item["tool_call_ids"], "tool_call_ids")])


def _tool_to_dict(tool: ToolExecutionTrace) -> dict[str, object]:
    return {"tool_call_id": tool.tool_call_id, "tool_name": tool.tool_name, "arguments": _json_data(tool.arguments), "turn_index": tool.turn_index, "started_at": _datetime_to_string(tool.started_at), "ended_at": _optional_datetime_to_string(tool.ended_at), "duration_ms": _optional_number(tool.duration_ms, "duration_ms"), "result": tool.result, "is_error": tool.is_error, "outcome": tool.outcome.value if tool.outcome else None, "policy_decision": tool.policy_decision, "policy_reason": tool.policy_reason, "approval_required": tool.approval_required, "approval_decision": tool.approval_decision, "command": tool.command, "exit_code": tool.exit_code, "timed_out": tool.timed_out, "metadata": _json_data(tool.metadata)}


def _tool_from_dict(value: object) -> ToolExecutionTrace:
    item = _object(value, "tool execution", {"tool_call_id", "tool_name", "arguments", "turn_index", "started_at", "ended_at", "duration_ms", "result", "is_error", "outcome", "policy_decision", "policy_reason", "approval_required", "approval_decision", "command", "exit_code", "timed_out", "metadata"})
    return ToolExecutionTrace(_string(item["tool_call_id"], "tool_call_id"), _string(item["tool_name"], "tool_name"), _json_object(item["arguments"], "arguments"), _optional_integer(item["turn_index"], "turn_index"), _datetime(item["started_at"], "started_at"), _optional_datetime(item["ended_at"], "ended_at"), _optional_number(item["duration_ms"], "duration_ms"), _optional_string(item["result"], "result"), _optional_bool(item["is_error"], "is_error"), _optional_enum(item["outcome"], ToolOutcome, "outcome"), _optional_string(item["policy_decision"], "policy_decision"), _optional_string(item["policy_reason"], "policy_reason"), _optional_bool(item["approval_required"], "approval_required"), _optional_string(item["approval_decision"], "approval_decision"), _optional_string(item["command"], "command"), _optional_integer(item["exit_code"], "exit_code"), _optional_bool(item["timed_out"], "timed_out"), _json_object(item["metadata"], "metadata"))


def _compaction_to_dict(compaction: CompactionTrace) -> dict[str, object]:
    return {"started_at": _datetime_to_string(compaction.started_at), "trigger": compaction.trigger.value, "first_kept_entry_id": compaction.first_kept_entry_id, "ended_at": _optional_datetime_to_string(compaction.ended_at), "duration_ms": _optional_number(compaction.duration_ms, "duration_ms"), "status": compaction.status.value, "pressure_before": compaction.pressure_before, "pressure_after": compaction.pressure_after, "before_estimated_tokens": compaction.before_estimated_tokens, "after_estimated_tokens": compaction.after_estimated_tokens, "context_window": compaction.context_window, "reserve_tokens": compaction.reserve_tokens, "summary_size_chars": compaction.summary_size_chars, "kept_recent_estimated_tokens": compaction.kept_recent_estimated_tokens, "error": _error_to_dict(compaction.error)}


def _compaction_from_dict(value: object) -> CompactionTrace:
    if isinstance(value, dict) and "before_estimated_tokens" not in value:
        value = {
            **value,
            "before_estimated_tokens": value.get("pressure_before"),
            "after_estimated_tokens": value.get("pressure_after"),
            "context_window": None,
            "reserve_tokens": None,
            "summary_size_chars": None,
            "kept_recent_estimated_tokens": None,
        }
    item = _object(value, "compaction", {"started_at", "trigger", "first_kept_entry_id", "ended_at", "duration_ms", "status", "pressure_before", "pressure_after", "before_estimated_tokens", "after_estimated_tokens", "context_window", "reserve_tokens", "summary_size_chars", "kept_recent_estimated_tokens", "error"})
    return CompactionTrace(_datetime(item["started_at"], "started_at"), CompactionTrigger(_string(item["trigger"], "trigger")), _optional_string(item["first_kept_entry_id"], "first_kept_entry_id"), _optional_datetime(item["ended_at"], "ended_at"), _optional_number(item["duration_ms"], "duration_ms"), CompactionStatus(_string(item["status"], "status")), _optional_integer(item["pressure_before"], "pressure_before"), _optional_integer(item["pressure_after"], "pressure_after"), _optional_integer(item["before_estimated_tokens"], "before_estimated_tokens"), _optional_integer(item["after_estimated_tokens"], "after_estimated_tokens"), _optional_integer(item["context_window"], "context_window"), _optional_integer(item["reserve_tokens"], "reserve_tokens"), _optional_integer(item["summary_size_chars"], "summary_size_chars"), _optional_integer(item["kept_recent_estimated_tokens"], "kept_recent_estimated_tokens"), _error_from_dict(item["error"]))


def _memory_event_to_dict(event: MemoryMaintenanceTrace) -> dict[str, object]:
    return {
        "kind": event.kind.value,
        "status": event.status.value,
        "changed_documents": list(event.changed_documents),
        "error": _error_to_dict(event.error),
    }


def _memory_event_from_dict(value: object) -> MemoryMaintenanceTrace:
    item = _object(value, "memory event", {"kind", "status", "changed_documents", "error"})
    documents = [_string(item, "changed_document") for item in _list(item["changed_documents"], "changed_documents")]
    if any(item not in {"USER.md", "MEMORY.md"} for item in documents):
        raise ValueError("invalid memory changed document")
    return MemoryMaintenanceTrace(
        MemoryMaintenanceKind(_string(item["kind"], "kind")),
        MemoryMaintenanceStatus(_string(item["status"], "status")),
        documents,
        _error_from_dict(item["error"]),
    )


def _message_to_dict(message: AssistantMessage | None) -> dict[str, object] | None:
    if message is None:
        return None
    content: list[dict[str, object]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            content.append({"type": "toolCall", "id": block.id, "name": block.name, "arguments": _json_data(block.arguments)})
        else:
            raise TypeError(f"unsupported assistant block: {type(block).__name__}")
    return {"content": content, "stop_reason": message.stop_reason, "partial": message.partial, "usage": _usage_to_dict(message.usage)}


def _message_from_dict(value: object) -> AssistantMessage | None:
    if value is None:
        return None
    item = _object(value, "assistant message", {"content", "stop_reason", "partial", "usage"})
    content = []
    for block in _list(item["content"], "content"):
        block_item = _object(block, "assistant block", {"type", "text"} if isinstance(block, dict) and block.get("type") == "text" else {"type", "id", "name", "arguments"})
        if block_item["type"] == "text":
            content.append(TextBlock(_string(block_item["text"], "text")))
        elif block_item["type"] == "toolCall":
            content.append(ToolCall(_string(block_item["id"], "id"), _string(block_item["name"], "name"), _json_object(block_item["arguments"], "arguments")))
        else:
            raise ValueError("unsupported assistant block type")
    return AssistantMessage(content, _string(item["stop_reason"], "stop_reason"), _bool(item["partial"], "partial"), usage=_usage_from_dict(item["usage"]))


def _usage_to_dict(usage: Usage | None) -> dict[str, int] | None:
    return None if usage is None else {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens}


def _usage_from_dict(value: object) -> Usage | None:
    if value is None:
        return None
    item = _object(value, "usage", {"input_tokens", "output_tokens", "total_tokens"})
    return Usage(_integer(item["input_tokens"], "input_tokens"), _integer(item["output_tokens"], "output_tokens"), _integer(item["total_tokens"], "total_tokens"))


def _error_to_dict(error: TraceError | None) -> dict[str, str] | None:
    return None if error is None else {"error_type": error.error_type, "message": error.message}


def _error_from_dict(value: object) -> TraceError | None:
    if value is None:
        return None
    item = _object(value, "error", {"error_type", "message"})
    return TraceError(_string(item["error_type"], "error_type"), _string(item["message"], "message"))


def _datetime_to_string(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _optional_datetime_to_string(value: datetime | None) -> str | None:
    return None if value is None else _datetime_to_string(value)


def _datetime(value: object, name: str) -> datetime:
    text = _string(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    return parsed


def _optional_datetime(value: object, name: str) -> datetime | None:
    return None if value is None else _datetime(value, name)


def _object(value: object, name: str, expected_keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{name} has an invalid schema")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    return None if value is None else _bool(value, name)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _optional_number(value: object, name: str) -> float | int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _optional_enum(value: object, enum_type, name: str):
    return None if value is None else enum_type(_string(value, name))


def _json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return _json_data(value)


def _json_data(value: object):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON number must be finite")
        return value
    if isinstance(value, list):
        return [_json_data(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_data(item) for key, item in value.items()}
    raise ValueError("value is not JSON-compatible")
