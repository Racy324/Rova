from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from rova.ai.messages import AssistantMessage, Usage


class RunStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    HARNESS_ERROR = "harness_error"
    ABORTED = "aborted"
    PROVIDER_ERROR = "provider_error"
    SESSION_PERSISTENCE_ERROR = "session_persistence_error"


class TerminationReason(Enum):
    FINAL_RESPONSE = "final_response"
    MAX_TURNS = "max_turns"
    PROVIDER_ERROR = "provider_error"
    HARNESS_ERROR = "harness_error"
    ABORTED = "aborted"
    SESSION_PERSISTENCE_ERROR = "session_persistence_error"


class ToolOutcome(Enum):
    SUCCESS = "success"
    TOOL_INPUT_ERROR = "tool_input_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    POLICY_DENIED = "policy_denied"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_ERROR = "approval_error"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    COMMAND_NONZERO_EXIT = "command_nonzero_exit"
    COMMAND_TIMEOUT = "command_timeout"


class CompactionStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning"


class CompactionTrigger(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class MemoryMaintenanceKind(Enum):
    EXTRACTION = "extraction"
    CONSOLIDATION = "consolidation"


class MemoryMaintenanceStatus(Enum):
    TRIGGERED = "triggered"
    NOOP = "noop"
    UPDATED = "updated"
    FAILED = "failed"


@dataclass(frozen=True)
class TraceError:
    error_type: str
    message: str


@dataclass
class TurnTrace:
    turn_index: int
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    assistant_message: AssistantMessage | None = None
    stop_reason: str | None = None
    usage: Usage | None = None
    tool_call_ids: list[str] = field(default_factory=list)


@dataclass
class ToolExecutionTrace:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    turn_index: int | None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    result: str | None = None
    is_error: bool | None = None
    outcome: ToolOutcome | None = None
    policy_decision: str | None = None
    policy_reason: str | None = None
    approval_required: bool | None = None
    approval_decision: str | None = None
    command: str | None = None
    exit_code: int | None = None
    timed_out: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactionTrace:
    started_at: datetime
    trigger: CompactionTrigger
    first_kept_entry_id: str | None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: CompactionStatus = CompactionStatus.RUNNING
    pressure_before: int | None = None
    pressure_after: int | None = None
    error: TraceError | None = None


@dataclass
class MemoryMaintenanceTrace:
    kind: MemoryMaintenanceKind
    status: MemoryMaintenanceStatus
    changed_documents: list[str] = field(default_factory=list)
    error: TraceError | None = None


@dataclass
class RunTrace:
    run_id: str
    started_at: datetime
    session_id: str | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: RunStatus = RunStatus.RUNNING
    termination_reason: TerminationReason | None = None
    turns: list[TurnTrace] = field(default_factory=list)
    tool_executions: list[ToolExecutionTrace] = field(default_factory=list)
    compactions: list[CompactionTrace] = field(default_factory=list)
    memory_events: list[MemoryMaintenanceTrace] = field(default_factory=list)
    usage: Usage | None = None
    final_message: AssistantMessage | None = None
    error: TraceError | None = None
