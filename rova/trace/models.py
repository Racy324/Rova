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
    CONTEXT_OVERFLOW = "context_overflow"
    SESSION_PERSISTENCE_ERROR = "session_persistence_error"


class TerminationReason(Enum):
    FINAL_RESPONSE = "final_response"
    MAX_TURNS = "max_turns"
    PROVIDER_ERROR = "provider_error"
    CONTEXT_OVERFLOW = "context_overflow"
    HARNESS_ERROR = "harness_error"
    ABORTED = "aborted"
    SESSION_PERSISTENCE_ERROR = "session_persistence_error"


class ToolOutcome(Enum):
    SUCCESS = "success"
    TOOL_INPUT_ERROR = "tool_input_error"
    HOOK_BLOCKED = "hook_blocked"
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
    PROACTIVE = "proactive"
    OVERFLOW_RECOVERY = "overflow_recovery"
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
class StepUsageTrace:
    """Request estimate and provider-reported usage stay semantically separate."""

    estimated_input_tokens: int | None = None
    actual_usage: Usage | None = None


@dataclass
class ToolResultTrace:
    """Canonical, model-visible ToolResult projection without raw artifact content."""

    content: str
    externalized: bool
    artifact_ref: str | None
    original_size_chars: int
    preview_truncated: bool
    preview_size_chars: int


@dataclass
class ToolCallTrace:
    """One Assistant ToolCall lifecycle, including calls that never execute."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    batch_id: str | None
    call_index: int
    batch_mode: str | None
    execution_mode: str | None
    executed: bool = False
    committed: bool = False
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    outcome: ToolOutcome | None = None
    failure_stage: str | None = None
    policy_decision: str | None = None
    policy_reason: str | None = None
    approval_required: bool | None = None
    approval_decision: str | None = None
    exit_code: int | None = None
    timed_out: bool | None = None
    result: ToolResultTrace | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepTrace:
    """One Provider decision and the ToolCall batch issued by that Assistant message."""

    step_index: int
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    assistant_message: AssistantMessage | None = None
    usage: StepUsageTrace = field(default_factory=StepUsageTrace)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)


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
    # V3 context-maintenance facts.  The older pressure names remain only for
    # in-memory legacy trace compatibility.
    before_estimated_tokens: int | None = None
    after_estimated_tokens: int | None = None
    context_window: int | None = None
    reserve_tokens: int | None = None
    summary_size_chars: int | None = None
    kept_recent_estimated_tokens: int | None = None
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
    input_entry_id: str | None = None
    input_message: str = ""
    steps: list[StepTrace] = field(default_factory=list)
    actual_usage_complete: bool = False
    actual_usage: Usage | None = None
    peak_estimated_context_tokens: int | None = None
    # Transitional in-memory V1 compatibility. V3 serialization deliberately
    # projects only the canonical `steps` structure.
    turns: list[TurnTrace] = field(default_factory=list)
    tool_executions: list[ToolExecutionTrace] = field(default_factory=list)
    compactions: list[CompactionTrace] = field(default_factory=list)
    memory_events: list[MemoryMaintenanceTrace] = field(default_factory=list)
    usage: Usage | None = None
    final_message: AssistantMessage | None = None
    error: TraceError | None = None
