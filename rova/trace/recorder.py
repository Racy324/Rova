from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

from rova.agent_core.agent import Agent
from rova.agent_core.events import AgentEvent, AgentTerminationReason
from rova.agent_session.agent_session import SessionPersistenceError
from rova.ai.messages import AssistantMessage

from .models import (
    CompactionStatus,
    CompactionTrace,
    CompactionTrigger,
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


Result = TypeVar("Result")


@dataclass
class _CaptureState:
    trace: RunTrace
    started_perf_counter: float
    open_turn: TurnTrace | None = None
    open_step: StepTrace | None = None
    turn_started_perf_counters: dict[int, float] = field(default_factory=dict)
    tool_turn_indexes: dict[str, int] = field(default_factory=dict)
    active_tools: dict[str, ToolExecutionTrace] = field(default_factory=dict)
    tool_started_perf_counters: dict[str, float] = field(default_factory=dict)
    tool_step_indexes: dict[str, int] = field(default_factory=dict)


class TraceRecorder:
    """Build in-memory finalized Run/Turn/Tool traces from Agent events."""

    def __init__(self) -> None:
        self.traces: list[RunTrace] = []
        self.last_trace: RunTrace | None = None
        self._state: _CaptureState | None = None

    async def capture_run(
        self,
        agent: Agent,
        operation: Callable[[], Awaitable[Result]],
        *,
        session_id: str | None = None,
    ) -> tuple[Result, RunTrace]:
        if self._state is not None:
            raise RuntimeError("TraceRecorder already has an active capture")
        trace = RunTrace(
            run_id=uuid4().hex,
            started_at=_now(),
            session_id=session_id,
        )
        state = _CaptureState(trace, time.perf_counter())
        self._state = state
        self.traces.append(trace)
        self.last_trace = trace
        unsubscribe = agent.subscribe(self._on_agent_event)
        try:
            with agent.bind_tool_output_scope(run_id=trace.run_id):
                result = await operation()
        except asyncio.CancelledError as error:
            self._finish(state, RunStatus.ABORTED, TerminationReason.ABORTED, error)
            raise
        except Exception as error:
            if isinstance(error, SessionPersistenceError):
                status = RunStatus.SESSION_PERSISTENCE_ERROR
                reason = TerminationReason.SESSION_PERSISTENCE_ERROR
            elif state.trace.termination_reason is TerminationReason.PROVIDER_ERROR:
                status = RunStatus.PROVIDER_ERROR
                reason = TerminationReason.PROVIDER_ERROR
            else:
                status = RunStatus.HARNESS_ERROR
                reason = TerminationReason.HARNESS_ERROR
            self._finish(state, status, reason, error)
            raise
        else:
            self._finish(state, self._normal_status(trace), trace.termination_reason or TerminationReason.FINAL_RESPONSE, None)
            return result, trace
        finally:
            unsubscribe()
            self._state = None

    def _on_agent_event(self, event: AgentEvent) -> None:
        state = self._state
        if state is None:
            return
        if event.type == "turn_start":
            self._start_turn(state)
        elif event.type == "provider_context_estimated":
            self._record_provider_context_estimate(state, event)
        elif event.type == "message_end" and isinstance(event.message, AssistantMessage):
            self._finish_turn(state, event.message)
            self._record_step_assistant_message(state, event.message)
        elif event.type == "turn_end" and isinstance(event.message, AssistantMessage):
            self._finish_step(state)
        elif event.type == "message_end" and event.tool_call_id is not None:
            self._commit_tool_result(state, event)
        elif event.type == "tool_execution_start":
            self._start_tool(state, event)
        elif event.type == "tool_execution_end":
            self._finish_tool(state, event)
        elif event.type == "tool_execution_state":
            self._observe_execution_state(state, event)
        elif event.type.startswith("compaction_"):
            self._on_compaction_agent_event(state, event)
        elif event.type == "agent_end" and event.message is not None:
            state.trace.final_message = copy.deepcopy(event.message)
            state.trace.termination_reason = _termination_reason(event.termination_reason)
        elif event.type == "provider_error":
            state.trace.termination_reason = TerminationReason.PROVIDER_ERROR
            if event.error_type is not None:
                state.trace.error = TraceError(event.error_type, event.error_message or "")


    @staticmethod
    def _on_compaction_agent_event(state: _CaptureState, event: AgentEvent) -> None:
        metadata = event.metadata or {}
        trigger_value = metadata.get("trigger")
        if trigger_value not in {item.value for item in CompactionTrigger}:
            return
        trigger = CompactionTrigger(trigger_value)
        first_kept_entry_id = metadata.get("first_kept_entry_id")
        if first_kept_entry_id is not None and not isinstance(first_kept_entry_id, str):
            return
        if event.type == "compaction_started":
            state.trace.compactions.append(CompactionTrace(
                started_at=_now(),
                trigger=trigger,
                first_kept_entry_id=first_kept_entry_id,
                pressure_before=_optional_int_metadata(metadata.get("pressure_before")),
                before_estimated_tokens=_optional_int_metadata(metadata.get("pressure_before")),
                context_window=_optional_int_metadata(metadata.get("context_window")),
                reserve_tokens=_optional_int_metadata(metadata.get("reserve_tokens")),
                kept_recent_estimated_tokens=_optional_int_metadata(metadata.get("kept_recent_estimated_tokens")),
            ))
            return
        compaction = _find_compaction(state.trace.compactions, trigger, first_kept_entry_id)
        if compaction is None:
            return
        pressure_after = _optional_int_metadata(metadata.get("pressure_after"))
        if pressure_after is not None:
            compaction.pressure_after = pressure_after
            compaction.after_estimated_tokens = pressure_after
        if event.type == "compaction_completed":
            compaction.ended_at = _now()
            compaction.duration_ms = _elapsed_ms_from_dates(compaction.started_at, compaction.ended_at)
            compaction.status = CompactionStatus.COMPLETED
            compaction.summary_size_chars = _optional_int_metadata(metadata.get("summary_size_chars"))
        elif event.type == "compaction_failed":
            compaction.ended_at = _now()
            compaction.duration_ms = _elapsed_ms_from_dates(compaction.started_at, compaction.ended_at)
            compaction.status = CompactionStatus.FAILED
            error_type = metadata.get("error_type")
            if isinstance(error_type, str):
                compaction.error = TraceError(error_type, str(metadata.get("error_message") or ""))
        elif event.type == "compaction_warning":
            compaction.status = CompactionStatus.WARNING
        elif event.type == "compaction_warning":
            compaction.status = CompactionStatus.WARNING
            compaction.error = _event_error(event)

    @staticmethod
    def _start_turn(state: _CaptureState) -> None:
        turn = TurnTrace(turn_index=len(state.trace.turns) + 1, started_at=_now())
        state.trace.turns.append(turn)
        state.open_turn = turn
        state.turn_started_perf_counters[turn.turn_index] = time.perf_counter()
        step = StepTrace(step_index=len(state.trace.steps) + 1, started_at=turn.started_at)
        state.trace.steps.append(step)
        state.open_step = step

    @staticmethod
    def _finish_turn(state: _CaptureState, message) -> None:
        turn = state.open_turn
        if turn is None:
            return
        turn.ended_at = _now()
        turn.duration_ms = _elapsed_ms(state.turn_started_perf_counters[turn.turn_index])
        turn.assistant_message = copy.deepcopy(message)
        turn.stop_reason = message.stop_reason
        turn.usage = message.usage
        turn.tool_call_ids = [tool_call.id for tool_call in message.tool_calls]
        for tool_call_id in turn.tool_call_ids:
            state.tool_turn_indexes[tool_call_id] = turn.turn_index
        state.open_turn = None

    @staticmethod
    def _record_step_assistant_message(state: _CaptureState, message: AssistantMessage) -> None:
        step = state.open_step
        if step is None:
            return
        step.assistant_message = copy.deepcopy(message)
        step.usage = StepUsageTrace(
            estimated_input_tokens=step.usage.estimated_input_tokens,
            actual_usage=message.usage,
        )
        step.tool_calls = [
            ToolCallTrace(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=copy.deepcopy(tool_call.arguments),
                batch_id=None,
                call_index=call_index,
                batch_mode=None,
                execution_mode=None,
            )
            for call_index, tool_call in enumerate(message.tool_calls)
        ]
        for tool_call in step.tool_calls:
            state.tool_step_indexes[tool_call.tool_call_id] = step.step_index

    @staticmethod
    def _finish_step(state: _CaptureState) -> None:
        step = state.open_step
        if step is None:
            return
        step.ended_at = _now()
        step.duration_ms = _elapsed_ms_from_dates(step.started_at, step.ended_at)
        state.open_step = None

    @staticmethod
    def _record_provider_context_estimate(state: _CaptureState, event: AgentEvent) -> None:
        step = state.open_step
        value = _optional_int_metadata((event.metadata or {}).get("estimated_input_tokens"))
        if step is None or value is None:
            return
        step.usage.estimated_input_tokens = value
        previous = state.trace.peak_estimated_context_tokens
        state.trace.peak_estimated_context_tokens = value if previous is None else max(previous, value)

    @staticmethod
    def _start_tool(state: _CaptureState, event: AgentEvent) -> None:
        if event.tool_call_id is None or event.tool_name is None or event.args is None:
            return
        execution = ToolExecutionTrace(
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            arguments=copy.deepcopy(event.args),
            turn_index=state.tool_turn_indexes.get(event.tool_call_id),
            started_at=_now(),
        )
        state.trace.tool_executions.append(execution)
        state.active_tools[event.tool_call_id] = execution
        state.tool_started_perf_counters[event.tool_call_id] = time.perf_counter()
        tool_call = _tool_call_trace(state, event.tool_call_id)
        if tool_call is not None:
            tool_call.executed = True
            tool_call.started_at = _now()
            tool_call.batch_id = event.batch_id
            tool_call.call_index = event.call_index if event.call_index is not None else tool_call.call_index
            tool_call.batch_mode = event.batch_mode
            tool_call.execution_mode = event.execution_mode
            tool_call.arguments = copy.deepcopy(event.args)

    @staticmethod
    def _finish_tool(state: _CaptureState, event: AgentEvent) -> None:
        if event.tool_call_id is None:
            return
        execution = state.active_tools.pop(event.tool_call_id, None)
        started = state.tool_started_perf_counters.pop(event.tool_call_id, None)
        if execution is None or started is None:
            return
        execution.ended_at = _now()
        execution.duration_ms = _elapsed_ms(started)
        execution.result = event.result
        execution.is_error = event.is_error
        execution.metadata = copy.deepcopy(event.metadata or {})
        metadata = execution.metadata
        outcome = metadata.get("outcome")
        execution.outcome = ToolOutcome(outcome) if outcome in {item.value for item in ToolOutcome} else None
        execution.policy_decision = metadata.get("policy_decision") if isinstance(metadata.get("policy_decision"), str) else None
        execution.policy_reason = metadata.get("policy_reason") if isinstance(metadata.get("policy_reason"), str) else None
        execution.approval_required = metadata.get("approval_required") if isinstance(metadata.get("approval_required"), bool) else None
        execution.approval_decision = metadata.get("approval_decision") if isinstance(metadata.get("approval_decision"), str) else None
        execution.command = metadata.get("command") if isinstance(metadata.get("command"), str) else None
        execution.exit_code = metadata.get("exit_code") if isinstance(metadata.get("exit_code"), int) and not isinstance(metadata.get("exit_code"), bool) else None
        execution.timed_out = metadata.get("timed_out") if isinstance(metadata.get("timed_out"), bool) else None
        tool_call = _tool_call_trace(state, event.tool_call_id)
        if tool_call is not None:
            tool_call.executed = True
            tool_call.ended_at = _now()
            tool_call.duration_ms = _elapsed_ms_from_dates(tool_call.started_at, tool_call.ended_at) if tool_call.started_at else None
            tool_call.batch_id = event.batch_id
            tool_call.call_index = event.call_index if event.call_index is not None else tool_call.call_index
            tool_call.batch_mode = event.batch_mode
            tool_call.execution_mode = event.execution_mode
            _apply_tool_metadata(tool_call, metadata)
            tool_call.result = _tool_result_trace(event.result or "", metadata)

    @staticmethod
    def _observe_execution_state(state: _CaptureState, event: AgentEvent) -> None:
        if event.tool_call_id is None or event.execution_state not in {"executor_completed", "completed", "cancelled", "interrupted"}:
            return
        tool_call = _tool_call_trace(state, event.tool_call_id)
        if tool_call is None:
            return
        if event.execution_state in {"executor_completed", "completed"}:
            tool_call.executed = True
        tool_call.ended_at = tool_call.ended_at or _now()
        tool_call.duration_ms = (
            _elapsed_ms_from_dates(tool_call.started_at, tool_call.ended_at)
            if tool_call.started_at and tool_call.ended_at
            else None
        )
        tool_call.batch_id = event.batch_id or tool_call.batch_id
        tool_call.call_index = event.call_index if event.call_index is not None else tool_call.call_index
        tool_call.batch_mode = event.batch_mode or tool_call.batch_mode
        tool_call.execution_mode = event.execution_mode or tool_call.execution_mode
        if event.outcome is not None:
            _apply_tool_metadata(tool_call, {"outcome": event.outcome})

    @staticmethod
    def _commit_tool_result(state: _CaptureState, event: AgentEvent) -> None:
        if event.message is None or not hasattr(event.message, "tool_call_id"):
            return
        tool_call = _tool_call_trace(state, event.tool_call_id)
        if tool_call is None:
            return
        metadata = event.metadata or {}
        tool_call.committed = True
        tool_call.batch_id = event.batch_id or tool_call.batch_id
        tool_call.call_index = event.call_index if event.call_index is not None else tool_call.call_index
        tool_call.batch_mode = event.batch_mode or tool_call.batch_mode
        tool_call.execution_mode = event.execution_mode or tool_call.execution_mode
        _apply_tool_metadata(tool_call, metadata)
        tool_call.result = _tool_result_trace(event.message.text, metadata)

    @staticmethod
    def _normal_status(trace: RunTrace) -> RunStatus:
        if trace.termination_reason is TerminationReason.ABORTED:
            return RunStatus.ABORTED
        if trace.termination_reason is TerminationReason.PROVIDER_ERROR:
            return RunStatus.PROVIDER_ERROR
        if trace.termination_reason is TerminationReason.CONTEXT_OVERFLOW:
            return RunStatus.CONTEXT_OVERFLOW
        return RunStatus.COMPLETED

    @staticmethod
    def _finish(state: _CaptureState, status: RunStatus, reason: TerminationReason, error: BaseException | None) -> None:
        trace = state.trace
        if trace.status is not RunStatus.RUNNING:
            return
        trace.ended_at = _now()
        trace.duration_ms = _elapsed_ms(state.started_perf_counter)
        trace.status = status
        trace.termination_reason = reason
        trace.usage = _aggregate_usage(trace.turns)
        step_usages = [step.usage.actual_usage for step in trace.steps]
        trace.actual_usage_complete = bool(step_usages) and all(usage is not None for usage in step_usages)
        trace.actual_usage = _aggregate_usage_values(step_usages) if trace.actual_usage_complete else None
        if error is not None:
            trace.error = TraceError(type(error).__name__, str(error))


def _aggregate_usage(turns: list[TurnTrace]):
    return _aggregate_usage_values([turn.usage for turn in turns])


def _aggregate_usage_values(usages):
    if not usages or any(usage is None for usage in usages):
        return None
    first = next(usage for usage in usages if usage is not None)
    return type(first)(
        input_tokens=sum(usage.input_tokens for usage in usages if usage is not None),
        output_tokens=sum(usage.output_tokens for usage in usages if usage is not None),
        total_tokens=sum(usage.total_tokens for usage in usages if usage is not None),
    )


def _tool_call_trace(state: _CaptureState, tool_call_id: str) -> ToolCallTrace | None:
    step_index = state.tool_step_indexes.get(tool_call_id)
    if step_index is None or step_index < 1 or step_index > len(state.trace.steps):
        return None
    for tool_call in state.trace.steps[step_index - 1].tool_calls:
        if tool_call.tool_call_id == tool_call_id:
            return tool_call
    return None


def _apply_tool_metadata(tool_call: ToolCallTrace, metadata: dict) -> None:
    tool_call.diagnostics = copy.deepcopy(metadata)
    outcome = metadata.get("outcome")
    tool_call.outcome = ToolOutcome(outcome) if outcome in {item.value for item in ToolOutcome} else None
    tool_call.failure_stage = metadata.get("failure_stage") if isinstance(metadata.get("failure_stage"), str) else None
    tool_call.policy_decision = metadata.get("policy_decision") if isinstance(metadata.get("policy_decision"), str) else None
    tool_call.policy_reason = metadata.get("policy_reason") if isinstance(metadata.get("policy_reason"), str) else None
    tool_call.approval_required = metadata.get("approval_required") if isinstance(metadata.get("approval_required"), bool) else None
    tool_call.approval_decision = metadata.get("approval_decision") if isinstance(metadata.get("approval_decision"), str) else None
    tool_call.exit_code = metadata.get("exit_code") if isinstance(metadata.get("exit_code"), int) and not isinstance(metadata.get("exit_code"), bool) else None
    tool_call.timed_out = metadata.get("timed_out") if isinstance(metadata.get("timed_out"), bool) else None


def _tool_result_trace(content: str, metadata: dict) -> ToolResultTrace:
    output = metadata.get("tool_output")
    if not isinstance(output, dict):
        return ToolResultTrace(content, False, None, len(content), False, len(content))
    artifact = output.get("artifact")
    artifact_ref = artifact.get("artifact_id") if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str) else None
    original_size_chars = output.get("original_size_chars")
    if not isinstance(original_size_chars, int) or isinstance(original_size_chars, bool):
        original_size_chars = len(content)
    preview_truncated = output.get("preview_truncated")
    if not isinstance(preview_truncated, bool):
        preview_truncated = bool(output.get("truncated", False))
    preview_size_chars = output.get("preview_size_chars")
    if not isinstance(preview_size_chars, int) or isinstance(preview_size_chars, bool):
        preview_size_chars = len(content)
    externalized = output.get("externalized")
    if not isinstance(externalized, bool):
        externalized = artifact_ref is not None
    return ToolResultTrace(content, externalized, artifact_ref, original_size_chars, preview_truncated, preview_size_chars)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed_ms(started_perf_counter: float) -> float:
    return max(0.0, (time.perf_counter() - started_perf_counter) * 1000)


def _elapsed_ms_from_dates(started_at: datetime, ended_at: datetime) -> float:
    return max(0.0, (ended_at - started_at).total_seconds() * 1000)


def _termination_reason(value: AgentTerminationReason | None) -> TerminationReason | None:
    if value is None:
        return None
    return TerminationReason(value.value)


def _find_compaction(
    compactions: list[CompactionTrace],
    trigger: CompactionTrigger,
    first_kept_entry_id: str | None,
) -> CompactionTrace | None:
    for compaction in reversed(compactions):
        if compaction.trigger is trigger and compaction.first_kept_entry_id == first_kept_entry_id:
            return compaction
    return None


def _optional_int_metadata(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
