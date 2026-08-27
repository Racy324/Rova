from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, TypeVar
from uuid import uuid4

from rova.agent_core.agent import Agent
from rova.agent_core.events import AgentEvent, AgentTerminationReason
from rova.agent_session.agent_session import SessionPersistenceError
from rova.agent_session.events import SessionMaintenanceEvent

from .models import (
    CompactionStatus,
    CompactionTrace,
    CompactionTrigger,
    MemoryMaintenanceKind,
    MemoryMaintenanceStatus,
    MemoryMaintenanceTrace,
    RunStatus,
    RunTrace,
    TerminationReason,
    ToolExecutionTrace,
    ToolOutcome,
    TraceError,
    TurnTrace,
)


Result = TypeVar("Result")


class SessionObservationSource(Protocol):
    @property
    def session_id(self) -> str | None: ...

    def subscribe_maintenance(self, listener): ...


class MemoryObservationSource(Protocol):
    def subscribe_memory(self, listener): ...


@dataclass
class _CaptureState:
    trace: RunTrace
    started_perf_counter: float
    open_turn: TurnTrace | None = None
    turn_started_perf_counters: dict[int, float] = field(default_factory=dict)
    tool_turn_indexes: dict[str, int] = field(default_factory=dict)
    active_tools: dict[str, ToolExecutionTrace] = field(default_factory=dict)
    tool_started_perf_counters: dict[str, float] = field(default_factory=dict)


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
        session: SessionObservationSource | None = None,
        memory: MemoryObservationSource | None = None,
    ) -> tuple[Result, RunTrace]:
        if self._state is not None:
            raise RuntimeError("TraceRecorder already has an active capture")
        trace = RunTrace(
            run_id=uuid4().hex,
            started_at=_now(),
            session_id=session_id if session_id is not None else (session.session_id if session is not None else None),
        )
        state = _CaptureState(trace, time.perf_counter())
        self._state = state
        self.traces.append(trace)
        self.last_trace = trace
        unsubscribe = agent.subscribe(self._on_agent_event)
        unsubscribe_maintenance = session.subscribe_maintenance(self._on_session_maintenance_event) if session is not None else None
        unsubscribe_memory = memory.subscribe_memory(self._on_memory_observation) if memory is not None else None
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
            if unsubscribe_maintenance is not None:
                unsubscribe_maintenance()
            if unsubscribe_memory is not None:
                unsubscribe_memory()
            self._state = None

    def _on_memory_observation(self, event) -> None:
        state = self._state
        if state is None:
            return
        try:
            kind = MemoryMaintenanceKind(event.kind)
            status = MemoryMaintenanceStatus(event.status)
        except (AttributeError, ValueError):
            return
        error = None
        error_message = getattr(event, "error_message", None)
        if isinstance(error_message, str) and error_message:
            error = TraceError("MemoryMaintenanceError", error_message[:200])
        changed_documents = getattr(event, "changed_documents", ())
        safe_documents = [
            item for item in changed_documents
            if isinstance(item, str) and item in {"USER.md", "MEMORY.md"}
        ]
        state.trace.memory_events.append(MemoryMaintenanceTrace(kind, status, safe_documents, error))

    def _on_agent_event(self, event: AgentEvent) -> None:
        state = self._state
        if state is None:
            return
        if event.type == "turn_start":
            self._start_turn(state)
        elif event.type == "message_end" and event.message is not None:
            self._finish_turn(state, event.message)
        elif event.type == "tool_execution_start":
            self._start_tool(state, event)
        elif event.type == "tool_execution_end":
            self._finish_tool(state, event)
        elif event.type == "agent_end" and event.message is not None:
            state.trace.final_message = copy.deepcopy(event.message)
            state.trace.termination_reason = _termination_reason(event.termination_reason)
        elif event.type == "provider_error":
            state.trace.termination_reason = TerminationReason.PROVIDER_ERROR
            if event.error_type is not None:
                state.trace.error = TraceError(event.error_type, event.error_message or "")

    def _on_session_maintenance_event(self, event: SessionMaintenanceEvent) -> None:
        state = self._state
        if state is None:
            return
        trigger = CompactionTrigger(event.trigger)
        if event.type == "compaction_started":
            state.trace.compactions.append(
                CompactionTrace(
                    started_at=_now(),
                    trigger=trigger,
                    first_kept_entry_id=event.first_kept_entry_id,
                    pressure_before=event.pressure_before,
                )
            )
            return
        compaction = _find_compaction(state.trace.compactions, trigger, event.first_kept_entry_id)
        if compaction is None:
            return
        if event.pressure_after is not None:
            compaction.pressure_after = event.pressure_after
        if event.type == "compaction_completed":
            compaction.ended_at = _now()
            compaction.duration_ms = _elapsed_ms_from_dates(compaction.started_at, compaction.ended_at)
            compaction.status = CompactionStatus.COMPLETED
        elif event.type == "compaction_failed":
            compaction.ended_at = _now()
            compaction.duration_ms = _elapsed_ms_from_dates(compaction.started_at, compaction.ended_at)
            compaction.status = CompactionStatus.FAILED
            compaction.error = _event_error(event)
        elif event.type == "compaction_warning":
            compaction.status = CompactionStatus.WARNING
            compaction.error = _event_error(event)

    @staticmethod
    def _start_turn(state: _CaptureState) -> None:
        turn = TurnTrace(turn_index=len(state.trace.turns) + 1, started_at=_now())
        state.trace.turns.append(turn)
        state.open_turn = turn
        state.turn_started_perf_counters[turn.turn_index] = time.perf_counter()

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

    @staticmethod
    def _normal_status(trace: RunTrace) -> RunStatus:
        if trace.termination_reason is TerminationReason.ABORTED:
            return RunStatus.ABORTED
        if trace.termination_reason is TerminationReason.PROVIDER_ERROR:
            return RunStatus.PROVIDER_ERROR
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
        if error is not None:
            trace.error = TraceError(type(error).__name__, str(error))


def _aggregate_usage(turns: list[TurnTrace]):
    if not turns or any(turn.usage is None for turn in turns):
        return None
    return type(turns[0].usage)(
        input_tokens=sum(turn.usage.input_tokens for turn in turns if turn.usage is not None),
        output_tokens=sum(turn.usage.output_tokens for turn in turns if turn.usage is not None),
        total_tokens=sum(turn.usage.total_tokens for turn in turns if turn.usage is not None),
    )


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


def _event_error(event: SessionMaintenanceEvent) -> TraceError | None:
    if event.error_type is None:
        return None
    return TraceError(event.error_type, event.error_message or "")
