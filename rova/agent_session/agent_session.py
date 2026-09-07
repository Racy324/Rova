from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rova.ai.messages import AssistantMessage, Message, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.context import Context
from rova.agent_core.agent import Agent
from rova.agent_core.events import AgentEvent

from .compaction import (
    CompactionPlan,
    CompactionPolicy,
    ConservativeTokenEstimator,
    ContextPressure,
    TokenEstimator,
    estimate_compaction_summary_input,
    build_compaction_summarization_request,
    estimate_message_tokens,
    find_compaction_plan,
    generate_compaction_summary,
    plan_compaction_at,
    resolve_current_run_pressure,
    should_compact,
    summary_max_tokens,
    validate_compaction_policy,
)
from .context_builder import build_session_messages, build_session_projection
from .events import SessionMaintenanceEvent
from .execution_journal import ExecutionJournalError, JournalRecord, ToolExecutionJournal
from .session_store import CompactionEntry, DurableSession, JsonlSessionStore, SessionStoreError
from .summarization import SUMMARIZATION_SYSTEM_PROMPT, SummaryFn, summarize_with_stream


class SessionPersistenceError(RuntimeError):
    pass


class SessionIncompleteError(RuntimeError):
    pass


class SessionBranchError(RuntimeError):
    pass


class CompactionError(RuntimeError):
    pass


class CompactionInputTooLarge(CompactionError):
    pass


class CompactionHeadroomWarning(CompactionError):
    pass


class PreRunContextTooLarge(CompactionError):
    pass


class SessionRecoveryError(SessionPersistenceError):
    pass


@dataclass(frozen=True)
class RecoveryItem:
    tool_name: str
    call_index: int
    outcome: str
    side_effects_unknown: bool


@dataclass(frozen=True)
class RecoveryReport:
    items: tuple[RecoveryItem, ...] = ()

    @property
    def recovered_count(self) -> int:
        return len(self.items)


_DURABLE_BINDINGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class AgentSession:
    def __init__(
        self,
        agent: Agent,
        durable_session: DurableSession | None = None,
        *,
        compaction_policy: CompactionPolicy | None = None,
        token_estimator: TokenEstimator | None = None,
        summary_fn: SummaryFn | None = None,
        provider_context_estimator: Callable[[Context], int] | None = None,
    ) -> None:
        self.agent = agent
        self._durable_session = durable_session
        self._compaction_policy = compaction_policy
        if compaction_policy is not None:
            validate_compaction_policy(agent.model, compaction_policy)
        self._token_estimator = token_estimator or ConservativeTokenEstimator()
        self._summary_fn = summary_fn or self._summarize_with_agent_stream
        self._provider_context_estimator = provider_context_estimator
        self.persisted_message_count = len(agent.messages) if durable_session is not None else 0
        self._faulted = False
        self._incomplete_tail = False
        self._closed = False
        self._prompt_active = False
        self.last_maintenance_error: Exception | None = None
        self.last_prompt_input_entry_id: str | None = None
        self._maintenance_listeners: list[Callable[[SessionMaintenanceEvent], None]] = []
        self._unsubscribe: Callable[[], None] | None = None
        self._execution_journal: ToolExecutionJournal | None = None
        self._assistant_entry_ids: dict[str, str] = {}
        self.recovery_report = RecoveryReport()
        if durable_session is not None:
            _ensure_agent_has_no_durable_session(agent)
            self._unsubscribe = self.agent.subscribe(self._on_agent_event)
            _DURABLE_BINDINGS[agent] = weakref.ref(self)
            self._execution_journal = ToolExecutionJournal(durable_session.store.root, durable_session.session_id)
            self._refresh_assistant_entry_ids()

    @classmethod
    def create(
        cls,
        agent: Agent,
        *,
        session_root: Path | None = None,
        compaction_policy: CompactionPolicy | None = None,
        token_estimator: TokenEstimator | None = None,
        summary_fn: SummaryFn | None = None,
        provider_context_estimator: Callable[[Context], int] | None = None,
    ) -> AgentSession:
        if agent.messages:
            raise ValueError("a new durable session requires an empty Agent; use load() to restore history")
        _ensure_agent_has_no_durable_session(agent)
        durable_session = JsonlSessionStore(session_root).create()
        return cls(
            agent,
            durable_session,
            compaction_policy=compaction_policy,
            token_estimator=token_estimator,
            summary_fn=summary_fn,
            provider_context_estimator=provider_context_estimator,
        )

    @classmethod
    def load(
        cls,
        agent: Agent,
        session_id: str,
        *,
        session_root: Path | None = None,
        leaf_id: str | None = None,
        compaction_policy: CompactionPolicy | None = None,
        token_estimator: TokenEstimator | None = None,
        summary_fn: SummaryFn | None = None,
        provider_context_estimator: Callable[[Context], int] | None = None,
    ) -> AgentSession:
        if agent.messages:
            raise ValueError("load requires a fresh Agent with no runtime history")
        _ensure_agent_has_no_durable_session(agent)
        durable_session = JsonlSessionStore(session_root).load(session_id)
        if leaf_id is not None:
            durable_session.branch(leaf_id)
        session = cls(
            agent,
            durable_session,
            compaction_policy=compaction_policy,
            token_estimator=token_estimator,
            summary_fn=summary_fn,
            provider_context_estimator=provider_context_estimator,
        )
        try:
            session._reconcile_selected_branch()
        except Exception:
            session.close()
            raise
        return session

    @property
    def session_id(self) -> str | None:
        return self._durable_session.session_id if self._durable_session is not None else None

    @property
    def faulted(self) -> bool:
        return self._faulted

    def close(self) -> None:
        if self._closed:
            return
        if self._prompt_active:
            raise RuntimeError("cannot close while prompt is active")
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        binding = _DURABLE_BINDINGS.get(self.agent)
        if binding is not None and binding() is self:
            del _DURABLE_BINDINGS[self.agent]
        self._closed = True

    def subscribe_maintenance(self, listener: Callable[[SessionMaintenanceEvent], None]) -> Callable[[], None]:
        """Subscribe to content-free compaction observations for this session."""
        self._maintenance_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._maintenance_listeners:
                self._maintenance_listeners.remove(listener)

        return unsubscribe

    def branch(self, entry_id: str) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._faulted:
            raise SessionPersistenceError("session is faulted after a durable persistence failure")
        if self._prompt_active:
            raise SessionBranchError("cannot branch while prompt is active")
        if self._durable_session is None:
            raise SessionBranchError("branch requires a durable session")
        if entry_id not in self._durable_session.by_id:
            raise SessionBranchError(f"unknown branch entry: {entry_id}")
        previous_leaf_id = self._durable_session.leaf_id
        self._durable_session.branch(entry_id)
        try:
            self._reconcile_selected_branch()
        except Exception:
            self._durable_session.branch(previous_leaf_id) if previous_leaf_id is not None else None
            raise

    async def prompt(self, user_text: str) -> list[AssistantMessage]:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._faulted:
            raise SessionPersistenceError("session is faulted after a durable persistence failure")
        if self._prompt_active:
            raise SessionBranchError("prompt is already active")
        if self._incomplete_tail:
            raise SessionIncompleteError("session has an incomplete tool-call tail and cannot start a new provider run")
        self.last_maintenance_error = None
        self.last_prompt_input_entry_id = None
        self._prompt_active = True
        try:
            with self.agent.bind_tool_output_scope(session_id=self.session_id):
                if self._durable_session is None:
                    return await self.agent.run([UserMessage(user_text)])

                user_message = UserMessage(user_text)
                self._persist_user_message(user_message)
                if self._provider_context_estimator is None:
                    self._ensure_pre_run_context_fits()
                    current_run_start_index = len(self.agent.messages)
                assistant_messages = await self.agent.run([])
                if self._provider_context_estimator is None:
                    await self._maybe_compact_after_run(current_run_start_index)
                return assistant_messages
        finally:
            self._prompt_active = False

    async def prepare_provider_context(
        self,
        context: Context,
        *,
        rebuild_context: Callable[[], Context] | None = None,
        force_compaction: bool = False,
        trigger: Literal["proactive", "overflow_recovery"] = "proactive",
    ) -> Context:
        """Record the full provider-ready estimate before one model step.

        Compaction policy is added at this same narrow seam; the estimate is
        already based on the product's canonical Provider payload.
        """
        if self._provider_context_estimator is None:
            return context
        estimate = self._provider_context_estimator(context)
        if not isinstance(estimate, int) or isinstance(estimate, bool) or estimate < 0:
            raise ValueError("provider_context_estimator must return a non-negative integer")
        await self.agent.emit_runtime_event(AgentEvent(
            "provider_context_estimated",
            metadata={"estimated_input_tokens": estimate, "source": "estimated"},
        ))
        if self._durable_session is None or self._compaction_policy is None or self.agent.model.context_window is None:
            return context
        target = self.agent.model.context_window - self._compaction_policy.reserve_tokens
        fixed_context = Context(context.system_prompt, [], list(context.tools))
        fixed_estimate = self._provider_context_estimator(fixed_context)
        if fixed_estimate > target:
            raise PreRunContextTooLarge(
                "non-compactable provider context exceeds the context window reserve threshold"
            )
        if estimate <= target and not force_compaction:
            return context
        if rebuild_context is None:
            raise CompactionError("proactive compaction requires a provider context rebuild callback")
        summary_budget = self._summary_token_limit() or 0
        retained_budget = min(
            self._compaction_policy.keep_recent_tokens,
            max(0, target - fixed_estimate - summary_budget),
        )
        if retained_budget <= 0:
            raise PreRunContextTooLarge(
                "no compactable conversation budget remains after fixed provider context and summary reserve"
            )
        projection = build_session_projection(self._durable_session.path_to_leaf())
        plan = find_compaction_plan(
            projection,
            retained_token_budget=retained_budget,
            token_estimator=self._token_estimator,
        )
        if plan is None:
            raise PreRunContextTooLarge(
                "provider context exceeds the threshold but conversation has no safe compactable boundary"
            )
        await self._execute_compaction_plan(
            plan,
            trigger=trigger,
            pressure_before=estimate,
            context_window=self.agent.model.context_window,
            reserve_tokens=self._compaction_policy.reserve_tokens,
        )
        rebuilt = rebuild_context()
        after_estimate = self._provider_context_estimator(rebuilt)
        await self.agent.emit_runtime_event(AgentEvent(
            "provider_context_estimated",
            metadata={"estimated_input_tokens": after_estimate, "source": "estimated"},
        ))
        if after_estimate > target:
            raise CompactionHeadroomWarning(
                "compaction completed but full provider context still exceeds the context window reserve threshold"
            )
        return rebuilt

    def _persist_user_message(self, message: UserMessage) -> None:
        assert self._durable_session is not None
        try:
            entry_id = self._durable_session.append(message)
        except SessionStoreError as error:
            self._fault(error)
        self.agent.messages.append(message)
        self.persisted_message_count += 1
        self.last_prompt_input_entry_id = entry_id

    def _on_agent_event(self, event: AgentEvent) -> None:
        if event.type == "tool_execution_state":
            assert self._execution_journal is not None
            try:
                assistant_entry_id = self._assistant_entry_ids.get(event.tool_call_id or "")
                if assistant_entry_id is None:
                    raise SessionStoreError("tool execution state has no durable Assistant ToolCall entry")
                self._execution_journal.append(event, assistant_entry_id=assistant_entry_id)
            except (OSError, ValueError, SessionStoreError) as error:
                self._fault(SessionStoreError(f"failed to append execution journal: {error}"))
        if event.type == "message_end":
            self._persist_committed_suffix()

    def _persist_committed_suffix(self) -> None:
        assert self._durable_session is not None
        try:
            while self.persisted_message_count < len(self.agent.messages):
                message = self.agent.messages[self.persisted_message_count]
                entry_id = self._durable_session.append(message)
                if isinstance(message, AssistantMessage):
                    for tool_call in message.tool_calls:
                        self._assistant_entry_ids[tool_call.id] = entry_id
                self.persisted_message_count += 1
        except SessionStoreError as error:
            self._fault(error)

    async def compact_at(self, first_kept_entry_id: str | None) -> CompactionEntry:
        if self._closed:
            raise CompactionError("session is closed")
        if self._faulted:
            raise SessionPersistenceError("session is faulted after a durable persistence failure")
        if self._prompt_active:
            raise CompactionError("cannot compact while prompt is active")
        if self._durable_session is None:
            raise CompactionError("compaction requires a durable session")
        try:
            plan = plan_compaction_at(
                build_session_projection(self._durable_session.path_to_leaf()),
                first_kept_entry_id=first_kept_entry_id,
                token_estimator=self._token_estimator,
            )
            return await self._execute_compaction_plan(plan, trigger="manual")
        except SessionPersistenceError:
            raise
        except Exception as error:
            raise CompactionError(f"manual compaction failed: {error}") from error

    async def _maybe_compact_after_run(self, current_run_start_index: int) -> None:
        if self._durable_session is None or self._compaction_policy is None:
            return
        if self.agent.model.context_window is None:
            return
        try:
            pressure = resolve_current_run_pressure(
                self.agent.messages,
                current_run_start_index=current_run_start_index,
                token_estimator=self._token_estimator,
            )
            if not should_compact(pressure=pressure, model=self.agent.model, policy=self._compaction_policy):
                return
            projection = build_session_projection(self._durable_session.path_to_leaf())
            plan = find_compaction_plan(
                projection,
                retained_token_budget=self._compaction_policy.keep_recent_tokens,
                token_estimator=self._token_estimator,
            )
            if plan is None:
                raise CompactionError("automatic compaction found no useful plan")
            await self._execute_compaction_plan(
                plan,
                trigger="automatic",
                pressure_before=pressure.tokens,
            )
            await self._record_post_compaction_headroom_diagnostic(plan)
        except SessionPersistenceError:
            raise
        except Exception as error:
            self.last_maintenance_error = error

    async def _execute_compaction_plan(
        self,
        plan: CompactionPlan,
        *,
        trigger: Literal["automatic", "proactive", "overflow_recovery", "manual"],
        pressure_before: int | None = None,
        context_window: int | None = None,
        reserve_tokens: int | None = None,
    ) -> CompactionEntry:
        assert self._durable_session is not None
        await self._emit_maintenance(
            SessionMaintenanceEvent(
                "compaction_started",
                trigger,
                plan.first_kept_entry_id,
                pressure_before=pressure_before,
                context_window=context_window,
                reserve_tokens=reserve_tokens,
                kept_recent_estimated_tokens=plan.estimated_retained_tokens,
            )
        )
        try:
            max_tokens = self._summary_token_limit()
            self._ensure_summary_input_fits(plan, max_tokens)
            summary = await generate_compaction_summary(
                historical_messages=plan.messages_to_summarize,
                turn_prefix_messages=plan.turn_prefix_messages,
                summarize=self._summary_fn,
                max_tokens=max_tokens,
            )
        except Exception as error:
            await self._emit_maintenance(
                SessionMaintenanceEvent(
                    "compaction_failed",
                    trigger,
                    plan.first_kept_entry_id,
                    pressure_before=pressure_before,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            raise
        try:
            entry = self._durable_session.append_compaction(summary, plan.first_kept_entry_id)
        except SessionStoreError as error:
            await self._emit_maintenance(
                SessionMaintenanceEvent(
                    "compaction_failed",
                    trigger,
                    plan.first_kept_entry_id,
                    pressure_before=pressure_before,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            self._fault_compaction(error)
        try:
            messages = build_session_messages(self._durable_session.path_to_leaf())
            self.agent.messages[:] = messages
            self.persisted_message_count = len(messages)
            self._incomplete_tail = _has_incomplete_tool_calls(self.agent.messages)
        except Exception as error:
            await self._emit_maintenance(
                SessionMaintenanceEvent(
                    "compaction_failed",
                    trigger,
                    plan.first_kept_entry_id,
                    pressure_before=pressure_before,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            self._fault_compaction(error)
        await self._emit_maintenance(
            SessionMaintenanceEvent(
                "compaction_completed",
                trigger,
                plan.first_kept_entry_id,
                pressure_before=pressure_before,
                context_window=context_window,
                reserve_tokens=reserve_tokens,
                kept_recent_estimated_tokens=plan.estimated_retained_tokens,
                summary_size_chars=len(summary),
            )
        )
        return entry

    def _ensure_pre_run_context_fits(self) -> None:
        """Reject an obviously oversized new-user context before its first Provider call.

        Tool results arrive inside Agent.run and therefore need separate per-tool-output
        limits; post-run compaction cannot protect that in-flight Provider turn.
        """
        context_window = self.agent.model.context_window
        if context_window is None:
            return
        estimated_tokens = estimate_message_tokens(self._token_estimator, self.agent.messages)
        if estimated_tokens > context_window:
            raise PreRunContextTooLarge(
                "prospective runtime context is too large before provider invocation"
            )

    def _ensure_summary_input_fits(self, plan: CompactionPlan, max_tokens: int | None) -> None:
        context_window = self.agent.model.context_window
        if context_window is None:
            return
        if max_tokens is None:
            raise CompactionInputTooLarge(
                "compaction summary output budget is required when model context_window is configured"
            )
        if self._provider_context_estimator is not None:
            request = build_compaction_summarization_request(
                historical_messages=plan.messages_to_summarize,
                turn_prefix_messages=plan.turn_prefix_messages,
                max_tokens=max_tokens,
            )
            estimated_input = self._provider_context_estimator(Context(
                system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
                messages=[UserMessage(f"{request.instruction}\n\n{request.content}")],
                tools=[],
            ))
        else:
            estimated_input = estimate_compaction_summary_input(
                historical_messages=plan.messages_to_summarize,
                turn_prefix_messages=plan.turn_prefix_messages,
                max_tokens=max_tokens,
                token_estimator=self._token_estimator,
            )
        if estimated_input + max_tokens > context_window:
            raise CompactionInputTooLarge(
                "compaction summary input and output budget do not fit the model context window"
            )

    async def _record_post_compaction_headroom_diagnostic(self, plan: CompactionPlan) -> None:
        assert self._compaction_policy is not None
        estimated_pressure = estimate_message_tokens(self._token_estimator, self.agent.messages)
        if should_compact(
            pressure=ContextPressure(estimated_pressure, "estimated"),
            model=self.agent.model,
            policy=self._compaction_policy,
        ):
            self.last_maintenance_error = CompactionHeadroomWarning(
                "compaction completed durably but rebuilt context remains at or above the automatic threshold"
            )
            await self._emit_maintenance(
                SessionMaintenanceEvent(
                    "compaction_warning",
                    "automatic",
                    plan.first_kept_entry_id,
                    pressure_after=estimated_pressure,
                    error_type=type(self.last_maintenance_error).__name__,
                    error_message=str(self.last_maintenance_error),
                )
            )

    async def _emit_maintenance(self, event: SessionMaintenanceEvent) -> None:
        await self.agent.emit_runtime_event(AgentEvent(
            event.type,
            metadata={
                "trigger": event.trigger,
                "first_kept_entry_id": event.first_kept_entry_id,
                "pressure_before": event.pressure_before,
                "pressure_after": event.pressure_after,
                "error_type": event.error_type,
                "error_message": event.error_message,
                "context_window": event.context_window,
                "reserve_tokens": event.reserve_tokens,
                "kept_recent_estimated_tokens": event.kept_recent_estimated_tokens,
                "summary_size_chars": event.summary_size_chars,
            },
        ))
        for listener in tuple(self._maintenance_listeners):
            try:
                listener(event)
            except Exception:
                # Observability callbacks must not alter compaction or fault behavior.
                continue

    async def _summarize_with_agent_stream(self, request) -> str:
        return await summarize_with_stream(self.agent.model, self.agent.stream_fn, request)

    def _refresh_assistant_entry_ids(self) -> None:
        if self._durable_session is None:
            self._assistant_entry_ids = {}
            return
        self._assistant_entry_ids = {}
        for projected in build_session_projection(self._durable_session.path_to_leaf()):
            if not isinstance(projected.message, AssistantMessage) or projected.source_entry_id is None:
                continue
            for tool_call in projected.message.tool_calls:
                self._assistant_entry_ids[tool_call.id] = projected.source_entry_id

    def _reconcile_selected_branch(self) -> None:
        """Durably close only unresolved ToolCalls on the selected branch."""
        assert self._durable_session is not None
        try:
            records = self._execution_journal.load() if self._execution_journal is not None else []
            projection = build_session_projection(self._durable_session.path_to_leaf())
            pending = _pending_tool_calls(projection)
            records_by_call = _records_by_call(records, pending)
            legacy_records = [record for record in records if record.is_legacy]
            reports: list[RecoveryItem] = []
            for assistant_entry_id, call_index, call in pending:
                record = records_by_call.get((assistant_entry_id, call.id, call_index))
                if record is None:
                    candidates = [
                        item for item in legacy_records
                        if item.tool_call_id == call.id and item.call_index == call_index and item.tool_name == call.name
                    ]
                    if candidates:
                        record = candidates[-1]
                result, report = _recovery_result(call, record, call_index=call_index)
                self._durable_session.append(result)
                reports.append(report)
            messages = build_session_messages(self._durable_session.path_to_leaf())
            self.agent.messages[:] = messages
            self.persisted_message_count = len(messages)
            self._refresh_assistant_entry_ids()
            self._incomplete_tail = _has_incomplete_tool_calls(messages)
            self.recovery_report = RecoveryReport(tuple(reports))
        except (ExecutionJournalError, SessionStoreError, ValueError) as error:
            raise SessionRecoveryError(f"could not reconcile interrupted tool calls: {error}") from error

    def _summary_token_limit(self) -> int | None:
        if self._compaction_policy is None:
            return self.agent.model.max_tokens
        return summary_max_tokens(self.agent.model, self._compaction_policy)

    def _fault(self, error: Exception) -> None:
        self._faulted = True
        raise SessionPersistenceError("failed to persist committed session message") from error

    def _fault_compaction(self, error: Exception) -> None:
        self._faulted = True
        raise SessionPersistenceError("failed to apply durable compaction") from error


def _records_by_call(
    records: list[JournalRecord],
    pending: list[tuple[str, int, ToolCall]],
) -> dict[tuple[str, str, int], JournalRecord]:
    expected = {
        (assistant_entry_id, call.id, call_index): call.name
        for assistant_entry_id, call_index, call in pending
    }
    selected_assistant_ids = {assistant_entry_id for assistant_entry_id, _, _ in expected}
    result: dict[tuple[str, str, int], JournalRecord] = {}
    for record in records:
        if record.assistant_entry_id is None:
            continue
        key = (record.assistant_entry_id, record.tool_call_id, record.call_index)
        if record.assistant_entry_id not in selected_assistant_ids:
            continue
        expected_tool_name = expected.get(key)
        if expected_tool_name is None or expected_tool_name != record.tool_name:
            raise ExecutionJournalError("execution journal record does not match the selected Session ToolCall")
        result[key] = record
    return result


def _pending_tool_calls(projection) -> list[tuple[str, int, ToolCall]]:
    pending: list[tuple[str, int, ToolCall]] = []
    for projected in projection:
        message = projected.message
        if isinstance(message, AssistantMessage):
            if pending:
                raise SessionStoreError("AssistantMessage appeared before prior ToolCalls were resolved")
            if projected.source_entry_id is None and message.tool_calls:
                raise SessionStoreError("Assistant ToolCall has no durable source entry")
            pending.extend((projected.source_entry_id, index, call) for index, call in enumerate(message.tool_calls))
            continue
        if isinstance(message, ToolResultMessage):
            if not pending:
                raise SessionStoreError("ToolResultMessage has no preceding unresolved ToolCall")
            assistant_entry_id, call_index, call = pending[0]
            if message.tool_call_id != call.id or message.tool_name != call.name:
                raise SessionStoreError("ToolResultMessage does not match the next unresolved ToolCall")
            pending.pop(0)
            continue
        if pending:
            raise SessionStoreError("non-tool message appeared before ToolCalls were resolved")
    return pending


def _recovery_result(
    call: ToolCall,
    record: JournalRecord | None,
    *,
    call_index: int,
) -> tuple[ToolResultMessage, RecoveryItem]:
    if record is not None and record.state == "completed" and record.receipt is not None:
        receipt = record.receipt
        outcome = str(receipt.metadata.get("outcome", "success"))
        return (
            ToolResultMessage(call.id, call.name, [TextBlock(receipt.content)], receipt.is_error, dict(receipt.metadata)),
            RecoveryItem(call.name, record.call_index, outcome, False),
        )
    if record is None:
        outcome = "execution_not_started"
        unknown = False
        text = "The previous tool call was durably recorded, but its executor was never launched. Do not assume any side effects occurred."
    elif record.state == "cancelled":
        outcome = "execution_cancelled"
        unknown = True
        text = "The previous tool execution was cancelled before a durable result was recorded. Its side effects may have occurred. Inspect the current state before taking a new action."
    else:
        outcome = "execution_interrupted"
        unknown = True
        text = "The previous tool execution was interrupted before a durable result was recorded. Its side effects may have occurred. Inspect the current state before taking a new action."
    metadata = {"outcome": outcome, "recovered": True, "side_effects_unknown": unknown}
    if unknown:
        metadata["recovery_state"] = "orphaned" if record is None or record.state in {"started", "executor_completed", "completed"} else record.state
    report_call_index = record.call_index if record is not None else call_index
    return ToolResultMessage(call.id, call.name, [TextBlock(text)], True, metadata), RecoveryItem(call.name, report_call_index, outcome, unknown)

def _has_incomplete_tool_calls(messages: list[Message]) -> bool:
    pending_tool_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            pending_tool_call_ids.update(tool_call.id for tool_call in message.tool_calls)
        elif isinstance(message, ToolResultMessage):
            pending_tool_call_ids.discard(message.tool_call_id)
    return bool(pending_tool_call_ids)


def _ensure_agent_has_no_durable_session(agent: Agent) -> None:
    binding = _DURABLE_BINDINGS.get(agent)
    if binding is None:
        return
    session = binding()
    if session is not None and not session._closed:
        raise RuntimeError("Agent already has a durable AgentSession")
    del _DURABLE_BINDINGS[agent]
