from __future__ import annotations

import asyncio
import inspect
from uuid import uuid4
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar

from rova.ai.context import Context
from rova.ai.events import Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolResultMessage, UserMessage
from rova.ai.models import Model
from .events import AgentEvent, AgentTerminationReason
from .hooks import HookRegistry
from .tools import AgentTool, PreparedToolCall, ToolExecutionMode, ToolGovernance, ToolRegistry, ToolRuntime, resolve_batch_mode
from .tool_output import ToolOutputProcessor, ToolOutputScope
from .types import StreamFn


Listener = Callable[[AgentEvent], object]
ContextPreparer = Callable[[Context], Awaitable[Context]]
ContextOverflowRecovery = Callable[[Context], Awaitable[Context]]


class Agent:
    def __init__(
        self,
        model: Model,
        system_prompt: str,
        tools: list[AgentTool],
        stream_fn: StreamFn,
        *,
        max_turns: int = 8,
        tool_output_processor: ToolOutputProcessor | None = None,
        tool_execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL,
        tool_governance: ToolGovernance | None = None,
        hook_registry: HookRegistry | None = None,
        context_preparer: ContextPreparer | None = None,
        context_overflow_recovery: ContextOverflowRecovery | None = None,
    ) -> None:
        if not isinstance(tool_execution_mode, ToolExecutionMode):
            raise TypeError("tool_execution_mode must be a ToolExecutionMode")
        self.model = model
        self.system_prompt = system_prompt
        self.registry = ToolRegistry(tools)
        self.tool_runtime = ToolRuntime(
            self.registry,
            tool_output_processor=tool_output_processor,
            governance=tool_governance,
            hook_registry=hook_registry,
        )
        self._tool_output_scope: ContextVar[ToolOutputScope] = ContextVar("tool_output_scope", default=ToolOutputScope())
        self.stream_fn = stream_fn
        self.max_turns = max_turns
        self.tool_execution_mode = tool_execution_mode
        self.messages: list = []
        self.events: list[AgentEvent] = []
        self.listeners: list[Listener] = []
        self.last_context: Context | None = None
        self._context_preparer = context_preparer
        self._context_overflow_recovery = context_overflow_recovery

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def create_context_snapshot(self) -> Context:
        return Context(self.system_prompt, list(self.messages), self.registry.schemas)

    async def emit_runtime_event(self, event: AgentEvent) -> None:
        """Publish a committed Runtime fact from a collaborating product layer."""
        await self._emit(event)

    def set_context_preparer(self, preparer: ContextPreparer | None) -> None:
        """Set the product-owned provider-context preparation seam."""
        self._context_preparer = preparer

    def set_context_overflow_recovery(self, recovery: ContextOverflowRecovery | None) -> None:
        self._context_overflow_recovery = recovery

    def current_tool_output_scope(self) -> ToolOutputScope:
        return self._tool_output_scope.get()

    @contextmanager
    def bind_tool_output_scope(self, *, run_id: str | None = None, session_id: str | None = None):
        current = self._tool_output_scope.get()
        token = self._tool_output_scope.set(
            ToolOutputScope(run_id=run_id if run_id is not None else current.run_id, session_id=session_id if session_id is not None else current.session_id)
        )
        try:
            yield
        finally:
            self._tool_output_scope.reset(token)

    async def run(self, user_messages: list[UserMessage]) -> list[AssistantMessage]:
        self.events = []
        self.messages.extend(user_messages)
        await self._emit(AgentEvent("agent_start"))
        assistant_messages: list[AssistantMessage] = []
        for turn in range(1, self.max_turns + 1):
            await self._emit(AgentEvent("turn_start"))
            context = self.create_context_snapshot()
            if self._context_preparer is not None:
                context = await self._context_preparer(context)
            self.last_context = context
            assistant, started, termination_reason = await self._stream_assistant(context)
            assistant_messages.append(assistant)
            if not started:
                await self._emit(AgentEvent("message_start", message=assistant))
            self.messages.append(assistant)
            await self._emit(AgentEvent("message_end", message=assistant))
            if assistant.stop_reason in {"error", "aborted"}:
                await self._emit(AgentEvent("turn_end", message=assistant))
                await self._emit(AgentEvent("agent_end", message=assistant, termination_reason=termination_reason))
                return assistant_messages
            if not assistant.tool_calls:
                await self._emit(AgentEvent("turn_end", message=assistant))
                await self._emit(AgentEvent("agent_end", message=assistant, termination_reason=AgentTerminationReason.FINAL_RESPONSE))
                return assistant_messages
            batch_id = uuid4().hex
            resolved_tools = [self.registry.get(tool_call.name) for tool_call in assistant.tool_calls]
            batch_mode = resolve_batch_mode(self.tool_execution_mode, resolved_tools)
            call_facts = {
                tool_call.id: (
                    call_index,
                    (agent_tool.execution_mode if agent_tool is not None and agent_tool.execution_mode is not None else ToolExecutionMode.PARALLEL).value,
                )
                for call_index, (tool_call, agent_tool) in enumerate(zip(assistant.tool_calls, resolved_tools))
            }
            started: dict[str, PreparedToolCall] = {}
            completed: set[str] = set()
            finished_results: dict[str, ToolResultMessage] = {}

            async def emit_execution_state(
                prepared: PreparedToolCall,
                state: str,
                *,
                outcome: str | None = None,
                result: ToolResultMessage | None = None,
            ) -> None:
                await self._emit(AgentEvent(
                    "tool_execution_state",
                    tool_call_id=prepared.tool_call.id,
                    tool_name=prepared.tool_call.name,
                    batch_id=batch_id,
                    call_index=prepared.call_index,
                    batch_mode=batch_mode.value,
                    execution_mode=prepared.execution_mode.value,
                    execution_state=state,
                    outcome=outcome if result is None else result.metadata.get("outcome"),
                    result=result.text if result is not None else None,
                    is_error=result.is_error if result is not None else False,
                    metadata=dict(result.metadata) if result is not None else None,
                ))

            async def on_execution_start(prepared: PreparedToolCall) -> None:
                started[prepared.tool_call.id] = prepared
                await self._emit(AgentEvent(
                    "tool_execution_start",
                    tool_call_id=prepared.tool_call.id,
                    tool_name=prepared.tool_call.name,
                    batch_id=batch_id,
                    call_index=prepared.call_index,
                    batch_mode=batch_mode.value,
                    execution_mode=prepared.execution_mode.value,
                    args=dict(prepared.arguments),
                ))
                await emit_execution_state(prepared, "started")

            async def on_execution_end(prepared: PreparedToolCall, result: ToolResultMessage) -> None:
                completed.add(prepared.tool_call.id)
                finished_results[prepared.tool_call.id] = result
                await self._emit(AgentEvent(
                    "tool_execution_end",
                    tool_call_id=prepared.tool_call.id,
                    tool_name=prepared.tool_call.name,
                    batch_id=batch_id,
                    call_index=prepared.call_index,
                    batch_mode=batch_mode.value,
                    execution_mode=prepared.execution_mode.value,
                    result=result.text,
                    is_error=result.is_error,
                    metadata=result.metadata,
                ))
                await emit_execution_state(prepared, "completed", result=result)

            async def on_executor_completed(prepared: PreparedToolCall) -> None:
                completed.add(prepared.tool_call.id)
                await emit_execution_state(prepared, "executor_completed")

            async def on_execution_finished_uncommitted(prepared: PreparedToolCall) -> None:
                # `executor_completed` is emitted immediately after the inner
                # Tool function returns. A later lifecycle failure must not
                # manufacture a canonical completion or ToolResult commit.
                return

            async def on_result_committed(result: ToolResultMessage) -> None:
                self.messages.append(result)
                call_index, execution_mode = call_facts[result.tool_call_id]
                await self._emit(AgentEvent(
                    "message_end",
                    message=result,
                    tool_call_id=result.tool_call_id,
                    tool_name=result.tool_name,
                    batch_id=batch_id,
                    call_index=call_index,
                    batch_mode=batch_mode.value,
                    execution_mode=execution_mode,
                    metadata=result.metadata,
                ))

            try:
                await self.tool_runtime.execute_batch(
                    assistant.tool_calls,
                    runtime_mode=self.tool_execution_mode,
                    scope=self.current_tool_output_scope(),
                    on_execution_start=on_execution_start,
                    on_executor_completed=on_executor_completed,
                    on_execution_end=on_execution_end,
                    on_execution_finished_uncommitted=on_execution_finished_uncommitted,
                    on_result_committed=on_result_committed,
                )
            except asyncio.CancelledError:
                for tool_call_id, prepared in started.items():
                    if tool_call_id not in completed:
                        await emit_execution_state(prepared, "cancelled", outcome="cancelled")
                committed_call_ids = {
                    message.tool_call_id for message in self.messages if isinstance(message, ToolResultMessage)
                }
                for tool_call in assistant.tool_calls:
                    if tool_call.id in committed_call_ids:
                        continue
                    result = finished_results.get(tool_call.id)
                    if result is None:
                        result = ToolResultMessage(
                            tool_call.id,
                            tool_call.name,
                            [TextBlock("Tool execution cancelled by user.")],
                            is_error=True,
                            metadata={"outcome": "cancelled"},
                        )
                    await on_result_committed(result)
                raise
            except BaseException:
                for tool_call_id, prepared in started.items():
                    if tool_call_id not in completed:
                        await emit_execution_state(prepared, "interrupted", outcome="interrupted")
                raise
            await self._emit(AgentEvent("turn_end", message=assistant))
        failure = AssistantMessage(content=[TextBlock(f"Maximum turns ({self.max_turns}) reached")], stop_reason="error", partial=False)
        self.messages.append(failure)
        assistant_messages.append(failure)
        await self._emit(AgentEvent("message_start", message=failure))
        await self._emit(AgentEvent("message_end", message=failure))
        await self._emit(AgentEvent("turn_end", message=failure))
        await self._emit(AgentEvent("agent_end", message=failure, termination_reason=AgentTerminationReason.MAX_TURNS))
        return assistant_messages

    async def _stream_assistant(
        self,
        context: Context,
        *,
        overflow_retried: bool = False,
    ) -> tuple[AssistantMessage, bool, AgentTerminationReason | None]:
        started = False
        try:
            iterator = self.stream_fn(self.model, context, None).__aiter__()
        except Exception as error:
            await self._emit(AgentEvent("provider_error", error_type=type(error).__name__, error_message=str(error)))
            raise
        while True:
            try:
                event = await iterator.__anext__()
            except StopAsyncIteration:
                break
            except Exception as error:
                await self._emit(AgentEvent("provider_error", error_type=type(error).__name__, error_message=str(error)))
                raise
            if isinstance(event, Start):
                if not started:
                    started = True
                    await self._emit(AgentEvent("message_start", message=event.partial, assistant_message_event=event))
                continue
            if isinstance(event, (TextDelta, ToolCallDelta)):
                if not started:
                    started = True
                    await self._emit(AgentEvent("message_start", message=event.partial, assistant_message_event=event))
                await self._emit(AgentEvent("message_update", message=event.partial, assistant_message_event=event))
                continue
            if isinstance(event, StreamDone):
                event.message.partial = False
                return event.message, started, None
            if isinstance(event, StreamError):
                if (
                    event.failure is not None
                    and event.failure.classification == "context_overflow"
                    and self._context_overflow_recovery is not None
                    and not overflow_retried
                ):
                    try:
                        recovered_context = await self._context_overflow_recovery(context)
                    except Exception as error:
                        failure = AssistantMessage(
                            [TextBlock(f"Context overflow recovery failed: {error}")],
                            stop_reason="error",
                        )
                        await self._emit(AgentEvent(
                            "provider_error",
                            message=failure,
                            error_type=type(error).__name__,
                            error_message=str(error),
                        ))
                        return failure, started, AgentTerminationReason.CONTEXT_OVERFLOW
                    return await self._stream_assistant(recovered_context, overflow_retried=True)
                event.error.stop_reason = event.reason
                event.error.partial = False
                await self._emit(AgentEvent("provider_error", message=event.error, error_type="StreamError", error_message=event.error.text))
                if event.failure is not None and event.failure.classification == "context_overflow":
                    reason = AgentTerminationReason.CONTEXT_OVERFLOW
                else:
                    reason = AgentTerminationReason.PROVIDER_ERROR if event.reason == "error" else AgentTerminationReason.ABORTED
                return event.error, started, reason
        error = AssistantMessage(content=[TextBlock("Provider ended without a final message")], stop_reason="error")
        await self._emit(AgentEvent("provider_error", message=error, error_type="ProviderStreamEnded", error_message=error.text))
        return error, started, AgentTerminationReason.PROVIDER_ERROR

    async def _emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        for listener in list(self.listeners):
            outcome = listener(event)
            if inspect.isawaitable(outcome):
                await outcome
