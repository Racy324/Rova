from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

from rova.ai.context import Context
from rova.ai.events import Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.ai.models import Model
from .events import AgentEvent, AgentTerminationReason
from .tools import AgentTool, ToolRegistry
from .tool_output import ToolOutputProcessor, ToolOutputScope
from .types import StreamFn


Listener = Callable[[AgentEvent], object]


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
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.registry = ToolRegistry(tools, tool_output_processor=tool_output_processor)
        self._tool_output_scope: ContextVar[ToolOutputScope] = ContextVar("tool_output_scope", default=ToolOutputScope())
        self.stream_fn = stream_fn
        self.max_turns = max_turns
        self.messages: list = []
        self.events: list[AgentEvent] = []
        self.listeners: list[Listener] = []
        self.last_context: Context | None = None

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def create_context_snapshot(self) -> Context:
        return Context(self.system_prompt, list(self.messages), self.registry.schemas)

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
            for tool_call in assistant.tool_calls:
                await self._emit(AgentEvent("tool_execution_start", tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments))
                result = await self.registry.execute(tool_call, scope=self.current_tool_output_scope())
                self.messages.append(result)
                await self._emit(AgentEvent("tool_execution_end", tool_call_id=tool_call.id, tool_name=tool_call.name, result=result.text, is_error=result.is_error, metadata=result.metadata))
            await self._emit(AgentEvent("turn_end", message=assistant))
        failure = AssistantMessage(content=[TextBlock(f"Maximum turns ({self.max_turns}) reached")], stop_reason="error", partial=False)
        self.messages.append(failure)
        assistant_messages.append(failure)
        await self._emit(AgentEvent("message_start", message=failure))
        await self._emit(AgentEvent("message_end", message=failure))
        await self._emit(AgentEvent("turn_end", message=failure))
        await self._emit(AgentEvent("agent_end", message=failure, termination_reason=AgentTerminationReason.MAX_TURNS))
        return assistant_messages

    async def _stream_assistant(self, context: Context) -> tuple[AssistantMessage, bool, AgentTerminationReason | None]:
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
                event.error.stop_reason = event.reason
                event.error.partial = False
                await self._emit(AgentEvent("provider_error", message=event.error, error_type="StreamError", error_message=event.error.text))
                return event.error, started, AgentTerminationReason.PROVIDER_ERROR if event.reason == "error" else AgentTerminationReason.ABORTED
        error = AssistantMessage(content=[TextBlock("Provider ended without a final message")], stop_reason="error")
        await self._emit(AgentEvent("provider_error", message=error, error_type="ProviderStreamEnded", error_message=error.text))
        return error, started, AgentTerminationReason.PROVIDER_ERROR

    async def _emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        for listener in list(self.listeners):
            outcome = listener(event)
            if inspect.isawaitable(outcome):
                await outcome
