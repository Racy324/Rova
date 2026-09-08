from __future__ import annotations

import asyncio

import pytest

from rova.ai.events import ProviderFailure, Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.agent_core.agent import Agent
from rova.agent_core.retry import ProviderRetryPolicy
from rova.agent_core.tools import AgentTool, AgentToolResult
from rova.agent_session.agent_session import AgentSession
from rova.agent_session.session_store import JsonlSessionStore


class TrackingAttempt:
    def __init__(self, events: list[object]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = next(self._events)
        if isinstance(event, BaseException):
            raise event
        return event

    async def aclose(self) -> None:
        self.closed = True


def _transient_error() -> StreamError:
    return StreamError(
        "error",
        AssistantMessage([TextBlock("connection reset")], stop_reason="error"),
        failure=ProviderFailure("transient", code="network_interruption"),
    )


def _partial_text(text: str) -> list[object]:
    partial = AssistantMessage([TextBlock(text)], partial=True)
    return [Start(AssistantMessage([], partial=True)), TextDelta(text, partial)]


def _retry_policy(*, max_retries: int) -> ProviderRetryPolicy:
    async def no_sleep(_delay: float) -> None:
        return None

    return ProviderRetryPolicy(max_retries=max_retries, sleep=no_sleep, random_source=lambda: 0.5)


@pytest.mark.asyncio
async def test_transient_partial_text_attempt_is_closed_discarded_and_never_committed(tmp_path):
    attempts: list[TrackingAttempt] = []
    contexts = []
    discarded = []

    def stream(_model, context, _options):
        contexts.append(context)
        attempt = TrackingAttempt(
            [*_partial_text("old partial"), _transient_error()]
            if not attempts
            else [StreamDone(AssistantMessage([TextBlock("new final")]))]
        )
        attempts.append(attempt)
        return attempt

    agent = Agent(Model("mock"), "", [], stream, provider_retry_policy=_retry_policy(max_retries=1))
    agent.subscribe(lambda event: discarded.append(event) if event.type == "provider_attempt_discarded" else None)

    session = AgentSession.create(agent, session_root=tmp_path)
    result = await session.prompt("hello")

    assert [message.text for message in result] == ["new final"]
    assert [message.content if isinstance(message, UserMessage) else message.text for message in agent.messages] == ["hello", "new final"]
    assert contexts[0] is contexts[1]
    assert attempts[0].closed is True
    assert attempts[1].closed is True
    assert len(discarded) == 1
    assert not any(event.type == "provider_error" for event in agent.events)
    assert [message.content if isinstance(message, UserMessage) else message.text for message in JsonlSessionStore(tmp_path).load(session.session_id).messages] == ["hello", "new final"]


@pytest.mark.asyncio
async def test_retry_exhaustion_discards_all_partial_text_and_persists_only_terminal_failure(tmp_path):
    attempts: list[TrackingAttempt] = []

    def stream(_model, _context, _options):
        attempt = TrackingAttempt([*_partial_text(f"partial-{len(attempts)}"), _transient_error()])
        attempts.append(attempt)
        return attempt

    agent = Agent(Model("mock"), "", [], stream, provider_retry_policy=_retry_policy(max_retries=1))
    session = AgentSession.create(agent, session_root=tmp_path)

    result = await session.prompt("hello")

    assert [message.text for message in result] == ["connection reset"]
    assert [message.content if isinstance(message, UserMessage) else message.text for message in agent.messages] == ["hello", "connection reset"]
    assert all(attempt.closed for attempt in attempts)
    assert [message.content if isinstance(message, UserMessage) else message.text for message in JsonlSessionStore(tmp_path).load(session.session_id).messages] == ["hello", "connection reset"]


@pytest.mark.asyncio
async def test_partial_tool_call_is_discarded_and_only_complete_retry_tool_call_executes():
    attempts: list[TrackingAttempt] = []
    executions: list[dict] = []

    async def execute(_tool_call_id, arguments):
        executions.append(arguments)
        return AgentToolResult([TextBlock("tool result")])

    def stream(_model, _context, _options):
        if not attempts:
            attempt = TrackingAttempt([
                Start(AssistantMessage([], partial=True)),
                ToolCallDelta(
                    0,
                    AssistantMessage([], partial=True),
                    id_fragment="partial-call",
                    name_fragment="sample",
                    arguments_fragment='{"value":',
                ),
                _transient_error(),
            ])
        elif len(attempts) == 1:
            attempt = TrackingAttempt([
                StreamDone(AssistantMessage([ToolCall("complete-call", "sample", {"value": 2})], stop_reason="tool_calls")),
            ])
        else:
            attempt = TrackingAttempt([StreamDone(AssistantMessage([TextBlock("done")]))])
        attempts.append(attempt)
        return attempt

    tool = AgentTool(Tool("sample", "sample", {"value": int}), execute)
    agent = Agent(Model("mock"), "", [tool], stream, provider_retry_policy=_retry_policy(max_retries=1))

    result = await agent.run([UserMessage("run tool")])

    assert executions == [{"value": 2}]
    assert [message.text for message in result] == ["", "done"]
    assistant_calls = [message for message in agent.messages if isinstance(message, AssistantMessage) and message.tool_calls]
    assert assistant_calls == [AssistantMessage([ToolCall("complete-call", "sample", {"value": 2})], stop_reason="tool_calls")]
    assert all(attempt.closed for attempt in attempts)


@pytest.mark.asyncio
async def test_partial_tool_call_on_transient_exhaustion_never_executes():
    executions: list[dict] = []

    async def execute(_tool_call_id, arguments):
        executions.append(arguments)
        return AgentToolResult([TextBlock("unreachable")])

    partial_attempt = TrackingAttempt([
        Start(AssistantMessage([], partial=True)),
        ToolCallDelta(
            0,
            AssistantMessage([], partial=True),
            id_fragment="partial-call",
            name_fragment="sample",
            arguments_fragment='{"value":',
        ),
        _transient_error(),
    ])
    agent = Agent(
        Model("mock"),
        "",
        [AgentTool(Tool("sample", "sample", {"value": int}), execute)],
        lambda *_: partial_attempt,
        provider_retry_policy=_retry_policy(max_retries=0),
    )

    result = await agent.run([UserMessage("run tool")])

    assert executions == []
    assert [message.text for message in result] == ["connection reset"]
    assert partial_attempt.closed is True


@pytest.mark.asyncio
async def test_cancellation_discards_partial_attempt_without_commit_or_retry():
    attempt = TrackingAttempt([*_partial_text("partial"), asyncio.CancelledError()])
    calls = 0

    def stream(_model, _context, _options):
        nonlocal calls
        calls += 1
        return attempt

    agent = Agent(Model("mock"), "", [], stream, provider_retry_policy=_retry_policy(max_retries=2))

    with pytest.raises(asyncio.CancelledError):
        await agent.run([UserMessage("cancel")])

    assert calls == 1
    assert agent.messages == [UserMessage("cancel")]
    assert attempt.closed is True
