from __future__ import annotations

import asyncio

import pytest

from rova.ai.events import ProviderFailure, Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from rova.ai.messages import AssistantMessage, TextBlock, Usage, UserMessage
from rova.ai.models import Model
from rova.agent_core.retry import ProviderRetryPolicy
from rova.agent_session.session_store import JsonlSessionStore
from rova.app.runtime import build_rova_runtime
from rova.trace import JsonlTraceStore


def _transient_failure(message: str = "connection reset") -> StreamError:
    return StreamError(
        "error",
        AssistantMessage([TextBlock(message)], stop_reason="error"),
        failure=ProviderFailure("transient", code="network_interruption"),
    )


def _no_wait_retry_policy(max_retries: int = 1) -> ProviderRetryPolicy:
    async def no_sleep(_delay: float) -> None:
        return None

    return ProviderRetryPolicy(max_retries=max_retries, sleep=no_sleep, random_source=lambda: 0.5)


@pytest.mark.asyncio
async def test_unified_runtime_discards_interrupted_attempt_before_session_and_trace_commit(tmp_path) -> None:
    attempts = 0

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            partial = AssistantMessage([TextBlock("discarded partial")], partial=True)
            yield Start(AssistantMessage([], partial=True))
            yield TextDelta("discarded partial", partial)
            yield _transient_failure()
            return
        yield StreamDone(AssistantMessage([TextBlock("committed final")], usage=Usage(12, 3, 15)))

    session_root = tmp_path / "sessions"
    trace_root = tmp_path / "traces"
    runtime = build_rova_runtime(
        model=Model("mock"),
        stream_fn=stream,
        session_root=session_root,
        artifact_root=tmp_path / "artifacts",
        trace_root=trace_root,
        experience_review_enabled=False,
    )
    runtime.agent.provider_retry_policy = _no_wait_retry_policy()
    try:
        responses = await runtime.prompt("hello")
    finally:
        await runtime.close()

    durable_messages = JsonlSessionStore(session_root).load(runtime.session.session_id).messages
    trace = JsonlTraceStore(trace_root / "runs.jsonl").load_all()[0]

    assert attempts == 2
    assert [response.text for response in responses] == ["committed final"]
    assert [message.content if isinstance(message, UserMessage) else message.text for message in durable_messages] == [
        "hello",
        "committed final",
    ]
    assert len(trace.steps) == 1
    assert trace.steps[0].assistant_message is not None
    assert trace.steps[0].assistant_message.text == "committed final"
    assert trace.steps[0].usage.actual_usage == Usage(12, 3, 15)
    assert trace.actual_usage == Usage(12, 3, 15)


@pytest.mark.asyncio
async def test_unified_runtime_never_executes_a_partial_tool_call_from_discarded_attempt(tmp_path) -> None:
    attempts = 0

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield Start(AssistantMessage([], partial=True))
            yield ToolCallDelta(
                0,
                AssistantMessage([], partial=True),
                id_fragment="partial-memory",
                name_fragment="memory_manage",
                arguments_fragment='{"operation":"ADD",',
            )
            yield _transient_failure()
            return
        yield StreamDone(AssistantMessage([TextBlock("no tool was called")]))

    runtime = build_rova_runtime(
        model=Model("mock"),
        stream_fn=stream,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
        memory_root=tmp_path / "memory",
        experience_review_enabled=False,
    )
    runtime.agent.provider_retry_policy = _no_wait_retry_policy()
    try:
        responses = await runtime.prompt("remember something")
    finally:
        await runtime.close()

    assert attempts == 2
    assert [response.text for response in responses] == ["no tool was called"]
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert not any(getattr(message, "tool_calls", ()) for message in runtime.agent.messages)


@pytest.mark.asyncio
async def test_unified_runtime_persists_only_one_terminal_message_after_retry_exhaustion(tmp_path) -> None:
    attempts = 0

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        partial = AssistantMessage([TextBlock(f"partial-{attempts}")], partial=True)
        yield Start(AssistantMessage([], partial=True))
        yield TextDelta(partial.text, partial)
        yield _transient_failure("temporary outage")

    session_root = tmp_path / "sessions"
    trace_root = tmp_path / "traces"
    runtime = build_rova_runtime(
        model=Model("mock"),
        stream_fn=stream,
        session_root=session_root,
        artifact_root=tmp_path / "artifacts",
        trace_root=trace_root,
        experience_review_enabled=False,
    )
    runtime.agent.provider_retry_policy = _no_wait_retry_policy()
    try:
        responses = await runtime.prompt("hello")
    finally:
        await runtime.close()

    durable_messages = JsonlSessionStore(session_root).load(runtime.session.session_id).messages
    trace = JsonlTraceStore(trace_root / "runs.jsonl").load_all()[0]

    assert attempts == 2
    assert [response.text for response in responses] == ["temporary outage"]
    assert [message.content if isinstance(message, UserMessage) else message.text for message in durable_messages] == [
        "hello",
        "temporary outage",
    ]
    assert len(trace.steps) == 1
    assert trace.steps[0].assistant_message is not None
    assert trace.steps[0].assistant_message.text == "temporary outage"
    assert trace.steps[0].usage.actual_usage is None
    assert trace.actual_usage is None
