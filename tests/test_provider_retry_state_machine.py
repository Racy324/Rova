from __future__ import annotations

import asyncio

import pytest

from rova.ai.context import Context
from rova.ai.events import ProviderFailure, StreamDone, StreamError
from rova.ai.messages import AssistantMessage, TextBlock, Usage, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.agent_core.retry import ProviderRetryPolicy
from rova.trace import TraceRecorder


def _failure(category: str, *, text: str | None = None, usage: Usage | None = None) -> StreamError:
    return StreamError(
        "error",
        AssistantMessage([TextBlock(text or category)], stop_reason="error", usage=usage),
        failure=ProviderFailure(category, code=category),
    )


def _policy(max_retries: int) -> ProviderRetryPolicy:
    async def no_sleep(_delay: float) -> None:
        return None

    return ProviderRetryPolicy(max_retries=max_retries, sleep=no_sleep, random_source=lambda: 0.5)


@pytest.mark.asyncio
async def test_transient_then_overflow_then_transient_cannot_reset_the_transient_budget():
    attempts = 0
    recovered_contexts: list[Context] = []
    observed_contexts: list[Context] = []

    async def recover(context: Context) -> Context:
        recovered_contexts.append(context)
        return Context("compacted", list(context.messages), list(context.tools))

    async def stream(_model, context, _options):
        nonlocal attempts
        attempts += 1
        observed_contexts.append(context)
        if attempts == 1:
            yield _failure("transient")
        elif attempts == 2:
            yield _failure("context_overflow")
        elif attempts == 3:
            yield _failure("transient")
        else:
            yield StreamDone(AssistantMessage([TextBlock("must not run")]))

    agent = Agent(
        Model("mock"),
        "initial",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=1),
        context_overflow_recovery=recover,
    )

    result = await agent.run([UserMessage("hello")])

    assert attempts == 3
    assert len(recovered_contexts) == 1
    assert [context.system_prompt for context in observed_contexts] == ["initial", "initial", "compacted"]
    assert [message.text for message in result] == ["transient"]


@pytest.mark.asyncio
async def test_overflow_then_transient_retry_uses_compacted_context_and_succeeds():
    attempts = 0
    recovered_contexts: list[Context] = []
    observed_contexts: list[Context] = []

    async def recover(context: Context) -> Context:
        recovered_contexts.append(context)
        return Context("compacted", list(context.messages), list(context.tools))

    async def stream(_model, context, _options):
        nonlocal attempts
        attempts += 1
        observed_contexts.append(context)
        if attempts == 1:
            yield _failure("context_overflow")
        elif attempts == 2:
            yield _failure("transient")
        else:
            yield StreamDone(AssistantMessage([TextBlock("done")], usage=Usage(10, 2, 12)))

    agent = Agent(
        Model("mock"),
        "initial",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=1),
        context_overflow_recovery=recover,
    )

    result = await agent.run([UserMessage("hello")])

    assert attempts == 3
    assert len(recovered_contexts) == 1
    assert [context.system_prompt for context in observed_contexts] == ["initial", "compacted", "compacted"]
    assert [message.text for message in result] == ["done"]


@pytest.mark.asyncio
async def test_transient_then_overflow_recovery_rebuilds_context_before_success():
    attempts = 0
    observed_contexts: list[Context] = []

    async def recover(context: Context) -> Context:
        return Context("compacted", list(context.messages), list(context.tools))

    async def stream(_model, context, _options):
        nonlocal attempts
        attempts += 1
        observed_contexts.append(context)
        if attempts == 1:
            yield _failure("transient")
        elif attempts == 2:
            yield _failure("context_overflow")
        else:
            yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(
        Model("mock"),
        "initial",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=1),
        context_overflow_recovery=recover,
    )

    result = await agent.run([UserMessage("hello")])

    assert attempts == 3
    assert [context.system_prompt for context in observed_contexts] == ["initial", "initial", "compacted"]
    assert [message.text for message in result] == ["done"]


@pytest.mark.asyncio
async def test_second_overflow_is_terminal_without_a_second_recovery():
    attempts = 0
    recoveries = 0

    async def recover(context: Context) -> Context:
        nonlocal recoveries
        recoveries += 1
        return context

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield _failure("context_overflow")

    agent = Agent(
        Model("mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2),
        context_overflow_recovery=recover,
    )

    result = await agent.run([UserMessage("hello")])

    assert attempts == 2
    assert recoveries == 1
    assert [message.text for message in result] == ["context_overflow"]


@pytest.mark.asyncio
async def test_cancellation_during_overflow_recovery_propagates_without_attempt_or_commit():
    recover_started = asyncio.Event()
    never = asyncio.Event()
    attempts = 0

    async def recover(_context: Context) -> Context:
        recover_started.set()
        await never.wait()
        raise AssertionError("unreachable")

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield _failure("context_overflow")

    agent = Agent(
        Model("mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2),
        context_overflow_recovery=recover,
    )
    task = asyncio.create_task(agent.run([UserMessage("hello")]))
    await recover_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts == 1
    assert agent.messages == [UserMessage("hello")]


@pytest.mark.asyncio
async def test_trace_records_only_the_final_successful_attempt_usage():
    attempts = 0

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield _failure("transient", usage=Usage(100, 20, 120))
            return
        yield StreamDone(AssistantMessage([TextBlock("done")], usage=Usage(10, 2, 12)))

    agent = Agent(Model("mock"), "", [], stream, provider_retry_policy=_policy(max_retries=1))
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("hello")]))

    assert attempts == 2
    assert len(trace.steps) == 1
    assert trace.steps[0].usage.actual_usage == Usage(10, 2, 12)
    assert trace.actual_usage == Usage(10, 2, 12)
