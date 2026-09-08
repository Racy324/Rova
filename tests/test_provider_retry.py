from __future__ import annotations

import asyncio

import pytest

from rova.ai.events import ProviderFailure, Start, StreamDone, StreamError
from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.ai.models import Model
from rova.agent_core import agent as agent_module
from rova.agent_core.agent import Agent
from rova.app.runtime import build_rova_runtime
from rova.app.settings import AppSettings


def _transient_error() -> StreamError:
    return StreamError(
        "error",
        AssistantMessage([TextBlock("temporary provider failure")], stop_reason="error"),
        failure=ProviderFailure("transient", code="timeout"),
    )


def _terminal_error(category: str) -> StreamError:
    return StreamError(
        "error",
        AssistantMessage([TextBlock(category)], stop_reason="error"),
        failure=ProviderFailure(category),
    )


def _policy(*, max_retries: int, sleep, random_source=lambda: 0.5):
    return agent_module.ProviderRetryPolicy(
        max_retries=max_retries,
        sleep=sleep,
        random_source=random_source,
    )


@pytest.mark.asyncio
async def test_agent_retries_two_transient_failures_then_commits_only_successful_attempt():
    attempts = 0
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            yield _transient_error()
            return
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(
        Model(provider="mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2, sleep=record_sleep),
    )

    result = await agent.run([UserMessage("hello")])

    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert [message.text for message in result] == ["done"]
    assert [message.content if isinstance(message, UserMessage) else message.text for message in agent.messages] == ["hello", "done"]
    assert not any(event.type == "provider_error" for event in agent.events)


@pytest.mark.asyncio
async def test_agent_returns_one_terminal_error_after_transient_retry_budget_is_exhausted():
    attempts = 0
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield _transient_error()

    agent = Agent(
        Model(provider="mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2, sleep=record_sleep),
    )

    result = await agent.run([UserMessage("hello")])

    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert [message.text for message in result] == ["temporary provider failure"]
    assert [message.content if isinstance(message, UserMessage) else message.text for message in agent.messages] == ["hello", "temporary provider failure"]
    assert [event.type for event in agent.events].count("provider_error") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["permanent", "unclassified", "context_overflow"])
async def test_agent_does_not_retry_non_transient_provider_failures(category):
    attempts = 0
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield _terminal_error(category)

    agent = Agent(
        Model(provider="mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2, sleep=record_sleep),
    )

    await agent.run([UserMessage("hello")])

    assert attempts == 1
    assert delays == []


@pytest.mark.asyncio
async def test_agent_does_not_retry_when_max_retries_is_zero():
    attempts = 0

    async def unexpected_sleep(_delay: float) -> None:
        raise AssertionError("retry delay must not run")

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield _transient_error()

    agent = Agent(
        Model(provider="mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=0, sleep=unexpected_sleep),
    )

    await agent.run([UserMessage("hello")])

    assert attempts == 1


@pytest.mark.asyncio
async def test_cancellation_during_retry_backoff_propagates_without_another_attempt():
    attempts = 0
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def blocking_sleep(_delay: float) -> None:
        waiting.set()
        await release.wait()

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield _transient_error()

    agent = Agent(
        Model(provider="mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2, sleep=blocking_sleep),
    )
    task = asyncio.create_task(agent.run([UserMessage("hello")]))
    await waiting.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts == 1


@pytest.mark.asyncio
async def test_cancellation_from_provider_request_propagates_without_retry():
    attempts = 0

    async def unexpected_sleep(_delay: float) -> None:
        raise AssertionError("cancelled request must not back off")

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError()
        yield  # pragma: no cover - keeps this test stream-shaped

    agent = Agent(
        Model(provider="mock"),
        "",
        [],
        stream,
        provider_retry_policy=_policy(max_retries=2, sleep=unexpected_sleep),
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.run([UserMessage("hello")])

    assert attempts == 1


def test_retry_policy_uses_exponential_backoff_with_injectable_jitter():
    async def no_sleep(_delay: float) -> None:
        return None

    policy = _policy(max_retries=2, sleep=no_sleep, random_source=lambda: 1.0)

    assert policy.delay_for_retry(1) == pytest.approx(0.6)
    assert policy.delay_for_retry(2) == pytest.approx(1.2)


@pytest.mark.parametrize("value", ["-1", "not-an-integer", "1.5"])
def test_provider_retry_setting_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="ROVA_PROVIDER_MAX_RETRIES"):
        AppSettings.from_env({"ROVA_PROVIDER_MAX_RETRIES": value})


def test_provider_retry_setting_defaults_to_two_and_accepts_zero():
    assert AppSettings.from_env({}).provider_max_retries == 2
    assert AppSettings.from_env({"ROVA_PROVIDER_MAX_RETRIES": "0"}).provider_max_retries == 0


def test_unified_runtime_passes_provider_retry_budget_to_its_agent(tmp_path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        provider_max_retries=4,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert runtime.agent.provider_retry_policy.max_retries == 4
