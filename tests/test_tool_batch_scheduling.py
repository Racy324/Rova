from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.app.runtime import build_rova_runtime
from rova.agent_core.agent import Agent
from rova.agent_core.hooks import HookRegistry, ToolHookPoint
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode, resolve_batch_mode


async def _stream(*_args: object) -> AsyncIterator[Any]:
    if False:
        yield  # pragma: no cover


def _tool(name: str, mode: ToolExecutionMode | None = None) -> AgentTool:
    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        return AgentToolResult([])

    return AgentTool(Tool(name, name, {}), execute, execution_mode=mode)


def test_execution_modes_and_batch_resolution_default_to_parallel() -> None:
    assert set(ToolExecutionMode) == {ToolExecutionMode.PARALLEL, ToolExecutionMode.SEQUENTIAL}
    assert resolve_batch_mode(ToolExecutionMode.PARALLEL, [_tool("read", ToolExecutionMode.PARALLEL)]) is ToolExecutionMode.PARALLEL
    assert resolve_batch_mode(ToolExecutionMode.PARALLEL, [_tool("read"), None]) is ToolExecutionMode.PARALLEL
    assert resolve_batch_mode(ToolExecutionMode.PARALLEL, [_tool("write", ToolExecutionMode.SEQUENTIAL)]) is ToolExecutionMode.SEQUENTIAL
    assert resolve_batch_mode(ToolExecutionMode.SEQUENTIAL, [_tool("read", ToolExecutionMode.PARALLEL)]) is ToolExecutionMode.SEQUENTIAL


def test_agent_defaults_to_parallel_tool_execution_mode() -> None:
    agent = Agent(Model(), "", [], _stream)

    assert agent.tool_execution_mode is ToolExecutionMode.PARALLEL
    assert agent.tool_execution_mode is resolve_batch_mode(
        agent.tool_execution_mode,
        [_tool("unknown") if False else None],
    )


def test_build_runtime_passes_the_global_tool_execution_mode(tmp_path) -> None:
    runtime = build_rova_runtime(
        model=Model(),
        stream_fn=_stream,
        tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert runtime.agent.tool_execution_mode is ToolExecutionMode.SEQUENTIAL


@pytest.mark.asyncio
async def test_parallel_batch_keeps_completion_events_and_committed_messages_in_their_own_orders() -> None:
    a_started = asyncio.Event()
    release_a = asyncio.Event()
    b_finished = asyncio.Event()
    c_started = asyncio.Event()
    release_c = asyncio.Event()
    observed_events: list[tuple[str, str]] = []
    observed_hooks: list[tuple[str, str]] = []
    contexts = []

    async def execute_a(_call_id: str, _arguments: dict) -> AgentToolResult:
        a_started.set()
        await release_a.wait()
        return AgentToolResult([TextBlock("A")])

    async def execute_b(_call_id: str, _arguments: dict) -> AgentToolResult:
        b_finished.set()
        return AgentToolResult([TextBlock("B")])

    async def execute_c(_call_id: str, _arguments: dict) -> AgentToolResult:
        c_started.set()
        await release_c.wait()
        return AgentToolResult([TextBlock("C")])

    async def stream(_model: Model, context, _options: object) -> AsyncIterator[Any]:
        contexts.append(context)
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([
                ToolCall("a", "a", {}), ToolCall("b", "b", {}), ToolCall("c", "c", {}),
            ], stop_reason="tool_calls"))
            return
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    tools = [
        AgentTool(Tool("a", "a", {}), execute_a, execution_mode=ToolExecutionMode.PARALLEL),
        AgentTool(Tool("b", "b", {}), execute_b, execution_mode=ToolExecutionMode.PARALLEL),
        AgentTool(Tool("c", "c", {}), execute_c, execution_mode=ToolExecutionMode.PARALLEL),
    ]
    hooks = HookRegistry()
    hooks.register(ToolHookPoint.PRE_TOOL_USE, lambda context: observed_hooks.append(("pre", context.tool_name)), source="test.pre")
    hooks.register(ToolHookPoint.POST_TOOL_USE, lambda context: observed_hooks.append(("post", context.tool_name)), source="test.post")
    agent = Agent(Model(), "", tools, stream, hook_registry=hooks)
    agent.subscribe(lambda event: observed_events.append((event.type, event.tool_name or "")))

    run = asyncio.create_task(agent.run([UserMessage("run tools")]))
    await asyncio.wait_for(a_started.wait(), timeout=1)
    try:
        await asyncio.wait_for(b_finished.wait(), timeout=0.1)
        await asyncio.wait_for(c_started.wait(), timeout=1)
        release_c.set()
        await asyncio.sleep(0)
    finally:
        release_a.set()
    await run

    completed = [name for event_type, name in observed_events if event_type == "tool_execution_end"]
    lifecycle_events = [
        event
        for event in agent.events
        if event.type in {"tool_execution_start", "tool_execution_end"}
    ]
    committed_events = [
        event
        for event in agent.events
        if event.type == "message_end" and isinstance(event.message, ToolResultMessage)
    ]
    committed = [event.message.tool_name for event in agent.events if event.type == "message_end" and isinstance(event.message, ToolResultMessage)]
    results = [message.tool_name for message in agent.messages if isinstance(message, ToolResultMessage)]
    second_context_results = [message.tool_name for message in contexts[1].messages if isinstance(message, ToolResultMessage)]

    assert completed == ["b", "c", "a"]
    assert len({event.batch_id for event in lifecycle_events + committed_events}) == 1
    assert all(event.batch_id is not None for event in lifecycle_events + committed_events)
    assert [event.call_index for event in committed_events] == [0, 1, 2]
    assert {event.batch_mode for event in lifecycle_events + committed_events} == {"parallel"}
    assert {event.execution_mode for event in lifecycle_events + committed_events} == {"parallel"}
    assert observed_hooks == [
        ("pre", "a"), ("pre", "b"), ("pre", "c"),
        ("post", "b"), ("post", "c"), ("post", "a"),
    ]
    assert committed == ["a", "b", "c"]
    assert results == ["a", "b", "c"]
    assert second_context_results == ["a", "b", "c"]
