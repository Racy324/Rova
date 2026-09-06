from __future__ import annotations

import asyncio
import json

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.agent_core.agent import Agent
from rova.agent_core.hooks import HookRegistry, ToolHookPoint
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode
from rova.agent_session.agent_session import AgentSession


@pytest.mark.asyncio
async def test_cancelling_a_parallel_batch_cancels_all_started_tools_and_records_terminal_states(tmp_path) -> None:
    started = [asyncio.Event() for _ in range(3)]
    cancelled: list[str] = []

    def tool(index: int) -> AgentTool:
        async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
            started[index].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(str(index))
                raise
            raise AssertionError("unreachable")

        return AgentTool(Tool(f"tool_{index}", f"tool_{index}", {}), execute, execution_mode=ToolExecutionMode.PARALLEL)

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([
            ToolCall("a", "tool_0", {}), ToolCall("b", "tool_1", {}), ToolCall("c", "tool_2", {}),
        ], stop_reason="tool_calls"))

    agent = Agent(Model("mock"), "", [tool(0), tool(1), tool(2)], stream)
    session = AgentSession.create(agent, session_root=tmp_path)
    prompt = asyncio.create_task(session.prompt("run"))
    await asyncio.gather(*(item.wait() for item in started))

    prompt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prompt

    assert sorted(cancelled) == ["0", "1", "2"]
    journal = tmp_path / f"{session.session_id}.executions.jsonl"
    states = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [(item["tool_call_id"], item["state"]) for item in states] == [
        ("a", "started"), ("b", "started"), ("c", "started"),
        ("a", "cancelled"), ("b", "cancelled"), ("c", "cancelled"),
    ]


@pytest.mark.asyncio
async def test_parallel_harness_failure_cancels_siblings_without_committing_partial_results(tmp_path) -> None:
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def crash(_call_id: str, _arguments: dict) -> AgentToolResult:
        await sibling_started.wait()
        raise RuntimeError("harness failed")

    async def block(_call_id: str, _arguments: dict) -> AgentToolResult:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([
            ToolCall("a", "crash", {}), ToolCall("b", "block", {}),
        ], stop_reason="tool_calls"))

    agent = Agent(
        Model("mock"), "",
        [
            AgentTool(Tool("crash", "crash", {}), crash, execution_mode=ToolExecutionMode.PARALLEL),
            AgentTool(Tool("block", "block", {}), block, execution_mode=ToolExecutionMode.PARALLEL),
        ],
        stream,
    )
    session = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(RuntimeError, match="harness failed"):
        await session.prompt("run")

    assert sibling_cancelled.is_set()
    journal = tmp_path / f"{session.session_id}.executions.jsonl"
    states = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert {(item["tool_call_id"], item["state"]) for item in states} >= {
        ("a", "started"), ("b", "started"), ("a", "interrupted"), ("b", "interrupted"),
    }


@pytest.mark.asyncio
async def test_parallel_post_hook_failure_cancels_sibling_without_partial_commit_and_preserves_execution_states(tmp_path) -> None:
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    hooks = HookRegistry()

    async def complete(_call_id: str, _arguments: dict) -> AgentToolResult:
        return AgentToolResult([TextBlock("complete")])

    async def block(_call_id: str, _arguments: dict) -> AgentToolResult:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def fail_post(context):
        if context.tool_name == "complete":
            await sibling_started.wait()
            raise RuntimeError("post hook failed")

    hooks.register(ToolHookPoint.POST_TOOL_USE, fail_post, source="test.post")

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([
            ToolCall("a", "complete", {}), ToolCall("b", "block", {}),
        ], stop_reason="tool_calls"))

    agent = Agent(
        Model("mock"), "",
        [
            AgentTool(Tool("complete", "complete", {}), complete, execution_mode=ToolExecutionMode.PARALLEL),
            AgentTool(Tool("block", "block", {}), block, execution_mode=ToolExecutionMode.PARALLEL),
        ],
        stream,
        hook_registry=hooks,
    )
    session = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(RuntimeError, match="Lifecycle hook 'test.post' failed"):
        await session.prompt("run")

    assert sibling_cancelled.is_set()
    assert not [message for message in agent.messages if message.__class__.__name__ == "ToolResultMessage"]
    journal = tmp_path / f"{session.session_id}.executions.jsonl"
    states = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert {(item["tool_call_id"], item["state"]) for item in states} == {
        ("a", "started"), ("b", "started"), ("a", "executor_completed"), ("b", "interrupted"),
    }
