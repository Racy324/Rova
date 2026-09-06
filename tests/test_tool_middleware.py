from __future__ import annotations

import asyncio

import pytest

from rova.agent_core.tool_output import ToolOutputScope
from rova.agent_core.tools import (
    AgentTool,
    AgentToolResult,
    ToolExecutionContext,
    ToolExecutionError,
    ToolMiddleware,
    ToolRegistry,
    ToolRuntime,
)
from rova.ai.messages import TextBlock, ToolCall
from rova.ai.tools import Tool


class RecordingMiddleware:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events

    async def before_tool_execute(self, context: ToolExecutionContext) -> None:
        self.events.append(("before", self.name, context.tool_call_id, context.run_id, context.session_id))

    async def after_tool_execute(self, context: ToolExecutionContext, result) -> None:
        self.events.append(("after", self.name, context.tool_call_id, result.text, result.is_error))


def make_tool(execute):
    return AgentTool(Tool("sample", "sample", {"value": str}, required=("value",)), execute)


def make_runtime(tools, *, middlewares=()):
    return ToolRuntime(ToolRegistry(tools), middlewares=middlewares)


@pytest.mark.asyncio
async def test_middlewares_run_in_static_order_around_a_valid_tool_execution():
    events: list[tuple] = []

    async def execute(tool_call_id, params):
        events.append(("tool", tool_call_id, params["value"]))
        return AgentToolResult([TextBlock("ok")])

    runtime = make_runtime(
        [make_tool(execute)],
        middlewares=(RecordingMiddleware("first", events), RecordingMiddleware("second", events)),
    )

    result = await runtime.execute(ToolCall("call-1", "sample", {"value": "x"}), scope=ToolOutputScope("run-1", "session-1"))

    assert result.text == "ok"
    assert events == [
        ("before", "first", "call-1", "run-1", "session-1"),
        ("before", "second", "call-1", "run-1", "session-1"),
        ("tool", "call-1", "x"),
        ("after", "first", "call-1", "ok", False),
        ("after", "second", "call-1", "ok", False),
    ]


@pytest.mark.asyncio
async def test_unknown_tools_and_invalid_arguments_do_not_enter_middleware_lifecycle():
    events: list[tuple] = []

    async def execute(tool_call_id, params):
        return AgentToolResult([TextBlock("ok")])

    runtime = make_runtime([make_tool(execute)], middlewares=(RecordingMiddleware("only", events),))

    unknown = await runtime.execute(ToolCall("unknown", "missing", {}))
    invalid = await runtime.execute(ToolCall("invalid", "sample", {"value": 1}))

    assert unknown.is_error is True
    assert invalid.is_error is True
    assert events == []


@pytest.mark.asyncio
async def test_before_middleware_can_stop_a_valid_tool_with_declared_tool_error():
    calls: list[str] = []

    class BlockingMiddleware:
        async def before_tool_execute(self, context: ToolExecutionContext) -> None:
            raise ToolExecutionError("blocked", metadata={"outcome": "policy_denied"})

        async def after_tool_execute(self, context: ToolExecutionContext, result) -> None:
            calls.append("after")

    async def execute(tool_call_id, params):
        calls.append("tool")
        return AgentToolResult([TextBlock("not reached")])

    result = await make_runtime([make_tool(execute)], middlewares=(BlockingMiddleware(),)).execute(
        ToolCall("call", "sample", {"value": "x"})
    )

    assert result.is_error is True
    assert result.text == "blocked"
    assert result.metadata["outcome"] == "policy_denied"
    assert calls == []


@pytest.mark.asyncio
async def test_after_middleware_observes_declared_tool_failures_without_changing_them():
    observed: list[tuple[str, bool, str]] = []

    class Observer:
        async def before_tool_execute(self, context: ToolExecutionContext) -> None:
            return None

        async def after_tool_execute(self, context: ToolExecutionContext, result) -> None:
            observed.append((context.tool_call_id, result.is_error, result.text))
            result.content[:] = [TextBlock("changed only in observer")]

    async def execute(tool_call_id, params):
        raise ToolExecutionError("ordinary failure")

    result = await make_runtime([make_tool(execute)], middlewares=(Observer(),)).execute(
        ToolCall("call", "sample", {"value": "x"})
    )

    assert observed == [("call", True, "ordinary failure")]
    assert result.is_error is True
    assert result.text == "ordinary failure"


@pytest.mark.asyncio
async def test_programming_errors_from_middleware_propagate_as_runtime_failures():
    class BrokenMiddleware:
        async def before_tool_execute(self, context: ToolExecutionContext) -> None:
            return None

        async def after_tool_execute(self, context: ToolExecutionContext, result) -> None:
            raise AssertionError("middleware defect")

    async def execute(tool_call_id, params):
        return AgentToolResult([TextBlock("ok")])

    with pytest.raises(AssertionError, match="middleware defect"):
        await make_runtime([make_tool(execute)], middlewares=(BrokenMiddleware(),)).execute(
            ToolCall("call", "sample", {"value": "x"})
        )


@pytest.mark.asyncio
async def test_execution_context_identity_is_call_local_under_concurrency():
    observed: list[tuple[str, str | None, str | None]] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    class Observer:
        async def before_tool_execute(self, context: ToolExecutionContext) -> None:
            observed.append((context.tool_call_id, context.run_id, context.session_id))
            entered.set()
            await release.wait()

        async def after_tool_execute(self, context: ToolExecutionContext, result) -> None:
            return None

    async def execute(tool_call_id, params):
        return AgentToolResult([TextBlock(tool_call_id)])

    runtime = make_runtime([make_tool(execute)], middlewares=(Observer(),))
    first = asyncio.create_task(runtime.execute(ToolCall("first", "sample", {"value": "1"}), scope=ToolOutputScope("run-1", "session-1")))
    await entered.wait()
    second = asyncio.create_task(runtime.execute(ToolCall("second", "sample", {"value": "2"}), scope=ToolOutputScope("run-2", "session-2")))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    assert set(observed) == {("first", "run-1", "session-1"), ("second", "run-2", "session-2")}
