from __future__ import annotations

import pytest

from rova.ai.messages import TextBlock, ToolCall
from rova.ai.tools import Tool
from rova.agent_core.hooks import (
    HookRegistry,
    LifecycleHookError,
    PostToolUseContinue,
    PreToolUseBlock,
    PreToolUseContinue,
    ToolFailureContinue,
    ToolHookPoint,
)
from rova.agent_core.tools import (
    AgentTool,
    AgentToolResult,
    ToolExecutionError,
    ToolGovernancePreparation,
    ToolRegistry,
    ToolRuntime,
)


@pytest.mark.asyncio
async def test_pre_tool_hooks_are_ordered_and_pass_modified_arguments_to_next_handler() -> None:
    registry = HookRegistry()
    observed: list[tuple[str, dict[str, str]]] = []

    def first(context):
        observed.append(("first", dict(context.arguments)))
        return PreToolUseContinue({"value": "first"})

    async def second(context):
        observed.append(("second", dict(context.arguments)))
        return PreToolUseBlock("blocked")

    registry.register(ToolHookPoint.PRE_TOOL_USE, first, source="one")
    registry.register(ToolHookPoint.PRE_TOOL_USE, second, source="two")

    result = await registry.dispatch_pre_tool_use({"value": "original"})

    assert observed == [("first", {"value": "original"}), ("second", {"value": "first"})]
    assert result == PreToolUseBlock("blocked")


@pytest.mark.asyncio
async def test_hook_unregistration_is_idempotent() -> None:
    registry = HookRegistry()
    observed: list[str] = []
    unregister = registry.register(
        ToolHookPoint.PRE_TOOL_USE,
        lambda _context: observed.append("called"),
        source="test.unregister",
    )

    unregister()
    unregister()
    await registry.dispatch_pre_tool_use({})

    assert observed == []


@pytest.mark.asyncio
async def test_pre_tool_block_is_distinct_failure_and_never_reaches_execution() -> None:
    registry = HookRegistry()
    observed_failures: list[tuple[str, str]] = []

    def block(_context):
        return PreToolUseBlock("blocked by test")

    def observe_failure(context):
        observed_failures.append((context.stage, context.outcome))
        return ToolFailureContinue({"hook_diagnostic": "recorded"})

    registry.register(ToolHookPoint.PRE_TOOL_USE, block, source="test.block")
    registry.register(ToolHookPoint.TOOL_FAILURE, observe_failure, source="test.failure")

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        raise AssertionError("blocked tool must not execute")

    runtime = ToolRuntime(
        ToolRegistry([AgentTool(Tool("sample", "sample", {"value": str}), execute)]),
        hook_registry=registry,
    )

    result = await runtime.execute(ToolCall("call", "sample", {"value": "ok"}))

    assert result.is_error is True
    assert result.metadata == {
        "outcome": "hook_blocked",
        "hook_diagnostic": "recorded",
        "failure_stage": "pre_tool_use",
    }
    assert observed_failures == [("pre_tool_use", "hook_blocked")]


@pytest.mark.asyncio
async def test_post_tool_output_becomes_the_canonical_model_visible_result() -> None:
    registry = HookRegistry()

    def rewrite(_context):
        return PostToolUseContinue(content=(TextBlock("rewritten"),), metadata={"reviewed": True})

    registry.register(ToolHookPoint.POST_TOOL_USE, rewrite, source="test.post")

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        return AgentToolResult([TextBlock("original")], {"tool": "metadata"})

    runtime = ToolRuntime(
        ToolRegistry([AgentTool(Tool("sample", "sample", {}), execute)]),
        hook_registry=registry,
    )
    result = await runtime.execute(ToolCall("call", "sample", {}))

    assert result.text == "rewritten"
    assert result.metadata == {"outcome": "success", "tool": "metadata", "reviewed": True}


@pytest.mark.asyncio
async def test_post_or_failure_hook_cannot_override_established_runtime_metadata() -> None:
    registry = HookRegistry()

    registry.register(
        ToolHookPoint.POST_TOOL_USE,
        lambda _context: PostToolUseContinue(metadata={"outcome": "changed"}),
        source="test.post",
    )

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        return AgentToolResult([TextBlock("original")])

    runtime = ToolRuntime(
        ToolRegistry([AgentTool(Tool("sample", "sample", {}), execute)]),
        hook_registry=registry,
    )

    with pytest.raises(LifecycleHookError, match="override metadata key 'outcome'"):
        await runtime.execute(ToolCall("call", "sample", {}))


@pytest.mark.asyncio
async def test_tool_failure_reports_each_declared_stage_once_without_recursive_dispatch() -> None:
    registry = HookRegistry()
    observed: list[tuple[str, str]] = []

    def block(context):
        if context.tool_name == "blocked":
            return PreToolUseBlock("blocked")
        return None

    def failure(context):
        observed.append((context.stage, context.outcome))

    registry.register(ToolHookPoint.PRE_TOOL_USE, block, source="test.block")
    registry.register(ToolHookPoint.TOOL_FAILURE, failure, source="test.failure")

    class Governance:
        async def preflight(self, context, _tool):
            if context.tool_name == "policy":
                raise ToolExecutionError("policy", metadata={"outcome": "policy_denied"})
            if context.tool_name == "approval":
                raise ToolExecutionError("approval", metadata={"outcome": "approval_denied"})
            return ToolGovernancePreparation()

        async def after_success(self, _prepared, result):
            return result

        def enrich_error(self, _prepared, error):
            return error

    async def error_execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        raise ToolExecutionError("execution", metadata={"outcome": "tool_execution_error"})

    noop = lambda _call_id, _arguments: _successful_result()
    runtime = ToolRuntime(
        ToolRegistry([
            AgentTool(Tool("validated", "validated", {"value": str}, required=("value",)), noop),
            AgentTool(Tool("blocked", "blocked", {}), noop),
            AgentTool(Tool("policy", "policy", {}), noop),
            AgentTool(Tool("approval", "approval", {}), noop),
            AgentTool(Tool("execution", "execution", {}), error_execute),
        ]),
        hook_registry=registry,
        governance=Governance(),
    )

    for call in (
        ToolCall("lookup", "unknown", {}),
        ToolCall("validation", "validated", {"value": 1}),
        ToolCall("pre", "blocked", {}),
        ToolCall("policy", "policy", {}),
        ToolCall("approval", "approval", {}),
        ToolCall("execution", "execution", {}),
    ):
        assert (await runtime.execute(call)).is_error is True

    assert observed == [
        ("lookup", "tool_input_error"),
        ("validation", "tool_input_error"),
        ("pre_tool_use", "hook_blocked"),
        ("policy", "policy_denied"),
        ("approval", "approval_denied"),
        ("execution", "tool_execution_error"),
    ]


async def _successful_result() -> AgentToolResult:
    return AgentToolResult([TextBlock("ok")])
