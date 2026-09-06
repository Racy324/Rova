from __future__ import annotations

import pytest

from rova.ai.messages import TextBlock, ToolCall
from rova.ai.tools import Tool
from rova.agent_core.tool_output import ToolOutputScope
from rova.agent_core.tools import (
    AgentTool,
    AgentToolResult,
    PreToolUseHook,
    ToolExecutionContext,
    ToolExecutionMode,
    ToolRegistry,
    ToolRuntime,
)
from rova.app.workspace.approval import AlwaysDeny
from rova.app.workspace.controlled_tool import RovaToolGovernance
from rova.app.workspace.policy import ToolPolicyDecision, ToolPolicyResult


def _runtime(*, hooks: tuple[PreToolUseHook, ...]) -> ToolRuntime:
    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        return AgentToolResult([TextBlock("executed")])

    tool = AgentTool(Tool("sample", "sample", {"value": str}, required=("value",)), execute)
    return ToolRuntime(ToolRegistry([tool]), pre_tool_hooks=hooks)


@pytest.mark.asyncio
async def test_pre_tool_use_revalidates_modified_arguments_before_execution() -> None:
    observed: list[ToolExecutionContext] = []

    class InvalidatingHook:
        async def pre_tool_use(self, context: ToolExecutionContext):
            observed.append(context)
            return {"value": 3}

    result = await _runtime(hooks=(InvalidatingHook(),)).execute(
        ToolCall("call", "sample", {"value": "ok"}),
        scope=ToolOutputScope("run", "session"),
    )

    assert len(observed) == 1
    assert observed[0].arguments == {"value": "ok"}
    assert result.is_error is True
    assert result.metadata["outcome"] == "tool_input_error"
    assert "value must be str" in result.text


@pytest.mark.asyncio
async def test_pre_tool_use_cannot_bypass_the_remaining_execution_pipeline() -> None:
    order: list[str] = []

    class RecordingHook:
        async def pre_tool_use(self, context: ToolExecutionContext):
            order.append("pre_tool_use")
            return None

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        order.append("execute")
        return AgentToolResult([TextBlock("ok")])

    runtime = ToolRuntime(
        ToolRegistry([AgentTool(Tool("sample", "sample", {}), execute)]),
        pre_tool_hooks=(RecordingHook(),),
    )

    result = await runtime.execute(ToolCall("call", "sample", {}))

    assert result.is_error is False
    assert order == ["pre_tool_use", "execute"]


@pytest.mark.asyncio
async def test_policy_denial_is_a_preflight_result_and_does_not_start_that_executor() -> None:
    class DenyPolicy:
        def evaluate(self, _request):
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "approval required")

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        raise AssertionError("denied tool must not execute")

    tool = AgentTool(Tool("write", "write", {}), execute)
    starts: list[str] = []
    runtime = ToolRuntime(
        ToolRegistry([tool]),
        governance=RovaToolGovernance(DenyPolicy(), AlwaysDeny()),
    )

    results = await runtime.execute_batch(
        [ToolCall("denied", "write", {})],
        runtime_mode=ToolExecutionMode.PARALLEL,
        on_execution_start=lambda prepared: _record_start(starts, prepared.tool_call.name),
    )

    assert results[0].is_error is True
    assert results[0].metadata["outcome"] == "approval_denied"
    assert starts == []


@pytest.mark.asyncio
async def test_runtime_governance_applies_policy_after_pre_tool_use_without_a_controlled_wrapper() -> None:
    order: list[str] = []

    class DenyPolicy:
        def evaluate(self, request):
            order.append(f"policy:{request.arguments['value']}")
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "approval required")

    class RewriteHook:
        async def pre_tool_use(self, _context: ToolExecutionContext):
            order.append("hook")
            return {"value": "rewritten"}

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        raise AssertionError("approval denial must prevent execution")

    runtime = ToolRuntime(
        ToolRegistry([AgentTool(Tool("write", "write", {"value": str}), execute)]),
        pre_tool_hooks=(RewriteHook(),),
        governance=RovaToolGovernance(DenyPolicy(), AlwaysDeny()),
    )

    result = await runtime.execute(ToolCall("denied", "write", {"value": "original"}))

    assert result.is_error is True
    assert result.metadata["outcome"] == "approval_denied"
    assert order == ["hook", "policy:rewritten"]


async def _record_start(starts: list[str], name: str) -> None:
    starts.append(name)


@pytest.mark.asyncio
async def test_mixed_parallel_preflight_failures_commit_one_source_ordered_result_per_call() -> None:
    class DenyPolicy:
        def evaluate(self, request):
            if request.tool_name == "policy_denied":
                return ToolPolicyResult(ToolPolicyDecision.DENY, "policy denied")
            if request.tool_name == "valid":
                return ToolPolicyResult(ToolPolicyDecision.ALLOW, "allowed")
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "approval required")

    executed: list[str] = []

    async def good_execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        executed.append("good")
        return AgentToolResult([TextBlock("good")])

    async def denied_execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        raise AssertionError("preflight denial must not execute")

    valid = AgentTool(Tool("valid", "valid", {"value": str}, required=("value",)), good_execute, execution_mode=ToolExecutionMode.SEQUENTIAL)
    policy_denied = AgentTool(Tool("policy_denied", "policy denied", {}), denied_execute)
    approval_denied = AgentTool(Tool("approval_denied", "approval denied", {}), denied_execute)
    runtime = ToolRuntime(
        ToolRegistry([valid, policy_denied, approval_denied]),
        governance=RovaToolGovernance(DenyPolicy(), AlwaysDeny()),
    )

    results = await runtime.execute_batch(
        [
            ToolCall("a", "unknown", {}),
            ToolCall("b", "valid", {"value": 1}),
            ToolCall("c", "policy_denied", {}),
            ToolCall("d", "approval_denied", {}),
            ToolCall("e", "valid", {"value": "ok"}),
        ],
        runtime_mode=ToolExecutionMode.PARALLEL,
    )

    assert [result.tool_call_id for result in results] == ["a", "b", "c", "d", "e"]
    assert [result.metadata["outcome"] for result in results] == [
        "tool_input_error", "tool_input_error", "policy_denied", "approval_denied", "success",
    ]
    assert executed == ["good"]


@pytest.mark.asyncio
async def test_resolved_sequential_tool_makes_a_batch_sequential_despite_failed_siblings() -> None:
    completed: list[str] = []

    class ObservePreflight:
        async def pre_tool_use(self, context: ToolExecutionContext):
            if context.tool_name == "parallel":
                assert completed == ["sequential"]
            return None

    async def execute_sequential(_call_id: str, _arguments: dict) -> AgentToolResult:
        completed.append("sequential")
        return AgentToolResult([TextBlock("sequential")])

    async def execute_parallel(_call_id: str, _arguments: dict) -> AgentToolResult:
        completed.append("parallel")
        return AgentToolResult([TextBlock("parallel")])

    runtime = ToolRuntime(
        ToolRegistry([
            AgentTool(Tool("sequential", "sequential", {}), execute_sequential, execution_mode=ToolExecutionMode.SEQUENTIAL),
            AgentTool(Tool("parallel", "parallel", {}), execute_parallel, execution_mode=ToolExecutionMode.PARALLEL),
        ]),
        pre_tool_hooks=(ObservePreflight(),),
    )

    results = await runtime.execute_batch(
        [ToolCall("a", "unknown", {}), ToolCall("b", "sequential", {}), ToolCall("c", "parallel", {})],
        runtime_mode=ToolExecutionMode.PARALLEL,
    )

    assert [result.tool_call_id for result in results] == ["a", "b", "c"]
    assert [result.metadata["outcome"] for result in results] == ["tool_input_error", "success", "success"]
    assert completed == ["sequential", "parallel"]
