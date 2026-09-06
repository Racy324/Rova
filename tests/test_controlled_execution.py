from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.agent_core.agent import Agent
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError, ToolRegistry, ToolRuntime
from rova.app.workspace import Workspace
from rova.app.workspace.approval import ApprovalDecision, AlwaysApprove, AlwaysDeny
from rova.app.workspace.controlled_tool import ControlledTool, ToolExecutionDenied, build_controlled_coding_tools
from rova.app.workspace.policy import DefaultCodingToolPolicy, ToolExecutionRequest, ToolPolicyDecision, ToolPolicyResult


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "read.txt").write_text("observed\n", encoding="utf-8")
    (root / "edit.txt").write_text("old value\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("tool_name", "decision", "reason"),
    [
        ("read", ToolPolicyDecision.ALLOW, "read-only workspace operation"),
        ("list_dir", ToolPolicyDecision.ALLOW, "read-only workspace operation"),
        ("search", ToolPolicyDecision.ALLOW, "read-only workspace operation"),
        ("write", ToolPolicyDecision.REQUIRE_APPROVAL, "this operation modifies workspace files"),
        ("edit", ToolPolicyDecision.REQUIRE_APPROVAL, "this operation modifies workspace files"),
        ("shell", ToolPolicyDecision.REQUIRE_APPROVAL, "shell commands require approval"),
        ("unknown", ToolPolicyDecision.DENY, "tool is not permitted by this policy"),
    ],
)
def test_default_coding_policy_has_stable_decisions_and_reasons(tool_name, decision, reason):
    result = DefaultCodingToolPolicy().evaluate(ToolExecutionRequest(tool_name, {}))
    assert result == ToolPolicyResult(decision, reason)
    assert result.reason


def make_inner(calls: list[dict], *, failure: Exception | None = None) -> AgentTool:
    async def execute(tool_call_id: str, params: dict) -> AgentToolResult:
        calls.append(params)
        if failure is not None:
            raise failure
        return AgentToolResult([TextBlock("inner result")])

    return AgentTool(Tool("inner", "Inner test tool", {"value": str}), execute)


class StaticPolicy:
    def __init__(self, result: ToolPolicyResult) -> None:
        self.result = result

    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult:
        return self.result


class FailingApprovalHandler:
    async def request_approval(self, request):
        raise RuntimeError("approval UI failed")


class FailingPolicy:
    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult:
        raise RuntimeError("policy implementation failed")


@pytest.mark.asyncio
async def test_controlled_tool_allow_calls_inner_and_preserves_schema():
    calls: list[dict] = []
    inner = make_inner(calls)
    controlled = ControlledTool(inner, StaticPolicy(ToolPolicyResult(ToolPolicyDecision.ALLOW, "allowed")))
    result = await controlled.execute("call", {"value": "x"})
    assert result.content == [TextBlock("inner result")]
    assert calls == [{"value": "x"}]
    assert controlled.tool is inner.tool


@pytest.mark.asyncio
async def test_controlled_tool_policy_deny_does_not_call_inner_and_becomes_tool_error():
    calls: list[dict] = []
    controlled = ControlledTool(make_inner(calls), StaticPolicy(ToolPolicyResult(ToolPolicyDecision.DENY, "denied by test")))
    with pytest.raises(ToolExecutionDenied, match="Tool execution denied by policy"):
        await controlled.execute("call", {"value": "x"})
    result = await ToolRuntime(ToolRegistry([controlled])).execute(ToolCall("call", "inner", {"value": "x"}))
    assert result.is_error is True
    assert "Reason: denied by test" in result.text
    assert result.metadata == {
        "outcome": "policy_denied",
        "policy_decision": "deny",
        "policy_reason": "denied by test",
    }
    assert calls == []


@pytest.mark.asyncio
async def test_controlled_tool_requires_explicit_approval_and_fails_closed():
    calls: list[dict] = []
    policy = StaticPolicy(ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "approval needed"))
    approved = ControlledTool(make_inner(calls), policy, AlwaysApprove())
    assert (await approved.execute("call", {"value": "x"})).content == [TextBlock("inner result")]
    denied = ControlledTool(make_inner(calls), policy, AlwaysDeny())
    with pytest.raises(ToolExecutionDenied, match="was not approved"):
        await denied.execute("call", {"value": "x"})
    no_handler = ControlledTool(make_inner(calls), policy)
    with pytest.raises(ToolExecutionDenied, match="no approval handler"):
        await no_handler.execute("call", {"value": "x"})
    failing_handler = ControlledTool(make_inner(calls), policy, FailingApprovalHandler())
    with pytest.raises(ToolExecutionDenied, match="approval failed"):
        await failing_handler.execute("call", {"value": "x"})
    assert calls == [{"value": "x"}]


@pytest.mark.asyncio
async def test_registry_exposes_approval_outcomes_as_structured_metadata():
    policy = StaticPolicy(ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "approval needed"))
    cases = [
        (AlwaysApprove(), False, "success", "approve"),
        (AlwaysDeny(), True, "approval_denied", "deny"),
        (None, True, "approval_unavailable", "unavailable"),
        (FailingApprovalHandler(), True, "approval_error", "error"),
    ]

    for handler, is_error, outcome, approval_decision in cases:
        result = await ToolRuntime(ToolRegistry([ControlledTool(make_inner([]), policy, handler)])).execute(
            ToolCall("call", "inner", {"value": "x"})
        )
        assert result.is_error is is_error
        assert result.metadata["outcome"] == outcome
        assert result.metadata["policy_decision"] == "require_approval"
        assert result.metadata["policy_reason"] == "approval needed"
        assert result.metadata["approval_required"] is True
        assert result.metadata["approval_decision"] == approval_decision


@pytest.mark.asyncio
async def test_inner_failures_still_follow_existing_tool_result_error_semantics():
    controlled = ControlledTool(
        make_inner([], failure=ToolExecutionError("ordinary inner failure")),
        StaticPolicy(ToolPolicyResult(ToolPolicyDecision.ALLOW, "allowed")),
    )
    result = await ToolRuntime(ToolRegistry([controlled])).execute(ToolCall("call", "inner", {"value": "x"}))
    assert result.is_error is True
    assert result.text == "ordinary inner failure"


@pytest.mark.asyncio
async def test_controlled_inner_failure_preserves_policy_and_approval_metadata():
    policy = StaticPolicy(ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "approval needed"))
    controlled = ControlledTool(
        make_inner([], failure=ToolExecutionError("ordinary inner failure")),
        policy,
        AlwaysApprove(),
    )

    result = await ToolRuntime(ToolRegistry([controlled])).execute(ToolCall("call", "inner", {"value": "x"}))

    assert result.is_error is True
    assert result.metadata == {
        "outcome": "tool_execution_error",
        "policy_decision": "require_approval",
        "policy_reason": "approval needed",
        "approval_required": True,
        "approval_decision": "approve",
    }


@pytest.mark.asyncio
async def test_registry_propagates_unmarked_harness_failures():
    registry = ToolRegistry([make_inner([], failure=AssertionError("invariant violated"))])

    with pytest.raises(AssertionError, match="invariant violated"):
        await ToolRuntime(registry).execute(ToolCall("call", "inner", {"value": "x"}))


@pytest.mark.asyncio
async def test_registry_rejects_non_json_tool_metadata_as_a_harness_failure():
    async def execute(tool_call_id, params):
        return AgentToolResult([TextBlock("bad metadata")], {"value": float("nan")})

    registry = ToolRegistry([AgentTool(Tool("inner", "Inner", {"value": str}), execute)])

    with pytest.raises(TypeError, match="JSON-compatible"):
        await ToolRuntime(registry).execute(ToolCall("call", "inner", {"value": "x"}))


@pytest.mark.asyncio
async def test_policy_implementation_errors_are_not_reclassified_as_policy_denials():
    controlled = ControlledTool(make_inner([]), FailingPolicy())
    with pytest.raises(RuntimeError, match="policy implementation failed"):
        await controlled.execute("call", {"value": "x"})


@pytest.mark.asyncio
async def test_controlled_coding_tools_enforce_write_and_edit_approval_side_effects(workspace_root: Path):
    workspace = Workspace(workspace_root)
    denied_tools = build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny())
    denied_registry = ToolRegistry(denied_tools)
    write = await ToolRuntime(denied_registry).execute(ToolCall("write", "write", {"path": "created.txt", "content": "created"}))
    edit = await ToolRuntime(denied_registry).execute(ToolCall("edit", "edit", {"path": "edit.txt", "old_text": "old", "new_text": "new"}))
    assert write.is_error and edit.is_error
    assert not (workspace_root / "created.txt").exists()
    assert (workspace_root / "edit.txt").read_text(encoding="utf-8") == "old value\n"

    approved_registry = ToolRegistry(build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysApprove()))
    approved_write = await ToolRuntime(approved_registry).execute(ToolCall("write-approved", "write", {"path": "created.txt", "content": "created"}))
    approved_edit = await ToolRuntime(approved_registry).execute(ToolCall("edit-approved", "edit", {"path": "edit.txt", "old_text": "old", "new_text": "new"}))
    assert not approved_write.is_error and not approved_edit.is_error
    assert (workspace_root / "created.txt").read_text(encoding="utf-8") == "created"
    assert (workspace_root / "edit.txt").read_text(encoding="utf-8") == "new value\n"


@pytest.mark.asyncio
async def test_shell_approval_happens_before_process_spawn(workspace_root: Path):
    workspace = Workspace(workspace_root)
    marker = workspace_root / "marker.txt"
    command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'marker.txt\').write_text(\'ran\')"'
    denied = ToolRegistry(build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny()))
    denied_result = await ToolRuntime(denied).execute(ToolCall("shell-denied", "shell", {"command": command}))
    assert denied_result.is_error is True
    assert not marker.exists()

    approved = ToolRegistry(build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysApprove()))
    approved_result = await ToolRuntime(approved).execute(ToolCall("shell-approved", "shell", {"command": command}))
    assert not approved_result.is_error
    assert "exit_code: 0" in approved_result.text
    assert marker.read_text(encoding="utf-8") == "ran"


def test_approval_summaries_are_stable_and_bounded(workspace_root: Path):
    workspace = Workspace(workspace_root)
    tools = {tool.tool.name: tool for tool in build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny())}
    write_request = tools["write"].approval_request({"path": "src/config.py", "content": "x" * 1234}, "reason")
    edit_request = tools["edit"].approval_request({"path": "src/config.py", "old_text": "a", "new_text": "b"}, "reason")
    shell_request = tools["shell"].approval_request({"command": "pytest tests/test_auth.py"}, "reason")
    assert write_request.summary == "Modify file:\nsrc/config.py\ncontent size: 1234 chars"
    assert edit_request.summary == "Edit file:\nsrc/config.py\nreplace_all: false"
    assert shell_request.summary == f"Run command:\npytest tests/test_auth.py\ncwd:\n{workspace.root}"


@pytest.mark.asyncio
async def test_controlled_tools_integrate_with_agent_without_changing_agent_core(workspace_root: Path):
    workspace = Workspace(workspace_root)

    async def allowed_stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("read", "read", {"path": "read.txt"})], stop_reason="tool_calls"))
            return
        assert any(isinstance(message, ToolResultMessage) and "observed" in message.text for message in context.messages)
        yield StreamDone(AssistantMessage([TextBlock("allowed read observed")]))

    allowed_agent = Agent(Model(provider="mock"), "", build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny()), allowed_stream)
    assert (await allowed_agent.run([UserMessage("read")]))[-1].text == "allowed read observed"

    async def denied_stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("edit", "edit", {"path": "edit.txt", "old_text": "old", "new_text": "new"})], stop_reason="tool_calls"))
            return
        result = next(message for message in context.messages if isinstance(message, ToolResultMessage))
        assert result.is_error is True
        yield StreamDone(AssistantMessage([TextBlock("approval denial observed")]))

    denied_agent = Agent(Model(provider="mock"), "", build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny()), denied_stream)
    assert (await denied_agent.run([UserMessage("edit")]))[-1].text == "approval denial observed"
    assert (workspace_root / "edit.txt").read_text(encoding="utf-8") == "old value\n"

    async def denied_write_stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("write", "write", {"path": "agent-created.txt", "content": "no"})], stop_reason="tool_calls"))
            return
        assert next(message for message in context.messages if isinstance(message, ToolResultMessage)).is_error
        yield StreamDone(AssistantMessage([TextBlock("write denial observed")]))

    denied_write_agent = Agent(Model(provider="mock"), "", build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny()), denied_write_stream)
    assert (await denied_write_agent.run([UserMessage("write")]))[-1].text == "write denial observed"
    assert not (workspace_root / "agent-created.txt").exists()

    async def approved_edit_stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("edit-approved", "edit", {"path": "edit.txt", "old_text": "old", "new_text": "approved"})], stop_reason="tool_calls"))
            return
        assert not next(message for message in context.messages if isinstance(message, ToolResultMessage)).is_error
        yield StreamDone(AssistantMessage([TextBlock("approved edit observed")]))

    approved_edit_agent = Agent(Model(provider="mock"), "", build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysApprove()), approved_edit_stream)
    assert (await approved_edit_agent.run([UserMessage("edit")]))[-1].text == "approved edit observed"
    assert (workspace_root / "edit.txt").read_text(encoding="utf-8") == "approved value\n"

    marker = workspace_root / "agent-marker.txt"
    command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'agent-marker.txt\').write_text(\'ran\')"'

    async def denied_shell_stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("shell", "shell", {"command": command})], stop_reason="tool_calls"))
            return
        assert next(message for message in context.messages if isinstance(message, ToolResultMessage)).is_error
        yield StreamDone(AssistantMessage([TextBlock("shell denial observed")]))

    denied_shell_agent = Agent(Model(provider="mock"), "", build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysDeny()), denied_shell_stream)
    assert (await denied_shell_agent.run([UserMessage("shell")]))[-1].text == "shell denial observed"
    assert not marker.exists()


def test_build_controlled_coding_tools_wraps_all_six_with_shared_dependencies(workspace_root: Path):
    workspace = Workspace(workspace_root)
    policy = DefaultCodingToolPolicy()
    approval = AlwaysDeny()
    tools = build_controlled_coding_tools(workspace, policy, approval)
    assert [tool.tool.name for tool in tools] == ["read", "list_dir", "search", "write", "edit", "shell"]
    assert all(tool.policy is policy and tool.approval_handler is approval for tool in tools)
    assert [tool.tool.required for tool in tools] == [("path",), (), ("query",), None, ("path", "old_text", "new_text"), ("command",)]
