from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError

from .approval import ApprovalDecision, ApprovalHandler, ApprovalRequest
from .policy import ToolExecutionRequest, ToolPolicy, ToolPolicyDecision
from .tools import (
    create_edit_tool,
    create_list_dir_tool,
    create_read_tool,
    create_search_tool,
    create_shell_tool,
    create_write_tool,
)
from .terminal import LocalTerminalBackend, TerminalBackend, TerminalEnvironment
from .workspace import Workspace
from .context import WorkspaceContext


class ToolExecutionDenied(ToolExecutionError):
    """A normal controlled-execution denial mapped by ToolRegistry to a tool error."""


@dataclass
class ControlledTool:
    """AgentTool-compatible policy and approval wrapper around one executable tool."""

    inner: AgentTool
    policy: ToolPolicy
    approval_handler: ApprovalHandler | None = None
    workspace_root: Path | None = None
    workspace_context: WorkspaceContext | None = None
    terminal_environment: TerminalEnvironment | None = None

    @property
    def tool(self):
        return self.inner.tool

    async def execute(self, tool_call_id: str, params: dict) -> AgentToolResult:
        result = self.policy.evaluate(ToolExecutionRequest(self.tool.name, params, self.inner.metadata))
        if result.decision is ToolPolicyDecision.ALLOW:
            return await self._execute_inner(tool_call_id, params, result, None, None)
        if result.decision is ToolPolicyDecision.DENY:
            raise ToolExecutionDenied(f"Tool execution denied by policy.\nReason: {result.reason}", metadata={"outcome": "policy_denied", "policy_decision": result.decision.value, "policy_reason": result.reason})
        if result.decision is ToolPolicyDecision.REQUIRE_APPROVAL:
            approval = await self._require_approval(params, result.reason)
            return await self._execute_inner(tool_call_id, params, result, True, approval)
        raise RuntimeError(f"unsupported tool policy decision: {result.decision!r}")

    def approval_request(self, arguments: Mapping[str, Any], policy_reason: str) -> ApprovalRequest:
        return ApprovalRequest(self.tool.name, arguments, policy_reason, self._summary(arguments))

    async def _require_approval(self, params: dict, reason: str) -> str:
        if self.approval_handler is None:
            raise ToolExecutionDenied("Tool execution requires approval, but no approval handler is configured.", metadata={"outcome": "approval_unavailable", "policy_decision": "require_approval", "policy_reason": reason, "approval_required": True, "approval_decision": "unavailable"})
        try:
            decision = await self.approval_handler.request_approval(self.approval_request(params, reason))
        except Exception as error:
            raise ToolExecutionDenied("Tool approval failed; execution was denied.", metadata={"outcome": "approval_error", "policy_decision": "require_approval", "policy_reason": reason, "approval_required": True, "approval_decision": "error"}) from error
        if decision is not ApprovalDecision.APPROVE:
            raise ToolExecutionDenied("Tool execution was not approved.", metadata={"outcome": "approval_denied", "policy_decision": "require_approval", "policy_reason": reason, "approval_required": True, "approval_decision": "deny"})
        return "approve"

    def _with_policy(self, result: AgentToolResult, policy_result, approval_required: bool | None, approval_decision: str | None) -> AgentToolResult:
        metadata = {
            **self.inner.metadata,
            **result.metadata,
            "policy_decision": policy_result.decision.value,
            "policy_reason": policy_result.reason,
        }
        if approval_required is not None:
            metadata["approval_required"] = approval_required
            metadata["approval_decision"] = approval_decision
        return AgentToolResult(result.content, metadata)

    async def _execute_inner(self, tool_call_id: str, params: dict, policy_result, approval_required: bool | None, approval_decision: str | None) -> AgentToolResult:
        try:
            write_existed_before = self._write_existed_before(params)
            result = await self.inner.execute(tool_call_id, params)
        except ToolExecutionError as error:
            metadata = {
                "outcome": "tool_execution_error",
                **self.inner.metadata,
                **error.metadata,
                "policy_decision": policy_result.decision.value,
                "policy_reason": policy_result.reason,
            }
            if approval_required is not None:
                metadata["approval_required"] = approval_required
                metadata["approval_decision"] = approval_decision
            raise ToolExecutionError(str(error), metadata=metadata) from error
        self._record_successful_file_operation(params, write_existed_before)
        return self._with_policy(result, policy_result, approval_required, approval_decision)

    def _write_existed_before(self, params: Mapping[str, Any]) -> bool | None:
        if self.tool.name != "write" or self.workspace_context is None:
            return None
        return self.workspace_context.workspace.resolve(params["path"]).exists()

    def _record_successful_file_operation(self, params: Mapping[str, Any], write_existed_before: bool | None) -> None:
        if self.workspace_context is None:
            return
        if self.tool.name == "read":
            self.workspace_context.record_read(self.workspace_context.workspace.resolve(params["path"]))
        elif self.tool.name == "write":
            assert write_existed_before is not None
            self.workspace_context.record_write(
                self.workspace_context.workspace.resolve(params["path"]),
                existed_before=write_existed_before,
            )
        elif self.tool.name == "edit":
            self.workspace_context.record_edit(self.workspace_context.workspace.resolve(params["path"]))

    def _summary(self, arguments: Mapping[str, Any]) -> str:
        if self.tool.name == "write":
            return f"Modify file:\n{arguments.get('path', '')}\ncontent size: {len(arguments.get('content', ''))} chars"
        if self.tool.name == "edit":
            replace_all = str(arguments.get("replace_all", False)).lower()
            return f"Edit file:\n{arguments.get('path', '')}\nreplace_all: {replace_all}"
        if self.tool.name == "shell":
            details = (
                self.terminal_environment.approval_summary
                if self.terminal_environment is not None
                else f"cwd:\n{self.workspace_root if self.workspace_root is not None else '<workspace root>'}"
            )
            return f"Run command:\n{arguments.get('command', '')}\n{details}"
        if self.inner.metadata.get("origin") == "mcp":
            return "MCP tool:\n{0}\nserver:\n{1}\noperation:\n{2}".format(
                self.inner.metadata.get("public_name", self.tool.name),
                self.inner.metadata.get("server_id", "unknown"),
                self.inner.metadata.get("raw_tool_name", "unknown"),
            )
        return f"Approval required for tool: {self.tool.name}"


def build_controlled_coding_tools(
    workspace: Workspace,
    policy: ToolPolicy,
    approval_handler: ApprovalHandler | None = None,
    workspace_context: WorkspaceContext | None = None,
    terminal_backend: TerminalBackend | None = None,
) -> list[ControlledTool]:
    """Build the only six-tool composition entry point intended for a coding runtime."""
    effective_terminal_backend = terminal_backend or LocalTerminalBackend(workspace)
    raw_tools = [
        create_read_tool(workspace),
        create_list_dir_tool(workspace),
        create_search_tool(workspace),
        create_write_tool(workspace),
        create_edit_tool(workspace),
        create_shell_tool(effective_terminal_backend),
    ]
    return [
        ControlledTool(
            tool,
            policy,
            approval_handler,
            workspace.root,
            workspace_context,
            effective_terminal_backend.environment if tool.tool.name == "shell" else None,
        )
        for tool in raw_tools
    ]


def wrap_controlled_tools(
    tools: Sequence[AgentTool | ControlledTool],
    policy: ToolPolicy,
    approval_handler: ApprovalHandler | None,
) -> list[ControlledTool]:
    """Apply the product's single execution boundary without double wrapping Workspace tools."""
    return [
        tool if isinstance(tool, ControlledTool) else ControlledTool(tool, policy, approval_handler)
        for tool in tools
    ]
