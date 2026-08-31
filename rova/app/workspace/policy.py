from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol


class ToolPolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class ToolPolicyResult:
    decision: ToolPolicyDecision
    reason: str


@dataclass(frozen=True)
class ToolExecutionRequest:
    tool_name: str
    arguments: Mapping[str, Any]
    metadata: Mapping[str, str] | None = None


class ToolPolicy(Protocol):
    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult: ...


class DefaultCodingToolPolicy:
    """Conservative deterministic policy for the six Phase 4.1 coding tools."""

    _READ_ONLY_TOOLS = frozenset({"read", "list_dir", "search"})
    _MUTATION_TOOLS = frozenset({"write", "edit"})

    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult:
        if request.metadata is not None and request.metadata.get("origin") == "mcp":
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "MCP tools require approval by default")
        if request.tool_name in self._READ_ONLY_TOOLS:
            return ToolPolicyResult(ToolPolicyDecision.ALLOW, "read-only workspace operation")
        if request.tool_name in self._MUTATION_TOOLS:
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "this operation modifies workspace files")
        if request.tool_name == "shell":
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "shell commands require approval")
        return ToolPolicyResult(ToolPolicyDecision.DENY, "tool is not permitted by this policy")


class DefaultRovaToolPolicy:
    """Product default policy while retaining the coding policy's Workspace rules."""

    _WORKSPACE_TOOLS = frozenset({"read", "list_dir", "search", "write", "edit", "shell"})

    def __init__(self, workspace_policy: ToolPolicy | None = None) -> None:
        self._workspace_policy = workspace_policy or DefaultCodingToolPolicy()

    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult:
        if request.metadata is not None and request.metadata.get("origin") == "mcp":
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "MCP tools require approval by default")
        if request.tool_name in self._WORKSPACE_TOOLS:
            return self._workspace_policy.evaluate(request)
        return ToolPolicyResult(ToolPolicyDecision.ALLOW, "non-workspace product tool")
