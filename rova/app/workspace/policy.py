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


class ToolPolicy(Protocol):
    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult: ...


class DefaultCodingToolPolicy:
    """Conservative deterministic policy for the six Phase 4.1 coding tools."""

    _READ_ONLY_TOOLS = frozenset({"read", "list_dir", "search"})
    _MUTATION_TOOLS = frozenset({"write", "edit"})

    def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult:
        if request.tool_name in self._READ_ONLY_TOOLS:
            return ToolPolicyResult(ToolPolicyDecision.ALLOW, "read-only workspace operation")
        if request.tool_name in self._MUTATION_TOOLS:
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "this operation modifies workspace files")
        if request.tool_name == "shell":
            return ToolPolicyResult(ToolPolicyDecision.REQUIRE_APPROVAL, "shell commands require approval")
        return ToolPolicyResult(ToolPolicyDecision.DENY, "tool is not permitted by this policy")
