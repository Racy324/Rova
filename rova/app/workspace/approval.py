from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol


class ApprovalDecision(Enum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: Mapping[str, Any]
    policy_reason: str
    summary: str


class ApprovalHandler(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision: ...


class AlwaysApprove:
    """Simple injected approval handler for tests and controlled callers."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE


class AlwaysDeny:
    """Simple injected approval handler for tests and controlled callers."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.DENY


class ConsoleApprovalHandler:
    """Minimal asynchronous console approval with an explicit fail-closed default."""

    def __init__(
        self,
        reader: Callable[[str], str] = input,
        writer: Callable[[str], None] = print,
        shell_executor: str | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._shell_executor = shell_executor

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self._writer("--------------------------------------------------")
        self._writer("Agent requests a controlled operation")
        self._writer(f"Tool:\n{request.tool_name}")
        self._writer(f"Reason:\n{request.policy_reason}")
        self._render_operation_details(request)
        self._writer("--------------------------------------------------")
        answer = await asyncio.to_thread(self._reader, "Approve? [y/N]: ")
        return ApprovalDecision.APPROVE if answer.strip().lower() in {"y", "yes"} else ApprovalDecision.DENY

    def _render_operation_details(self, request: ApprovalRequest) -> None:
        if request.tool_name == "shell":
            self._writer(f"Command:\n{request.arguments.get('command', '')}")
            cwd = request.summary.partition("\ncwd:\n")[2]
            if cwd:
                self._writer(f"cwd:\n{cwd}")
            if self._shell_executor is not None:
                self._writer(f"Shell executor:\n{self._shell_executor}")
            self._writer("Permission:\nApprove this command only")
            self._writer("Warning:\nShell commands are not sandboxed and may access external resources.")
            return
        if request.tool_name in {"write", "edit"}:
            self._writer(f"File:\n{request.arguments.get('path', '')}")
            self._writer("Operation:\nModify workspace file")
            self._writer("Permission:\nApprove this change only")
            return
        self._writer(f"Details:\n{request.summary}")
