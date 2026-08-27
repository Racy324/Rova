from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from rova.ai.messages import TextBlock, ToolCall, ToolResultMessage
from rova.ai.tools import Tool, validate_tool_arguments

from .tool_output import ToolOutputProcessor, ToolOutputScope


class ToolExecutionError(Exception):
    """A declared, provider-visible failure from a tool operation.

    Unexpected exceptions are harness failures and must propagate rather than
    being committed as ordinary tool results.
    """

    def __init__(self, message: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = _metadata(metadata or {})


@dataclass
class AgentToolResult:
    content: list[TextBlock]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTool:
    tool: Tool
    execute: Callable[[str, dict], Awaitable[AgentToolResult]]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Call-local identity and validated input exposed to Tool Runtime middleware."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    run_id: str | None
    session_id: str | None


class ToolMiddleware(Protocol):
    async def before_tool_execute(self, context: ToolExecutionContext) -> None: ...

    async def after_tool_execute(self, context: ToolExecutionContext, result: ToolResultMessage) -> None: ...


class ToolRegistry:
    def __init__(
        self,
        tools: list[AgentTool],
        *,
        tool_output_processor: ToolOutputProcessor | None = None,
        middlewares: Sequence[ToolMiddleware] = (),
    ) -> None:
        self._tools = {agent_tool.tool.name: agent_tool for agent_tool in tools}
        self._tool_output_processor = tool_output_processor
        self._middlewares = tuple(middlewares)

    @property
    def schemas(self) -> list[Tool]:
        return [agent_tool.tool for agent_tool in self._tools.values()]

    async def execute(self, tool_call: ToolCall, *, scope: ToolOutputScope | None = None) -> ToolResultMessage:
        agent_tool = self._tools.get(tool_call.name)
        if agent_tool is None:
            return self._error(tool_call, f"Unknown tool: {tool_call.name}", {"outcome": "tool_input_error"}, scope)
        try:
            arguments = validate_tool_arguments(agent_tool.tool, tool_call.arguments)
        except ValueError as error:
            return self._error(tool_call, str(error), {"outcome": "tool_input_error"}, scope)
        context = ToolExecutionContext(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=MappingProxyType(dict(arguments)),
            run_id=scope.run_id if scope is not None else None,
            session_id=scope.session_id if scope is not None else None,
        )
        try:
            for middleware in self._middlewares:
                await middleware.before_tool_execute(context)
        except ToolExecutionError as error:
            return self._error(tool_call, str(error), error.metadata or {"outcome": "tool_execution_error"}, scope)
        try:
            result = await agent_tool.execute(tool_call.id, arguments)
            final_result = self._finalize(tool_call, result.content, False, {"outcome": "success", **result.metadata}, scope)
        except ToolExecutionError as error:
            final_result = self._error(tool_call, str(error), error.metadata or {"outcome": "tool_execution_error"}, scope)
        for middleware in self._middlewares:
            await middleware.after_tool_execute(context, copy.deepcopy(final_result))
        return final_result

    def _error(
        self,
        tool_call: ToolCall,
        text: str,
        metadata: Mapping[str, Any],
        scope: ToolOutputScope | None = None,
    ) -> ToolResultMessage:
        return self._finalize(tool_call, [TextBlock(text)], True, metadata, scope)

    def _finalize(
        self,
        tool_call: ToolCall,
        content: list[TextBlock],
        is_error: bool,
        metadata: Mapping[str, Any],
        scope: ToolOutputScope | None,
    ) -> ToolResultMessage:
        final_metadata = dict(metadata)
        final_content = content
        if self._tool_output_processor is not None:
            processed = self._tool_output_processor.process(
                tool_call.id,
                tool_call.name,
                content,
                final_metadata,
                is_error=is_error,
                scope=scope,
            )
            final_content = processed.preview
            final_metadata["tool_output"] = processed.metadata.to_dict()
        return ToolResultMessage(
            tool_call.id,
            tool_call.name,
            final_content,
            is_error=is_error,
            metadata=_metadata(final_metadata),
        )


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    if not _json_value(copied):
        raise TypeError("tool metadata must be JSON-compatible")
    return copied


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    return False
