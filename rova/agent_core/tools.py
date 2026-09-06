from __future__ import annotations

import copy
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
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


class ToolExecutionMode(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass
class AgentTool:
    tool: Tool
    execute: Callable[[str, dict], Awaitable[AgentToolResult]]
    metadata: dict[str, str] = field(default_factory=dict)
    execution_mode: ToolExecutionMode | None = None


def resolve_batch_mode(
    runtime_mode: ToolExecutionMode,
    tools: Sequence[AgentTool | None],
) -> ToolExecutionMode:
    if runtime_mode is ToolExecutionMode.SEQUENTIAL:
        return ToolExecutionMode.SEQUENTIAL
    if any(tool is not None and tool.execution_mode is ToolExecutionMode.SEQUENTIAL for tool in tools):
        return ToolExecutionMode.SEQUENTIAL
    return ToolExecutionMode.PARALLEL


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


class PreToolUseHook(Protocol):
    """Internal lifecycle seam between schema validation and controlled execution."""

    async def pre_tool_use(self, context: ToolExecutionContext) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class ToolGovernancePreparation:
    """Opaque product governance state created during ToolRuntime preflight."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    state: Any = None


class ToolGovernance(Protocol):
    """Product policy/approval authority invoked by ToolRuntime at fixed seams."""

    async def preflight(
        self,
        context: ToolExecutionContext,
        agent_tool: AgentTool,
    ) -> ToolGovernancePreparation: ...

    async def after_success(
        self,
        prepared: "PreparedToolCall",
        result: AgentToolResult,
    ) -> AgentToolResult: ...

    def enrich_error(
        self,
        prepared: "PreparedToolCall",
        error: ToolExecutionError,
    ) -> ToolExecutionError: ...


@dataclass(frozen=True)
class PreparedToolCall:
    tool_call: ToolCall
    call_index: int
    agent_tool: AgentTool
    arguments: Mapping[str, Any]
    context: ToolExecutionContext
    execution_mode: ToolExecutionMode
    scope: ToolOutputScope | None
    governance: ToolGovernancePreparation | None = None


class ToolRegistry:
    def __init__(self, tools: Sequence[AgentTool]) -> None:
        batch = tuple(tools)
        self._validate_registration_batch(batch)
        self._tools = {agent_tool.tool.name: agent_tool for agent_tool in batch}

    @property
    def schemas(self) -> list[Tool]:
        return [agent_tool.tool for agent_tool in self._tools.values()]

    def register_tools(self, tools: Sequence[AgentTool]) -> None:
        batch = tuple(tools)
        self._validate_registration_batch(batch)
        names = [tool.tool.name for tool in batch]
        duplicate = next((name for name in names if name in self._tools), None)
        if duplicate is not None:
            raise ValueError(f"duplicate tool name: {duplicate}")
        self._tools = {**self._tools, **{tool.tool.name: tool for tool in batch}}

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    @staticmethod
    def _validate_registration_batch(tools: Sequence[AgentTool]) -> None:
        names = [tool.tool.name for tool in tools]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool name in registration batch")


class ToolRuntime:
    """Execute registered tools through the product's common tool pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tool_output_processor: ToolOutputProcessor | None = None,
        middlewares: Sequence[ToolMiddleware] = (),
        pre_tool_hooks: Sequence[PreToolUseHook] = (),
        governance: ToolGovernance | None = None,
    ) -> None:
        self._registry = registry
        self._tool_output_processor = tool_output_processor
        self._middlewares = tuple(middlewares)
        self._pre_tool_hooks = tuple(pre_tool_hooks)
        self._governance = governance

    async def preflight(
        self,
        tool_call: ToolCall,
        *,
        call_index: int = 0,
        scope: ToolOutputScope | None = None,
    ) -> PreparedToolCall | ToolResultMessage:
        agent_tool = self._registry.get(tool_call.name)
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
            for hook in self._pre_tool_hooks:
                modified_arguments = await hook.pre_tool_use(context)
                if modified_arguments is None:
                    continue
                arguments = validate_tool_arguments(agent_tool.tool, dict(modified_arguments))
                context = ToolExecutionContext(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=MappingProxyType(dict(arguments)),
                    run_id=scope.run_id if scope is not None else None,
                    session_id=scope.session_id if scope is not None else None,
                )
            for middleware in self._middlewares:
                await middleware.before_tool_execute(context)
            governance = (
                await self._governance.preflight(context, agent_tool)
                if self._governance is not None
                else None
            )
        except ValueError as error:
            return self._error(tool_call, str(error), {"outcome": "tool_input_error"}, scope)
        except ToolExecutionError as error:
            return self._error(tool_call, str(error), error.metadata or {"outcome": "tool_execution_error"}, scope)
        return PreparedToolCall(
            tool_call=tool_call,
            call_index=call_index,
            agent_tool=agent_tool,
            arguments=MappingProxyType(dict(arguments)),
            context=context,
            execution_mode=agent_tool.execution_mode or ToolExecutionMode.PARALLEL,
            scope=scope,
            governance=governance,
        )

    async def execute(self, tool_call: ToolCall, *, scope: ToolOutputScope | None = None) -> ToolResultMessage:
        prepared = await self.preflight(tool_call, scope=scope)
        if isinstance(prepared, ToolResultMessage):
            return prepared
        return await self._execute_prepared(prepared)

    async def execute_batch(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        runtime_mode: ToolExecutionMode,
        scope: ToolOutputScope | None = None,
        on_execution_start: Callable[[PreparedToolCall], Awaitable[None]] | None = None,
        on_execution_end: Callable[[PreparedToolCall, ToolResultMessage], Awaitable[None]] | None = None,
        on_result_committed: Callable[[ToolResultMessage], Awaitable[None]] | None = None,
    ) -> list[ToolResultMessage]:
        """Execute one AssistantMessage ToolCall batch without reordering committed results."""
        registered = [self._registry.get(tool_call.name) for tool_call in tool_calls]
        batch_mode = resolve_batch_mode(runtime_mode, registered)
        if batch_mode is ToolExecutionMode.SEQUENTIAL:
            results: list[ToolResultMessage] = []
            for call_index, tool_call in enumerate(tool_calls):
                prepared = await self.preflight(tool_call, call_index=call_index, scope=scope)
                result = (
                    prepared
                    if isinstance(prepared, ToolResultMessage)
                    else await self._execute_observed(prepared, on_execution_start, on_execution_end)
                )
                results.append(result)
                if on_result_committed is not None:
                    await on_result_committed(result)
            return results

        preflight = [
            await self.preflight(tool_call, call_index=call_index, scope=scope)
            for call_index, tool_call in enumerate(tool_calls)
        ]
        tasks: dict[int, asyncio.Task[ToolResultMessage]] = {
            prepared.call_index: asyncio.create_task(
                self._execute_observed(prepared, on_execution_start, on_execution_end)
            )
            for prepared in preflight
            if isinstance(prepared, PreparedToolCall)
        }
        try:
            executed = await asyncio.gather(*tasks.values())
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        executed_by_index = dict(zip(tasks, executed))
        results = [
            item if isinstance(item, ToolResultMessage) else executed_by_index[item.call_index]
            for item in preflight
        ]
        if on_result_committed is not None:
            for result in results:
                await on_result_committed(result)
        return results

    async def _execute_observed(
        self,
        prepared: PreparedToolCall,
        on_execution_start: Callable[[PreparedToolCall], Awaitable[None]] | None,
        on_execution_end: Callable[[PreparedToolCall, ToolResultMessage], Awaitable[None]] | None,
    ) -> ToolResultMessage:
        if on_execution_start is not None:
            await on_execution_start(prepared)
        result = await self._execute_prepared(prepared)
        if on_execution_end is not None:
            await on_execution_end(prepared, result)
        return result

    async def _execute_prepared(self, prepared: PreparedToolCall) -> ToolResultMessage:
        tool_call = prepared.tool_call
        try:
            result = await prepared.agent_tool.execute(tool_call.id, dict(prepared.arguments))
            if self._governance is not None and prepared.governance is not None:
                result = await self._governance.after_success(prepared, result)
            final_result = self._finalize(tool_call, result.content, False, {"outcome": "success", **result.metadata}, prepared.scope)
        except ToolExecutionError as error:
            if self._governance is not None and prepared.governance is not None:
                error = self._governance.enrich_error(prepared, error)
            final_result = self._error(tool_call, str(error), error.metadata or {"outcome": "tool_execution_error"}, prepared.scope)
        for middleware in self._middlewares:
            await middleware.after_tool_execute(prepared.context, copy.deepcopy(final_result))
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
