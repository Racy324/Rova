"""Runtime lifecycle hooks for tool execution.

Lifecycle hooks are a control surface owned by the runtime.  They are distinct
from :class:`AgentEvent` subscribers: a hook can affect the next lifecycle
step, while an event subscriber only observes an already emitted fact.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, TypeAlias


class ToolHookPoint(str, Enum):
    """Supported tool lifecycle intervention points."""

    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    TOOL_FAILURE = "tool_failure"


class LifecycleHookError(RuntimeError):
    """A lifecycle hook failed or returned an invalid control outcome."""

    def __init__(self, point: ToolHookPoint, source: str, detail: str) -> None:
        super().__init__(f"Lifecycle hook '{source}' failed at {point.value}: {detail}")
        self.point = point
        self.source = source


@dataclass(frozen=True)
class PreToolUseContext:
    """Validated tool input offered to a ``pre_tool_use`` hook."""

    arguments: Mapping[str, Any]
    tool_name: str | None = None
    tool_call_id: str | None = None
    call_index: int | None = None
    run_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class PreToolUseContinue:
    """Continue preflight, optionally replacing validated arguments."""

    arguments: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreToolUseBlock:
    """Stop this tool call before policy and execution."""

    message: str = "Tool use was blocked by a lifecycle hook."
    metadata: Mapping[str, Any] | None = None


PreToolUseOutcome: TypeAlias = PreToolUseContinue | PreToolUseBlock | None


@dataclass(frozen=True)
class PostToolUseContext:
    """Successful raw tool output offered before model-output processing."""

    tool_name: str
    tool_call_id: str
    call_index: int
    arguments: Mapping[str, Any]
    content: tuple[Any, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PostToolUseContinue:
    """Keep or replace successful raw output before canonicalization."""

    content: tuple[Any, ...] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolFailureContext:
    """A declared, provider-visible tool failure before result finalization."""

    stage: str
    outcome: str
    tool_name: str
    tool_call_id: str
    call_index: int
    arguments: Mapping[str, Any] | None
    message: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ToolFailureContinue:
    """Attach diagnostic metadata without changing a failure's meaning."""

    metadata: Mapping[str, Any] | None = None


PreToolUseHandler: TypeAlias = Callable[[PreToolUseContext], PreToolUseOutcome | Awaitable[PreToolUseOutcome]]
PostToolUseHandler: TypeAlias = Callable[[PostToolUseContext], PostToolUseContinue | None | Awaitable[PostToolUseContinue | None]]
ToolFailureHandler: TypeAlias = Callable[[ToolFailureContext], ToolFailureContinue | None | Awaitable[ToolFailureContinue | None]]
ToolHookHandler: TypeAlias = PreToolUseHandler | PostToolUseHandler | ToolFailureHandler
ArgumentsValidator: TypeAlias = Callable[[Mapping[str, Any]], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class _Registration:
    point: ToolHookPoint
    handler: ToolHookHandler
    source: str


class HookRegistry:
    """Small ordered registry for runtime lifecycle hooks.

    Registration order is execution order.  This deliberately is not a
    general-purpose event bus.
    """

    def __init__(self) -> None:
        self._registrations: list[_Registration] = []

    def register(
        self,
        point: ToolHookPoint,
        handler: ToolHookHandler,
        *,
        source: str,
    ) -> Callable[[], None]:
        registration = _Registration(point=point, handler=handler, source=source)
        self._registrations.append(registration)
        removed = False

        def unregister() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            try:
                self._registrations.remove(registration)
            except ValueError:
                pass

        return unregister

    async def dispatch_pre_tool_use(
        self,
        arguments: Mapping[str, Any],
        *,
        context: PreToolUseContext | None = None,
        validate: ArgumentsValidator | None = None,
    ) -> PreToolUseContinue | PreToolUseBlock:
        """Run pre-tool hooks in order and pass modifications downstream.

        ``validate`` is intentionally injected by ``ToolRuntime`` so every
        modification is validated before the next hook can observe it.
        """

        current_arguments = dict(arguments)
        base_context = context or PreToolUseContext(arguments=current_arguments)
        for registration in self._matching(ToolHookPoint.PRE_TOOL_USE):
            hook_context = replace(base_context, arguments=current_arguments)
            outcome = await self._invoke(registration, hook_context)
            if outcome is None:
                continue
            if isinstance(outcome, PreToolUseBlock):
                return outcome
            if not isinstance(outcome, PreToolUseContinue):
                raise LifecycleHookError(
                    ToolHookPoint.PRE_TOOL_USE,
                    registration.source,
                    "it returned an unsupported outcome",
                )
            if outcome.arguments is not None:
                current_arguments = dict(outcome.arguments)
                if validate is not None:
                    current_arguments = dict(await _maybe_await(validate(current_arguments)))

        return PreToolUseContinue(arguments=current_arguments)

    async def dispatch_post_tool_use(
        self,
        context: PostToolUseContext,
    ) -> PostToolUseContinue:
        current_content = context.content
        current_metadata = dict(context.metadata)
        for registration in self._matching(ToolHookPoint.POST_TOOL_USE):
            hook_context = replace(
                context,
                content=current_content,
                metadata=MappingProxyType(current_metadata),
            )
            outcome = await self._invoke(registration, hook_context)
            if outcome is None:
                continue
            if not isinstance(outcome, PostToolUseContinue):
                raise LifecycleHookError(
                    ToolHookPoint.POST_TOOL_USE,
                    registration.source,
                    "it returned an unsupported outcome",
                )
            if outcome.content is not None:
                current_content = tuple(outcome.content)
            if outcome.metadata is not None:
                _append_metadata(
                    current_metadata,
                    outcome.metadata,
                    point=ToolHookPoint.POST_TOOL_USE,
                    source=registration.source,
                )
        return PostToolUseContinue(content=current_content, metadata=current_metadata)

    async def dispatch_tool_failure(
        self,
        context: ToolFailureContext,
    ) -> ToolFailureContinue:
        current_metadata = dict(context.metadata)
        for registration in self._matching(ToolHookPoint.TOOL_FAILURE):
            hook_context = replace(context, metadata=MappingProxyType(current_metadata))
            outcome = await self._invoke(registration, hook_context)
            if outcome is None:
                continue
            if not isinstance(outcome, ToolFailureContinue):
                raise LifecycleHookError(
                    ToolHookPoint.TOOL_FAILURE,
                    registration.source,
                    "it returned an unsupported outcome",
                )
            if outcome.metadata is not None:
                _append_metadata(
                    current_metadata,
                    outcome.metadata,
                    point=ToolHookPoint.TOOL_FAILURE,
                    source=registration.source,
                )
        return ToolFailureContinue(metadata=current_metadata)

    def _matching(self, point: ToolHookPoint) -> list[_Registration]:
        return [registration for registration in self._registrations if registration.point is point]

    async def _invoke(self, registration: _Registration, context: Any) -> Any:
        try:
            return await _maybe_await(registration.handler(context))
        except asyncio.CancelledError:
            raise
        except LifecycleHookError:
            raise
        except BaseException as exc:
            raise LifecycleHookError(
                registration.point,
                registration.source,
                type(exc).__name__,
            ) from exc


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _append_metadata(
    current: dict[str, Any],
    additions: Mapping[str, Any],
    *,
    point: ToolHookPoint,
    source: str,
) -> None:
    """Allow lifecycle hooks to add diagnostics, never rewrite Runtime facts."""

    for key, value in additions.items():
        if key in current:
            raise LifecycleHookError(point, source, f"it attempted to override metadata key '{key}'")
        current[key] = value
