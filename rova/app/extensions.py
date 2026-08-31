from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any, Iterator

from rova.agent_core.agent import Agent
from rova.agent_core.events import AgentEvent
from rova.agent_core.tools import AgentTool

from .paths import RovaDataPaths


_EVENT_TYPES = frozenset({
    "agent_start",
    "turn_start",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_end",
    "turn_end",
    "agent_end",
    "provider_error",
})


class ExtensionRegistrationError(ValueError):
    """An Extension used an unsupported or conflicting public API operation."""


@dataclass(frozen=True)
class ContextContribution:
    """One dynamic Context section supplied by a trusted local Extension."""

    name: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("context contribution name must be a non-empty string")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("context contribution content must be a non-empty string")


@dataclass(frozen=True)
class ExtensionIssue:
    extension_name: str
    phase: str
    message: str


@dataclass(frozen=True)
class ExtensionLoadReport:
    loaded: tuple[str, ...] = ()
    issues: tuple[ExtensionIssue, ...] = ()


@dataclass(frozen=True)
class _EventHook:
    extension_name: str
    event_type: str
    handler: Callable[[AgentEvent], object]


@dataclass(frozen=True)
class _ContextProvider:
    extension_name: str
    provider: Callable[[], ContextContribution | None]


class ExtensionAPI:
    """The intentionally small public surface available to local Extensions."""

    def __init__(self, reserved_tool_names: Sequence[str]) -> None:
        self._tools: list[AgentTool] = []
        self._tool_names = set(reserved_tool_names)
        self._hooks: list[_EventHook] = []
        self._context_providers: list[_ContextProvider] = []
        self._active_extension: str | None = None
        self._runtime_issues: list[ExtensionIssue] = []

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools)

    @property
    def runtime_issues(self) -> tuple[ExtensionIssue, ...]:
        return tuple(self._runtime_issues)

    def register_tool(self, tool: AgentTool) -> None:
        extension_name = self._require_active_extension()
        if not isinstance(tool, AgentTool):
            raise ExtensionRegistrationError("register_tool requires an AgentTool")
        name = tool.tool.name
        if name in self._tool_names:
            raise ExtensionRegistrationError(f"duplicate tool name: {name}")
        self._tool_names.add(name)
        self._tools.append(tool)

    def on(self, event_type: str, handler: Callable[[AgentEvent], object]) -> None:
        extension_name = self._require_active_extension()
        if event_type not in _EVENT_TYPES:
            raise ExtensionRegistrationError(f"unsupported AgentEvent type: {event_type}")
        if not callable(handler):
            raise ExtensionRegistrationError("event handler must be callable")
        self._hooks.append(_EventHook(extension_name, event_type, handler))

    def register_context_provider(self, provider: Callable[[], ContextContribution | None]) -> None:
        extension_name = self._require_active_extension()
        if not callable(provider):
            raise ExtensionRegistrationError("context provider must be callable")
        self._context_providers.append(_ContextProvider(extension_name, provider))

    @contextmanager
    def extension_setup(self, extension_name: str) -> Iterator[None]:
        if self._active_extension is not None:
            raise RuntimeError("nested Extension setup is not supported")
        tool_count = len(self._tools)
        hook_count = len(self._hooks)
        provider_count = len(self._context_providers)
        tool_names = set(self._tool_names)
        self._active_extension = extension_name
        try:
            yield
        except BaseException:
            del self._tools[tool_count:]
            del self._hooks[hook_count:]
            del self._context_providers[provider_count:]
            self._tool_names = tool_names
            raise
        finally:
            self._active_extension = None

    def bind_event_hooks(self, agent: Agent) -> Callable[[], None]:
        async def dispatch(event: AgentEvent) -> None:
            for hook in self._hooks:
                if hook.event_type != event.type:
                    continue
                try:
                    outcome = hook.handler(event)
                    if inspect.isawaitable(outcome):
                        await outcome
                except Exception as error:
                    self._runtime_issues.append(
                        ExtensionIssue(hook.extension_name, "event", _safe_error_message(error))
                    )

        return agent.subscribe(dispatch)

    def render_context_sections(self) -> list[str]:
        sections: list[str] = []
        for item in self._context_providers:
            try:
                contribution = item.provider()
                if inspect.isawaitable(contribution):
                    raise TypeError("context provider must return synchronously")
                if contribution is None:
                    continue
                if not isinstance(contribution, ContextContribution):
                    raise TypeError("context provider must return ContextContribution or None")
            except Exception as error:
                self._runtime_issues.append(
                    ExtensionIssue(item.extension_name, "context", _safe_error_message(error))
                )
                continue
            sections.append("\n".join([
                "Extension context:",
                f"Name: {contribution.name}",
                "This is dynamic context contributed by a local Extension, not stable system instructions.",
                "",
                contribution.content,
            ]))
        return sections

    def _require_active_extension(self) -> str:
        if self._active_extension is None:
            raise ExtensionRegistrationError("Extension API can only be used from setup(api)")
        return self._active_extension


class ExtensionLoader:
    """Deterministically load independent, trusted local Python Extension files."""

    def __init__(self, roots: Sequence[Path]) -> None:
        self.roots = tuple(Path(root).expanduser() for root in roots)

    @classmethod
    def default_roots(cls, workspace_root: Path | None) -> tuple[Path, ...]:
        roots = [RovaDataPaths.resolve().extensions]
        if workspace_root is not None:
            roots.append(Path(workspace_root) / ".rova" / "extensions")
        return tuple(roots)

    def load(self, api: ExtensionAPI) -> ExtensionLoadReport:
        loaded: list[str] = []
        issues: list[ExtensionIssue] = []
        for path in self._discover():
            extension_name = path.stem
            try:
                module = self._load_module(path)
            except Exception as error:
                issues.append(ExtensionIssue(extension_name, "import", _safe_error_message(error)))
                continue
            setup = getattr(module, "setup", None)
            if not callable(setup):
                issues.append(ExtensionIssue(extension_name, "setup", "Extension must define callable setup(api)"))
                continue
            try:
                with api.extension_setup(extension_name):
                    setup(api)
            except Exception as error:
                issues.append(ExtensionIssue(extension_name, "setup", _safe_error_message(error)))
                continue
            loaded.append(extension_name)
        return ExtensionLoadReport(tuple(loaded), tuple(issues))

    def _discover(self) -> list[Path]:
        discovered: list[Path] = []
        seen_roots: set[Path] = set()
        for root in self.roots:
            resolved_root = root.resolve()
            if resolved_root in seen_roots or not resolved_root.is_dir():
                continue
            seen_roots.add(resolved_root)
            discovered.extend(
                path for path in sorted(resolved_root.iterdir(), key=lambda item: item.name)
                if path.is_file() and path.suffix == ".py" and not path.name.startswith("_")
            )
        return discovered

    @staticmethod
    def _load_module(path: Path) -> Any:
        digest = sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        module_name = f"_rova_extension_{path.stem}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create module spec for {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module


def _safe_error_message(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
