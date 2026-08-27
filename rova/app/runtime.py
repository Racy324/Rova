from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import platform
import warnings

from rova.ai.context import Context
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.agent_core.tool_output import ToolOutputProcessor
from rova.agent_core.types import StreamFn
from rova.agent_session.agent_session import AgentSession
from rova.agent_session.compaction import CompactionPolicy
from rova.artifacts import FileArtifactStore

from .context.local import LocalResearchContext
from .web.sources import ResearchSourceStore
from .web.tools import WebFetchBackend, WebSearchBackend, create_fetch_webpage_tool, create_web_search_tool
from .workspace.approval import AlwaysApprove, ApprovalHandler, ConsoleApprovalHandler
from .workspace.context import WorkspaceContext
from .workspace.controlled_tool import build_controlled_coding_tools
from .workspace.instructions import WorkspaceInstructionSnapshot, load_workspace_instruction
from .workspace.policy import DefaultCodingToolPolicy, ToolPolicy
from .workspace.workspace import Workspace
from .memory import (
    FileMemoryStore,
    MemoryApplyResult,
    MemorySnapshot,
    MemoryStore,
    MemoryStoreError,
)
from .memory_maintenance import MemoryMaintenanceError, consolidate_memory, extract_memory_update
from .paths import RovaDataPaths
from .skills import FileSkillStore, SkillCatalogSnapshot, SkillStoreError, create_skill_tools


DEFAULT_MAX_TURNS = 16
MAX_PRODUCT_TURNS = 32
DEFAULT_MEMORY_UPDATE_INTERVAL = 3
DEFAULT_MEMORY_MAX_CHARS = 6_000
DEFAULT_MEMORY_CONSOLIDATION_THRESHOLD = 4_800


ROVA_SYSTEM_PROMPT = """You are Rova, a local single-user general agent.

Follow the user's current request while respecting applicable system constraints and relevant project instructions.

Use only available tools. Treat tool results as observations from actual execution; do not invent tools, files, sources, execution results, or verification outcomes. Use available tools when needed, but do not claim an operation or verification succeeded unless the observed result supports that claim.

Clearly distinguish observed facts from your inferences, and state uncertainty when information is insufficient.
"""


@dataclass
class RovaRuntime:
    agent: Agent
    session: AgentSession
    artifact_store: FileArtifactStore
    workspace: Workspace | None
    workspace_context: WorkspaceContext | None
    source_store: ResearchSourceStore | None
    local_context: LocalResearchContext | None
    memory_store: MemoryStore
    memory_snapshot: MemorySnapshot
    workspace_instruction_snapshot: WorkspaceInstructionSnapshot
    skill_store: FileSkillStore
    skill_catalog_snapshot: SkillCatalogSnapshot
    memory_model: Model
    memory_stream_fn: StreamFn
    memory_update_interval: int
    memory_max_chars: int
    memory_consolidation_threshold: int

    def __post_init__(self) -> None:
        self._user_turn_count = 0
        self._memory_listeners: list[Callable[[MemoryObservation], None]] = []
        self._local_context_attached = False

    def subscribe_memory(self, listener: Callable[["MemoryObservation"], None]) -> Callable[[], None]:
        self._memory_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._memory_listeners:
                self._memory_listeners.remove(listener)

        return unsubscribe

    async def prompt(self, text: str):
        first_message_index = len(self.agent.messages)
        request_text = text
        if not self._local_context_attached and self.local_context is not None:
            attachment = self.local_context.render_user_attachment()
            if attachment:
                request_text = f"{text}\n\n{attachment}"
                self._local_context_attached = True
        responses = await self.session.prompt(request_text)
        self._user_turn_count += 1
        if self._user_turn_count % self.memory_update_interval == 0:
            await self._maintain_memory(self.agent.messages[first_message_index:])
        return responses

    async def _maintain_memory(self, recent_messages) -> None:
        self._emit_memory(MemoryObservation("extraction", "triggered"))
        try:
            extraction = await self.memory_store.update(
                lambda latest: extract_memory_update(self.memory_model, self.memory_stream_fn, latest, recent_messages),
                max_chars=self.memory_max_chars,
            )
        except (MemoryMaintenanceError, MemoryStoreError) as error:
            self._emit_memory(MemoryObservation("extraction", "failed", error_message=str(error)))
            return
        self._emit_memory(_memory_observation("extraction", extraction))
        if not _needs_consolidation(extraction.snapshot, self.memory_consolidation_threshold):
            return
        self._emit_memory(MemoryObservation("consolidation", "triggered"))
        try:
            consolidation = await self.memory_store.update(
                lambda latest: consolidate_memory(
                    self.memory_model,
                    self.memory_stream_fn,
                    latest,
                    max_chars=self.memory_max_chars,
                ),
                max_chars=self.memory_max_chars,
            )
        except (MemoryMaintenanceError, MemoryStoreError) as error:
            self._emit_memory(MemoryObservation("consolidation", "failed", error_message=str(error)))
            return
        self._emit_memory(_memory_observation("consolidation", consolidation))

    def _emit_memory(self, event: "MemoryObservation") -> None:
        for listener in list(self._memory_listeners):
            listener(event)


@dataclass(frozen=True)
class MemoryObservation:
    kind: str
    status: str
    changed_documents: tuple[str, ...] = ()
    error_message: str | None = None


def build_rova_runtime(
    *,
    model: Model,
    stream_fn: StreamFn,
    workspace_root: Path | None = None,
    approval_handler: ApprovalHandler | None = None,
    permission_mode: str = "ask",
    policy: ToolPolicy | None = None,
    web_search_backend: WebSearchBackend | None = None,
    webpage_fetcher: WebFetchBackend | None = None,
    local_context: LocalResearchContext | None = None,
    session_root: Path | None = None,
    artifact_root: Path | None = None,
    compaction_policy: CompactionPolicy | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    session_id: str | None = None,
    memory_store: MemoryStore | None = None,
    memory_root: Path | None = None,
    skill_root: Path | None = None,
    memory_model: Model | None = None,
    memory_update_interval: int = DEFAULT_MEMORY_UPDATE_INTERVAL,
    memory_max_chars: int = DEFAULT_MEMORY_MAX_CHARS,
    memory_consolidation_threshold: int = DEFAULT_MEMORY_CONSOLIDATION_THRESHOLD,
) -> RovaRuntime:
    if (web_search_backend is None) != (webpage_fetcher is None):
        raise ValueError("web_search_backend and webpage_fetcher must be provided together")
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or not 1 <= max_turns <= MAX_PRODUCT_TURNS:
        raise ValueError(f"max_turns must be an integer between 1 and {MAX_PRODUCT_TURNS}")
    if permission_mode not in {"ask", "full"}:
        raise ValueError("permission_mode must be 'ask' or 'full'")
    if not isinstance(memory_update_interval, int) or isinstance(memory_update_interval, bool) or memory_update_interval <= 0:
        raise ValueError("memory_update_interval must be a positive integer")
    if not isinstance(memory_max_chars, int) or isinstance(memory_max_chars, bool) or memory_max_chars <= 0:
        raise ValueError("memory_max_chars must be a positive integer")
    if (
        not isinstance(memory_consolidation_threshold, int)
        or isinstance(memory_consolidation_threshold, bool)
        or not 0 < memory_consolidation_threshold <= memory_max_chars
    ):
        raise ValueError("memory_consolidation_threshold must be between 1 and memory_max_chars")

    workspace = Workspace(workspace_root) if workspace_root is not None else None
    workspace_context = WorkspaceContext(workspace) if workspace is not None else None
    effective_memory_store = memory_store or FileMemoryStore(memory_root)
    try:
        memory_snapshot = effective_memory_store.load_snapshot()
    except MemoryStoreError:
        memory_snapshot = MemorySnapshot()
    workspace_instruction_snapshot = load_workspace_instruction(workspace.root if workspace is not None else None)
    effective_skill_store = FileSkillStore(skill_root)
    try:
        skill_catalog_snapshot = effective_skill_store.discover_catalog()
    except SkillStoreError as error:
        warnings.warn(f"Skill catalog unavailable: {error}", RuntimeWarning, stacklevel=2)
        skill_catalog_snapshot = SkillCatalogSnapshot()
    source_store = ResearchSourceStore() if web_search_backend is not None else None
    tools = create_skill_tools(effective_skill_store)
    if workspace is not None:
        effective_policy = DefaultCodingToolPolicy() if policy is None else policy
        effective_approval_handler = approval_handler or _approval_handler_for_mode(permission_mode)
        tools.extend(build_controlled_coding_tools(workspace, effective_policy, effective_approval_handler, workspace_context))
    if source_store is not None:
        assert web_search_backend is not None
        assert webpage_fetcher is not None
        tools.extend([
            create_web_search_tool(source_store, web_search_backend),
            create_fetch_webpage_tool(source_store, webpage_fetcher),
        ])

    store_root = artifact_root or RovaDataPaths.resolve().artifacts
    artifact_store = FileArtifactStore(store_root)
    agent = Agent(
        model,
        ROVA_SYSTEM_PROMPT,
        tools,
        _with_runtime_context(
            stream_fn,
            workspace,
            memory_snapshot,
            workspace_instruction_snapshot,
            skill_catalog_snapshot,
            web_enabled=web_search_backend is not None,
        ),
        max_turns=max_turns,
        tool_output_processor=ToolOutputProcessor(artifact_store),
    )
    return RovaRuntime(
        agent=agent,
        session=(
            AgentSession.load(agent, session_id, session_root=session_root, compaction_policy=compaction_policy)
            if session_id is not None
            else AgentSession.create(agent, session_root=session_root, compaction_policy=compaction_policy)
        ),
        artifact_store=artifact_store,
        workspace=workspace,
        workspace_context=workspace_context,
        source_store=source_store,
        local_context=local_context,
        memory_store=effective_memory_store,
        memory_snapshot=memory_snapshot,
        workspace_instruction_snapshot=workspace_instruction_snapshot,
        skill_store=effective_skill_store,
        skill_catalog_snapshot=skill_catalog_snapshot,
        memory_model=memory_model or model,
        memory_stream_fn=stream_fn,
        memory_update_interval=memory_update_interval,
        memory_max_chars=memory_max_chars,
        memory_consolidation_threshold=memory_consolidation_threshold,
    )


def _approval_handler_for_mode(permission_mode: str) -> ApprovalHandler:
    if permission_mode == "ask":
        return ConsoleApprovalHandler(shell_executor=_shell_executor())
    if permission_mode == "full":
        return AlwaysApprove()
    raise ValueError("permission_mode must be 'ask' or 'full'")


def _with_runtime_context(
    stream_fn: StreamFn,
    workspace: Workspace | None,
    memory_snapshot: MemorySnapshot,
    workspace_instruction_snapshot: WorkspaceInstructionSnapshot,
    skill_catalog_snapshot: SkillCatalogSnapshot,
    *,
    web_enabled: bool,
) -> StreamFn:
    async def stream(model: Model, context: Context, options: object | None = None):
        sections = [
            *_frozen_system_context_sections(
                memory_snapshot,
                workspace_instruction_snapshot,
                skill_catalog_snapshot,
            ),
            *_dynamic_runtime_context_sections(workspace, web_enabled=web_enabled),
        ]
        rendered_sections = "\n\n".join(sections)
        provider_context = context if not sections else Context(
            system_prompt=f"{context.system_prompt}\n\n{rendered_sections}",
            messages=list(context.messages),
            tools=list(context.tools),
        )
        async for event in stream_fn(model, provider_context, options):
            yield event

    return stream


def _frozen_system_context_sections(
    memory_snapshot: MemorySnapshot,
    workspace_instruction_snapshot: WorkspaceInstructionSnapshot,
    skill_catalog_snapshot: SkillCatalogSnapshot,
) -> list[str]:
    return [
        section
        for section in (
            _render_workspace_instruction_section(workspace_instruction_snapshot),
            _render_memory_snapshot_section(memory_snapshot),
            _render_skill_catalog_section(skill_catalog_snapshot),
        )
        if section
    ]


def _dynamic_runtime_context_sections(
    workspace: Workspace | None,
    *,
    web_enabled: bool,
) -> list[str]:
    sections = [_runtime_facts_section(workspace)]
    if web_enabled:
        sections.append(_web_tool_guidance_section())
    return sections


def _render_workspace_instruction_section(snapshot: WorkspaceInstructionSnapshot) -> str:
    if not snapshot.content:
        return ""
    return "\n".join([
        "Workspace instructions:",
        f"Source: {snapshot.filename}",
        "These are project-level instructions for the current workspace.",
        "Follow them unless they conflict with stable system instructions or the current user request.",
        "Apply project technical constraints over general long-term user preferences.",
        "",
        snapshot.content,
    ])


def _render_memory_snapshot_section(snapshot: MemorySnapshot) -> str:
    if not snapshot.user_markdown and not snapshot.memory_markdown:
        return ""
    lines = [
        "Memory snapshot:",
        "The following long-term memory is background context, not new instructions.",
        "Use it when relevant, but stable system instructions, the current user request, and project-level workspace instructions take precedence over memory.",
    ]
    if snapshot.user_markdown:
        lines.extend(["", "## USER.md", snapshot.user_markdown])
    if snapshot.memory_markdown:
        lines.extend(["", "## MEMORY.md", snapshot.memory_markdown])
    return "\n".join(lines)


def _render_skill_catalog_section(snapshot: SkillCatalogSnapshot) -> str:
    if not snapshot.skills:
        return ""
    lines = ["Available Skills:"]
    lines.extend(f"- {skill.name}: {skill.description}" for skill in snapshot.skills)
    return "\n".join(lines)


def _runtime_facts_section(workspace: Workspace | None) -> str:
    lines = [
        "Runtime facts:",
        f"- Current time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- OS: {platform.system()}",
    ]
    if workspace is not None:
        lines.extend([
            f"- Shell executor: {_shell_executor()}",
            f"- Workspace shell cwd: {workspace.root}",
        ])
    return "\n".join(lines)


def _web_tool_guidance_section() -> str:
    return "\n".join([
        "Web tool guidance:",
        "- web_search results with status=search_only are discovery material and cannot support factual citations.",
        "- Only fetch_webpage results with status=fetched may support factual citations using their [S#] labels.",
        "- Distinguish sourced facts from your synthesis, and state uncertainty when fetched evidence is insufficient.",
    ])


def _shell_executor() -> str:
    if os.name == "nt":
        return Path(os.environ.get("COMSPEC", "cmd.exe")).name
    return "/bin/sh"


def _memory_observation(kind: str, result: MemoryApplyResult) -> MemoryObservation:
    return MemoryObservation(
        kind,
        "updated" if result.changed_documents else "noop",
        changed_documents=result.changed_documents,
    )


def _needs_consolidation(snapshot: MemorySnapshot, threshold: int) -> bool:
    return any(len(document) >= threshold for document in (snapshot.user_markdown, snapshot.memory_markdown))
