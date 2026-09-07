from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import platform
import warnings

from rova.ai.context import Context
from rova.ai.models import Model
from rova.ai.providers.openai_compatible import estimate_provider_input_tokens
from rova.agent_core.agent import Agent
from rova.agent_core.hooks import HookRegistry
from rova.agent_core.tools import ToolExecutionMode
from rova.agent_core.tool_output import ToolOutputProcessor
from rova.agent_core.types import StreamFn
from rova.agent_session.agent_session import AgentSession
from rova.agent_session.execution_journal import ExecutionEnvironmentIdentity, ExecutionEnvironmentStatus
from rova.agent_session.compaction import CompactionPolicy
from rova.artifacts import FileArtifactStore

from .context.local import LocalResearchContext
from .extensions import ExtensionAPI, ExtensionLoadReport, ExtensionLoader
from .web.sources import ResearchSourceStore
from .web.tools import WebFetchBackend, WebSearchBackend, create_fetch_webpage_tool, create_web_search_tool
from .workspace.approval import AlwaysApprove, ApprovalHandler, ConsoleApprovalHandler
from .workspace.context import WorkspaceContext
from .workspace.controlled_tool import RovaToolGovernance, build_coding_tools
from .workspace.environment import DockerSandboxEnvironment, ExecutionEnvironment, LocalExecutionEnvironment
from .workspace.instructions import WorkspaceInstructionSnapshot, load_workspace_instruction
from .workspace.policy import DefaultRovaToolPolicy, ToolPolicy
from .workspace.sandbox import SandboxError, SandboxState, SandboxStore, workspace_identity
from .workspace.sandbox_control import SandboxControl
from .workspace.terminal import DockerTerminalBackend, LocalTerminalBackend, TerminalBackend
from .workspace.workspace import Workspace
from .memory import (
    FileMemoryStore,
    MemorySnapshot,
    MemoryStore,
    MemoryStoreError,
    create_memory_tools,
)
from .experience_review import ExperienceReviewService, ExperienceReviewer, FileExperienceReviewStore
from .paths import RovaDataPaths
from .skills import FileSkillStore, SkillCatalogSnapshot, SkillStoreError, create_skill_tools
from .vision import VisionClient, create_vision_analyze_tool
from rova.mcp.client import MCPClient, MCPServerConfig
from rova.mcp.config import MCPServerSettings, load_mcp_settings, safe_stdio_environment
from rova.mcp.manager import MCPManager
from rova.trace import JsonlTraceStore, TraceRecorder, TraceStore, TraceStoreError


DEFAULT_MAX_TURNS = 16
MAX_PRODUCT_TURNS = 32
DEFAULT_MEMORY_MAX_CHARS = 6_000
DEFAULT_EXPERIENCE_REVIEW_TOOL_THRESHOLD = 10
DEFAULT_EXPERIENCE_REVIEW_TASK_THRESHOLD = 5


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
    execution_environment: ExecutionEnvironment | None
    terminal_backend: TerminalBackend | None
    workspace_context: WorkspaceContext | None
    source_store: ResearchSourceStore | None
    local_context: LocalResearchContext | None
    memory_store: MemoryStore
    memory_snapshot: MemorySnapshot
    workspace_instruction_snapshot: WorkspaceInstructionSnapshot
    skill_store: FileSkillStore
    skill_catalog_snapshot: SkillCatalogSnapshot
    memory_max_chars: int
    experience_review_service: ExperienceReviewService | None
    extension_api: ExtensionAPI
    extension_load_report: ExtensionLoadReport
    mcp_manager: MCPManager | None
    trace_store: TraceStore
    sandbox_control: SandboxControl | None = None
    sandbox_unavailable_state: SandboxState | None = None

    def __post_init__(self) -> None:
        self._memory_listeners: list[Callable[[MemoryObservation], None]] = []
        self._local_context_attached = False
        self._closed = False
        self._trace_issues: list[str] = []

    def subscribe_memory(self, listener: Callable[["MemoryObservation"], None]) -> Callable[[], None]:
        self._memory_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._memory_listeners:
                self._memory_listeners.remove(listener)

        return unsubscribe

    @property
    def extension_runtime_issues(self):
        return self.extension_api.runtime_issues

    @property
    def mcp_runtime_issues(self):
        return () if self.mcp_manager is None else tuple(self.mcp_manager.issues)

    @property
    def trace_runtime_issues(self) -> tuple[str, ...]:
        return tuple(self._trace_issues)

    @property
    def recovery_report(self):
        """Return the selected Session's load/branch reconciliation facts."""
        return self.session.recovery_report

    def start_mcp_discovery(self) -> None:
        if self.mcp_manager is not None:
            self.mcp_manager.start()

    async def close(self) -> None:
        """Release Runtime-owned resources without persisting backend state."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.mcp_manager is not None:
                await self.mcp_manager.close()
        finally:
            try:
                if self.execution_environment is not None:
                    await self.execution_environment.close()
                elif self.terminal_backend is not None:
                    await self.terminal_backend.close()
            finally:
                self.session.close()

    async def prompt(self, text: str):
        if self.sandbox_unavailable_state is not None:
            raise SandboxError(
                f"Sandbox is unavailable in state {self.sandbox_unavailable_state.value}; "
                "coding is blocked until the user explicitly creates a new Sandbox"
            )
        if self.sandbox_control is not None:
            state = self.sandbox_control.metadata().state
            if state is not SandboxState.READY:
                raise SandboxError(
                    f"Sandbox is in state {state.value}; coding is blocked until the user explicitly creates a new Sandbox"
                )
        self.start_mcp_discovery()
        if self.experience_review_service is not None:
            self.experience_review_service.begin_run(text)
        request_text = text
        if not self._local_context_attached and self.local_context is not None:
            attachment = self.local_context.render_user_attachment()
            if attachment:
                request_text = f"{text}\n\n{attachment}"
                self._local_context_attached = True
        try:
            recorder = TraceRecorder()
            responses, trace = await recorder.capture_run(
                self.agent,
                lambda: self.session.prompt(request_text),
                session_id=self.session.session_id,
            )
        except BaseException:
            if self.experience_review_service is not None:
                self.experience_review_service.discard_run()
            raise
        trace.input_entry_id = self.session.last_prompt_input_entry_id
        trace.input_message = request_text
        try:
            self.trace_store.append(trace)
        except TraceStoreError as error:
            self._trace_issues.append(str(error))
        if self.experience_review_service is not None:
            final = responses[-1] if responses else None
            if final is None or final.stop_reason != "stop" or final.tool_calls:
                self.experience_review_service.discard_run()
            else:
                try:
                    state = self.experience_review_service.commit_completed_task(
                        session_id=self.session.session_id,
                        final_response=final.text,
                    )
                    await self.experience_review_service.review_if_due(state)
                except Exception:
                    # Experience maintenance must not replace an already completed user response.
                    pass
        return responses

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
    trace_root: Path | None = None,
    compaction_policy: CompactionPolicy | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    session_id: str | None = None,
    memory_store: MemoryStore | None = None,
    memory_root: Path | None = None,
    skill_root: Path | None = None,
    memory_model: Model | None = None,
    memory_max_chars: int = DEFAULT_MEMORY_MAX_CHARS,
    experience_review_enabled: bool = True,
    experience_review_tool_threshold: int = DEFAULT_EXPERIENCE_REVIEW_TOOL_THRESHOLD,
    experience_review_task_threshold: int = DEFAULT_EXPERIENCE_REVIEW_TASK_THRESHOLD,
    experience_root: Path | None = None,
    vision_client: VisionClient | None = None,
    extension_roots: Sequence[Path] | None = None,
    terminal_backend: str = "local",
    docker_image: str | None = None,
    isolated_sandbox: bool = False,
    sandbox_root: Path | None = None,
    mcp_config_path: Path | None = None,
    tool_execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL,
) -> RovaRuntime:
    if (web_search_backend is None) != (webpage_fetcher is None):
        raise ValueError("web_search_backend and webpage_fetcher must be provided together")
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or not 1 <= max_turns <= MAX_PRODUCT_TURNS:
        raise ValueError(f"max_turns must be an integer between 1 and {MAX_PRODUCT_TURNS}")
    if permission_mode not in {"ask", "full"}:
        raise ValueError("permission_mode must be 'ask' or 'full'")
    if not isinstance(memory_max_chars, int) or isinstance(memory_max_chars, bool) or memory_max_chars <= 0:
        raise ValueError("memory_max_chars must be a positive integer")
    if not isinstance(experience_review_enabled, bool):
        raise ValueError("experience_review_enabled must be a boolean")
    for name, value in (
        ("experience_review_tool_threshold", experience_review_tool_threshold),
        ("experience_review_task_threshold", experience_review_task_threshold),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if vision_client is not None and workspace_root is None:
        raise ValueError("vision_client requires workspace_root")
    if terminal_backend not in {"local", "docker"}:
        raise ValueError("terminal_backend must be 'local' or 'docker'")
    if terminal_backend == "docker" and workspace_root is None:
        raise ValueError("terminal_backend='docker' requires workspace_root")
    if terminal_backend == "docker" and not docker_image:
        raise ValueError("docker_image is required when terminal_backend='docker'")
    if terminal_backend == "local" and docker_image is not None:
        raise ValueError("docker_image requires terminal_backend='docker'")
    if not isinstance(isolated_sandbox, bool):
        raise ValueError("isolated_sandbox must be a boolean")
    if isolated_sandbox and terminal_backend != "docker":
        raise ValueError("isolated_sandbox requires terminal_backend='docker'")

    workspace = Workspace(workspace_root) if workspace_root is not None else None
    effective_memory_store = memory_store or FileMemoryStore(memory_root)
    try:
        memory_snapshot = effective_memory_store.load_snapshot()
    except MemoryStoreError:
        memory_snapshot = MemorySnapshot()
    workspace_instruction_snapshot = load_workspace_instruction(workspace.root if workspace is not None else None)
    effective_skill_store = FileSkillStore(skill_root)
    mcp_servers = load_mcp_settings(mcp_config_path).servers if mcp_config_path is not None else ()
    try:
        skill_catalog_snapshot = effective_skill_store.discover_catalog()
    except SkillStoreError as error:
        warnings.warn(f"Skill catalog unavailable: {error}", RuntimeWarning, stacklevel=2)
        skill_catalog_snapshot = SkillCatalogSnapshot()
    execution_environment: ExecutionEnvironment | None = None
    pending_sandbox_store: SandboxStore | None = None
    pending_sandbox_id: str | None = None
    active_sandbox_store: SandboxStore | None = None
    active_sandbox_id: str | None = None
    unavailable_sandbox: SandboxState | None = None
    sandbox_resumed = False
    if workspace is not None:
        if isolated_sandbox:
            sandbox_store = SandboxStore(sandbox_root or RovaDataPaths.resolve().sandboxes)
            existing = (
                sandbox_store.load_for_session(session_id, workspace_identity(workspace.root))
                if session_id is not None
                else None
            )
            if existing is None:
                imported = sandbox_store.import_baseline(sandbox_store.create_unbound(workspace).sandbox_id)
                pending_sandbox_store = sandbox_store
                pending_sandbox_id = imported.sandbox_id
            else:
                imported = existing
                sandbox_resumed = True
            active_sandbox_store = sandbox_store
            active_sandbox_id = imported.sandbox_id
            if imported.state is SandboxState.READY or pending_sandbox_store is not None:
                execution_environment = DockerSandboxEnvironment(
                    host_workspace=workspace,
                    sandbox_workspace=Workspace(imported.sandbox_root),
                    image=docker_image or "",
                    skill_root=effective_skill_store.root,
                    resumed=sandbox_resumed,
                )
            else:
                unavailable_sandbox = imported.state
        else:
            effective_terminal_backend = (
                LocalTerminalBackend(workspace)
                if terminal_backend == "local"
                else DockerTerminalBackend(workspace, image=docker_image or "", skill_root=effective_skill_store.root)
            )
            execution_environment = LocalExecutionEnvironment(workspace, terminal=effective_terminal_backend)
    effective_terminal_backend = execution_environment.terminal if execution_environment is not None else None
    agent_workspace: Workspace | None = None
    if workspace is not None and execution_environment is not None:
        if execution_environment.descriptor.kind == "docker_sandbox":
            # Docker Sandbox paths remain a Runtime detail; use its filesystem resolver only internally.
            agent_workspace = Workspace(execution_environment.filesystem.resolve("."))
        else:
            agent_workspace = workspace
    workspace_context = WorkspaceContext(agent_workspace) if agent_workspace is not None else None
    source_store = ResearchSourceStore() if web_search_backend is not None else None
    tools = [
        *create_skill_tools(
            effective_skill_store,
            skill_directory_renderer=(
                effective_terminal_backend.render_skill_directory
                if effective_terminal_backend is not None
                else None
            ),
        ),
        *create_memory_tools(effective_memory_store, max_chars=memory_max_chars),
    ]
    effective_policy: ToolPolicy = DefaultRovaToolPolicy(policy)
    effective_approval_handler: ApprovalHandler | None = approval_handler
    if permission_mode == "full" and effective_approval_handler is None:
        effective_approval_handler = AlwaysApprove()
    if execution_environment is not None:
        assert effective_terminal_backend is not None
        effective_approval_handler = effective_approval_handler or _approval_handler_for_mode(
            permission_mode, effective_terminal_backend
        )
        tools.extend(build_coding_tools(execution_environment))
        if vision_client is not None:
            tools.append(create_vision_analyze_tool(execution_environment.filesystem, vision_client))
    if source_store is not None:
        assert web_search_backend is not None
        assert webpage_fetcher is not None
        tools.extend([
            create_web_search_tool(source_store, web_search_backend),
            create_fetch_webpage_tool(source_store, webpage_fetcher),
        ])
    hook_registry = HookRegistry()
    extension_api = ExtensionAPI([tool.tool.name for tool in tools], hook_registry=hook_registry)
    extension_loader = ExtensionLoader(
        ExtensionLoader.default_roots(workspace.root if workspace is not None else None)
        if extension_roots is None
        else extension_roots
    )
    extension_load_report = extension_loader.load(extension_api)
    tools.extend(extension_api.tools)
    tool_governance = RovaToolGovernance(
        effective_policy,
        effective_approval_handler,
        agent_workspace.root if agent_workspace is not None else None,
        workspace_context,
        effective_terminal_backend.environment if effective_terminal_backend is not None else None,
    )

    store_root = artifact_root or RovaDataPaths.resolve().artifacts
    artifact_store = FileArtifactStore(store_root)
    trace_store = JsonlTraceStore((trace_root or RovaDataPaths.resolve().traces) / "runs.jsonl")
    agent = Agent(
        model,
        ROVA_SYSTEM_PROMPT,
        tools,
        stream_fn,
        max_turns=max_turns,
        tool_output_processor=ToolOutputProcessor(artifact_store),
        tool_execution_mode=tool_execution_mode,
        tool_governance=tool_governance,
        hook_registry=hook_registry,
    )
    extension_api.bind_event_hooks(agent)
    mcp_manager = (
        MCPManager(
            mcp_servers,
            agent.registry,
            _create_mcp_client,
        )
        if mcp_servers
        else None
    )
    experience_review_service = None
    if experience_review_enabled:
        experience_review_service = ExperienceReviewService(
            FileExperienceReviewStore(experience_root),
            reviewer=ExperienceReviewer(
                model=memory_model or model,
                stream_fn=stream_fn,
                skill_store=effective_skill_store,
            ),
            memory_store=effective_memory_store,
            skill_store=effective_skill_store,
            memory_max_chars=memory_max_chars,
            tool_threshold=experience_review_tool_threshold,
            task_threshold=experience_review_task_threshold,
        )
        agent.subscribe(_isolated_experience_event_handler(experience_review_service))
    provider_context_estimator = lambda context: estimate_provider_input_tokens(model, context)
    branch_guard = (
        _sandbox_branch_guard(active_sandbox_store, active_sandbox_id)
        if active_sandbox_store is not None and active_sandbox_id is not None
        else None
    )
    environment_identity_resolver = (
        _sandbox_execution_identity_resolver(active_sandbox_store, active_sandbox_id)
        if active_sandbox_store is not None and active_sandbox_id is not None and unavailable_sandbox is None
        else (_local_execution_identity if not isolated_sandbox else None)
    )
    environment_status_resolver = (
        _sandbox_environment_status_resolver(active_sandbox_store)
        if active_sandbox_store is not None
        else _local_environment_status
    )
    session = (
        AgentSession.load(
            agent,
            session_id,
            session_root=session_root,
            compaction_policy=compaction_policy,
            provider_context_estimator=provider_context_estimator,
            branch_guard=branch_guard,
            execution_environment_identity_resolver=environment_identity_resolver,
            environment_status_resolver=environment_status_resolver,
        )
        if session_id is not None
        else AgentSession.create(
            agent,
            session_root=session_root,
            compaction_policy=compaction_policy,
            provider_context_estimator=provider_context_estimator,
            branch_guard=branch_guard,
            execution_environment_identity_resolver=environment_identity_resolver,
            environment_status_resolver=environment_status_resolver,
        )
    )
    if pending_sandbox_store is not None and pending_sandbox_id is not None:
        assert session.session_id is not None
        pending_sandbox_store.bind_session(pending_sandbox_id, session.session_id)
        pending_sandbox_store.mark_ready(pending_sandbox_id)
    sandbox_control = (
        SandboxControl(
            active_sandbox_store,
            workspace,
            session.session_id or "",
            active_sandbox_id,
            container_recreated_on_resume=sandbox_resumed,
        )
        if active_sandbox_store is not None and active_sandbox_id is not None and workspace is not None
        else None
    )

    async def prepare_runtime_context(base_context: Context) -> Context:
        provider_context = _assemble_runtime_context(
            base_context,
            workspace,
            execution_environment,
            memory_snapshot,
            workspace_instruction_snapshot,
            skill_catalog_snapshot,
            extension_api,
            web_enabled=web_search_backend is not None,
        )
        return await session.prepare_provider_context(
            provider_context,
            rebuild_context=lambda: _assemble_runtime_context(
                agent.create_context_snapshot(),
                workspace,
                execution_environment,
                memory_snapshot,
                workspace_instruction_snapshot,
                skill_catalog_snapshot,
                extension_api,
                web_enabled=web_search_backend is not None,
            ),
        )

    def rebuild_runtime_context() -> Context:
        return _assemble_runtime_context(
            agent.create_context_snapshot(),
            workspace,
            execution_environment,
            memory_snapshot,
            workspace_instruction_snapshot,
            skill_catalog_snapshot,
            extension_api,
            web_enabled=web_search_backend is not None,
        )

    async def recover_context_overflow(provider_context: Context) -> Context:
        return await session.prepare_provider_context(
            provider_context,
            rebuild_context=rebuild_runtime_context,
            force_compaction=True,
            trigger="overflow_recovery",
        )

    agent.set_context_preparer(prepare_runtime_context)
    agent.set_context_overflow_recovery(recover_context_overflow)

    return RovaRuntime(
        agent=agent,
        session=session,
        artifact_store=artifact_store,
        workspace=workspace,
        execution_environment=execution_environment,
        terminal_backend=effective_terminal_backend,
        workspace_context=workspace_context,
        source_store=source_store,
        local_context=local_context,
        memory_store=effective_memory_store,
        memory_snapshot=memory_snapshot,
        workspace_instruction_snapshot=workspace_instruction_snapshot,
        skill_store=effective_skill_store,
        skill_catalog_snapshot=skill_catalog_snapshot,
        memory_max_chars=memory_max_chars,
        experience_review_service=experience_review_service,
        extension_api=extension_api,
        extension_load_report=extension_load_report,
        mcp_manager=mcp_manager,
        trace_store=trace_store,
        sandbox_control=sandbox_control,
        sandbox_unavailable_state=unavailable_sandbox,
    )


def _create_mcp_client(server: MCPServerSettings) -> MCPClient:
    return MCPClient(MCPServerConfig(
        server.server_id,
        server.transport,
        url=server.url,
        headers=server.headers,
        command=server.command,
        args=server.args,
        environment=(safe_stdio_environment(os.environ, server.environment) if server.transport == "stdio" else {}),
    ))


def _approval_handler_for_mode(permission_mode: str, terminal_backend: TerminalBackend) -> ApprovalHandler:
    if permission_mode == "ask":
        return ConsoleApprovalHandler(shell_executor=terminal_backend.environment.executor)
    if permission_mode == "full":
        return AlwaysApprove()
    raise ValueError("permission_mode must be 'ask' or 'full'")


def _with_runtime_context(
    stream_fn: StreamFn,
    workspace: Workspace | None,
    execution_environment: ExecutionEnvironment | None,
    memory_snapshot: MemorySnapshot,
    workspace_instruction_snapshot: WorkspaceInstructionSnapshot,
    skill_catalog_snapshot: SkillCatalogSnapshot,
    extension_api: ExtensionAPI,
    *,
    web_enabled: bool,
) -> StreamFn:
    async def stream(model: Model, context: Context, options: object | None = None):
        provider_context = _assemble_runtime_context(
            context,
            workspace,
            execution_environment,
            memory_snapshot,
            workspace_instruction_snapshot,
            skill_catalog_snapshot,
            extension_api,
            web_enabled=web_enabled,
        )
        async for event in stream_fn(model, provider_context, options):
            yield event

    return stream


def _assemble_runtime_context(
    context: Context,
    workspace: Workspace | None,
    execution_environment: ExecutionEnvironment | None,
    memory_snapshot: MemorySnapshot,
    workspace_instruction_snapshot: WorkspaceInstructionSnapshot,
    skill_catalog_snapshot: SkillCatalogSnapshot,
    extension_api: ExtensionAPI,
    *,
    web_enabled: bool,
) -> Context:
    sections = [
        *_frozen_system_context_sections(memory_snapshot, workspace_instruction_snapshot, skill_catalog_snapshot),
        *_dynamic_runtime_context_sections(workspace, execution_environment, web_enabled=web_enabled),
        *extension_api.render_context_sections(),
    ]
    if not sections:
        return context
    return Context(
        system_prompt=f"{context.system_prompt}\n\n{'\n\n'.join(sections)}",
        messages=list(context.messages),
        tools=list(context.tools),
    )


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
    execution_environment: ExecutionEnvironment | None,
    *,
    web_enabled: bool,
) -> list[str]:
    sections = [_runtime_facts_section(workspace, execution_environment)]
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


def _runtime_facts_section(
    workspace: Workspace | None,
    execution_environment: ExecutionEnvironment | None,
) -> str:
    lines = [
        "Runtime facts:",
        f"- Current time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- OS: {platform.system()}",
    ]
    if workspace is not None and execution_environment is not None:
        terminal_backend = execution_environment.terminal
        if execution_environment.descriptor.kind == "docker_sandbox":
            lines.extend([
                "- Terminal backend: docker",
                f"- Shell executor: {terminal_backend.environment.executor}",
                "- Workspace shell cwd: /workspace",
                "- Logical workspace: /workspace (use relative paths with workspace-aware tools)",
                "- You are working in an isolated Sandbox; ordinary workspace tools cannot modify the Host Workspace.",
                "- Applying or discarding Sandbox changes is a user control-plane action, not an available tool.",
                "- This Sandbox has private Git baseline metadata only; it has no Host Git history or remotes.",
            ])
            if execution_environment.descriptor.resume_note:
                lines.append(f"- Sandbox environment: {execution_environment.descriptor.resume_note}")
        else:
            lines.extend([
                f"- Terminal backend: {terminal_backend.environment.kind}",
                f"- Shell executor: {terminal_backend.environment.executor}",
                f"- Workspace shell cwd: {terminal_backend.environment.cwd}",
            ])
            if terminal_backend.environment.kind == "docker":
                lines.append(f"- Workspace bind mount: {workspace.root} -> {terminal_backend.environment.cwd}")
    return "\n".join(lines)


def _web_tool_guidance_section() -> str:
    return "\n".join([
        "Web tool guidance:",
        "- web_search results with status=search_only are discovery material and cannot support factual citations.",
        "- Only fetch_webpage results with status=fetched may support factual citations using their [S#] labels.",
        "- Distinguish sourced facts from your synthesis, and state uncertainty when fetched evidence is insufficient.",
    ])


def _sandbox_branch_guard(store: SandboxStore, sandbox_id: str) -> Callable[[], str | None]:
    def guard() -> str | None:
        metadata = store._load_metadata(sandbox_id)
        if metadata.state is SandboxState.READY:
            return "Current conversation has an active mutable Sandbox. Apply or Discard it before switching branches."
        return None

    return guard


def _local_execution_identity() -> ExecutionEnvironmentIdentity:
    return ExecutionEnvironmentIdentity("local")


def _local_environment_status(identity: ExecutionEnvironmentIdentity) -> ExecutionEnvironmentStatus:
    if identity.kind != "local":
        return ExecutionEnvironmentStatus(identity, False, "unknown", environment_lost=True)
    return ExecutionEnvironmentStatus(identity, True, "local")


def _sandbox_execution_identity_resolver(
    store: SandboxStore,
    sandbox_id: str,
) -> Callable[[], ExecutionEnvironmentIdentity]:
    def resolve() -> ExecutionEnvironmentIdentity:
        metadata = store.load_for_execution(sandbox_id)
        if metadata.state is not SandboxState.READY:
            raise SandboxError(f"Sandbox is unavailable in state {metadata.state.value}; coding execution is blocked")
        return ExecutionEnvironmentIdentity("docker_sandbox", sandbox_id)

    return resolve


def _sandbox_environment_status_resolver(
    store: SandboxStore,
) -> Callable[[ExecutionEnvironmentIdentity], ExecutionEnvironmentStatus]:
    def resolve(identity: ExecutionEnvironmentIdentity) -> ExecutionEnvironmentStatus:
        if identity.kind != "docker_sandbox" or identity.sandbox_id is None:
            return ExecutionEnvironmentStatus(identity, False, "unknown", environment_lost=True)
        try:
            metadata = store.load_for_execution(identity.sandbox_id)
        except SandboxError:
            return ExecutionEnvironmentStatus(identity, False, "abandoned", environment_lost=True)
        available = metadata.state is SandboxState.READY
        return ExecutionEnvironmentStatus(
            identity,
            available,
            metadata.state.value,
            environment_lost=metadata.state is SandboxState.ABANDONED,
            lifecycle_contradiction=metadata.state in {SandboxState.APPLIED, SandboxState.DISCARDED},
        )

    return resolve


def _isolated_experience_event_handler(service: ExperienceReviewService):
    def handle(event) -> None:
        try:
            service.on_agent_event(event)
        except Exception:
            # This subscriber is product maintenance; it must not change the
            # exception semantics of other Agent subscribers.
            return

    return handle
