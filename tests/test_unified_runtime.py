from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage
from rova.ai.models import Model
from rova.app.workspace.approval import AlwaysApprove, ConsoleApprovalHandler
from rova.app.workspace.controlled_tool import ControlledTool
from rova.app.workspace.policy import ToolExecutionRequest, ToolPolicyDecision, ToolPolicyResult
from rova.app.context.local import LocalContextItem, LocalResearchContext
from rova.app.web.sources import FetchedPage, SearchHit
from rova.app.runtime import ROVA_SYSTEM_PROMPT, build_rova_runtime
from rova.app.skills import FileSkillStore


class FakeSearch:
    async def search(self, query: str, max_results: int):
        return [SearchHit("Example", "https://example.test/", "snippet")]


class FakeFetcher:
    async def fetch(self, url: str):
        return FetchedPage("Example", "content")


@pytest.fixture(autouse=True)
def _isolate_default_rova_data_dir(monkeypatch, tmp_path: Path) -> None:
    """Default-on review state must not leak from one runtime test into another."""
    monkeypatch.setenv("ROVA_DATA_DIR", str(tmp_path / "rova-data"))


def test_stable_system_prompt_contains_only_identity_and_general_behavior() -> None:
    assert "local single-user general agent" in ROVA_SYSTEM_PROMPT
    assert "Treat tool results as observations from actual execution" in ROVA_SYSTEM_PROMPT
    assert "do not invent tools, files, sources, execution results, or verification outcomes" in ROVA_SYSTEM_PROMPT
    assert "Workspace" not in ROVA_SYSTEM_PROMPT
    assert "status=fetched" not in ROVA_SYSTEM_PROMPT
    assert "[S#]" not in ROVA_SYSTEM_PROMPT
    assert "[L#]" not in ROVA_SYSTEM_PROMPT
    assert "Memory snapshot:" not in ROVA_SYSTEM_PROMPT
    assert "Workspace instructions:" not in ROVA_SYSTEM_PROMPT
    assert "Available Skills:" not in ROVA_SYSTEM_PROMPT
    assert "Runtime facts:" not in ROVA_SYSTEM_PROMPT
    assert "Current time:" not in ROVA_SYSTEM_PROMPT


def test_unified_runtime_without_optional_inputs_creates_one_tool_free_agent(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        permission_mode="full",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert runtime.agent.system_prompt == ROVA_SYSTEM_PROMPT
    assert [tool.name for tool in runtime.agent.registry.schemas] == ["skill_view", "skill_manage", "memory_manage"]
    assert runtime.workspace is None
    assert runtime.workspace_context is None
    assert runtime.source_store is None
    assert runtime.session.session_id is not None


def test_unified_runtime_uses_explicit_product_max_turns(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        max_turns=16,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert runtime.agent.max_turns == 16


def test_unified_runtime_selects_approval_handler_from_permission_mode(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    ask_runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        permission_mode="ask",
        session_root=tmp_path / "ask-sessions",
        artifact_root=tmp_path / "ask-artifacts",
    )
    full_runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        permission_mode="full",
        session_root=tmp_path / "full-sessions",
        artifact_root=tmp_path / "full-artifacts",
    )

    ask_tools = [ask_runtime.agent.registry._tools[name] for name in ("write", "edit", "shell")]
    full_tools = [full_runtime.agent.registry._tools[name] for name in ("write", "edit", "shell")]
    assert all(isinstance(tool, ControlledTool) and isinstance(tool.approval_handler, ConsoleApprovalHandler) for tool in ask_tools)
    assert all(isinstance(tool, ControlledTool) and isinstance(tool.approval_handler, AlwaysApprove) for tool in full_tools)


def test_unified_runtime_rejects_unknown_permission_mode(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    with pytest.raises(ValueError, match="permission_mode"):
        build_rova_runtime(
            model=Model(provider="mock"),
            stream_fn=stream,
            permission_mode="unknown",
            session_root=tmp_path / "sessions",
            artifact_root=tmp_path / "artifacts",
        )


@pytest.mark.asyncio
async def test_unified_runtime_full_mode_auto_approves_but_keeps_policy_metadata(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target = workspace_root / "target.txt"
    target.write_text("old", encoding="utf-8")
    marker = workspace_root / "marker.txt"
    command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'marker.txt\').write_text(\'ran\')"'

    async def stream(_model, context, _options):
        results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
        if not results:
            yield StreamDone(AssistantMessage([
                ToolCall("edit-call", "edit", {"path": "target.txt", "old_text": "old", "new_text": "new"}),
            ], stop_reason="tool_calls"))
        elif len(results) == 1:
            assert results[0].is_error is False
            yield StreamDone(AssistantMessage([
                ToolCall("shell-call", "shell", {"command": command}),
            ], stop_reason="tool_calls"))
        else:
            assert results[1].is_error is False
            yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        permission_mode="full",
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    response = (await runtime.session.prompt("update and verify"))[-1]
    tool_results = [message for message in runtime.agent.messages if isinstance(message, ToolResultMessage)]

    assert response.text == "done"
    assert target.read_text(encoding="utf-8") == "new"
    assert marker.read_text(encoding="utf-8") == "ran"
    assert [result.metadata["policy_decision"] for result in tool_results] == ["require_approval", "require_approval"]
    assert [result.metadata["approval_decision"] for result in tool_results] == ["approve", "approve"]


@pytest.mark.asyncio
async def test_unified_runtime_full_mode_does_not_bypass_policy_deny(tmp_path: Path):
    class DenyPolicy:
        def evaluate(self, request: ToolExecutionRequest) -> ToolPolicyResult:
            return ToolPolicyResult(ToolPolicyDecision.DENY, "denied by test")

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        permission_mode="full",
        policy=DenyPolicy(),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    result = await runtime.agent.registry.execute(ToolCall("write-call", "write", {"path": "blocked.txt", "content": "no"}))

    assert result.is_error is True
    assert result.metadata["policy_decision"] == "deny"
    assert not (workspace_root / "blocked.txt").exists()


@pytest.mark.parametrize("max_turns", [0, -1, 33])
def test_unified_runtime_rejects_max_turns_outside_product_limit(tmp_path: Path, max_turns: int):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    with pytest.raises(ValueError, match="max_turns"):
        build_rova_runtime(
            model=Model(provider="mock"),
            stream_fn=stream,
            max_turns=max_turns,
            session_root=tmp_path / "sessions",
            artifact_root=tmp_path / "artifacts",
        )


@pytest.mark.asyncio
async def test_unified_runtime_adds_workspace_tools_and_workspace_context(tmp_path: Path):
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        approval_handler=AlwaysApprove(),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.session.prompt("Inspect the workspace.")

    assert [tool.name for tool in runtime.agent.registry.schemas] == [
        "skill_view", "skill_manage", "memory_manage", "read", "list_dir", "search", "write", "edit", "shell",
    ]
    assert runtime.workspace is not None
    assert runtime.workspace_context is not None
    assert "Runtime facts:" in received_system_prompts[0]
    assert str(workspace_root.resolve()) in received_system_prompts[0]


@pytest.mark.asyncio
async def test_unified_runtime_injects_environment_facts_for_workspace_shell(tmp_path: Path):
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        approval_handler=AlwaysApprove(),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.session.prompt("Inspect the environment.")

    provider_context = received_system_prompts[0]
    assert "Runtime facts:" in provider_context
    assert "OS:" in provider_context
    assert "Shell executor:" in provider_context
    assert "Python version:" not in provider_context
    assert "Runtime Python:" not in provider_context
    assert f"Workspace shell cwd: {workspace_root.resolve()}" in provider_context


@pytest.mark.asyncio
async def test_unified_runtime_marks_shell_unavailable_without_workspace(tmp_path: Path):
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.session.prompt("Inspect the environment.")

    provider_context = received_system_prompts[0]
    assert "Runtime facts:" in provider_context
    assert "Shell execution tool is unavailable." not in provider_context
    assert "Shell executor:" not in provider_context
    assert "Workspace shell cwd:" not in provider_context


@pytest.mark.asyncio
async def test_unified_runtime_environment_section_omits_secret_and_path_values(monkeypatch, tmp_path: Path):
    received_system_prompts: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("PATH", "test-path")

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.session.prompt("Inspect the environment.")

    provider_context = received_system_prompts[0]
    assert "Runtime facts:" in provider_context
    assert "test-secret" not in provider_context
    assert "test-path" not in provider_context


@pytest.mark.asyncio
async def test_unified_runtime_does_not_persist_environment_section_to_session(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    session_root = tmp_path / "sessions"
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        permission_mode="full",
        session_root=session_root,
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.session.prompt("Inspect the environment.")

    serialized = "\n".join(path.read_text(encoding="utf-8") for path in session_root.rglob("*.jsonl"))
    assert "Runtime facts:" not in serialized
    assert "Permission mode: full" not in serialized


@pytest.mark.asyncio
async def test_unified_runtime_adds_web_tools_and_keeps_source_store_run_local(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        web_search_backend=FakeSearch(),
        webpage_fetcher=FakeFetcher(),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    result = await runtime.agent.registry.execute(ToolCall("call-1", "web_search", {"query": "Rova"}))

    assert [tool.name for tool in runtime.agent.registry.schemas] == [
        "skill_view", "skill_manage", "memory_manage", "web_search", "fetch_webpage",
    ]
    assert result.is_error is False
    assert runtime.source_store is not None
    assert [source.source_id for source in runtime.source_store.all()] == ["S1"]


def test_unified_runtime_combines_workspace_and_web_tools_in_one_agent(tmp_path: Path):
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        approval_handler=AlwaysApprove(),
        web_search_backend=FakeSearch(),
        webpage_fetcher=FakeFetcher(),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert [tool.name for tool in runtime.agent.registry.schemas] == [
        "skill_view", "skill_manage", "memory_manage", "read", "list_dir", "search", "write", "edit", "shell",
        "web_search", "fetch_webpage",
    ]
    assert runtime.workspace is not None
    assert runtime.source_store is not None
    assert runtime.agent is runtime.session.agent


@pytest.mark.asyncio
async def test_unified_runtime_attaches_local_research_context_to_the_current_user_message(tmp_path: Path):
    received_contexts = []

    async def stream(_model, context, _options):
        received_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    local_context = LocalResearchContext((
        LocalContextItem("L1", "notes.md", 12, "a" * 64, "Local notes."),
    ))
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        local_context=local_context,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("Summarize my local notes.")

    assert "[L1] notes.md" not in received_contexts[0].system_prompt
    assert received_contexts[0].messages[-1].content.startswith("Summarize my local notes.")
    assert "[L1] notes.md" in received_contexts[0].messages[-1].content
    assert "Local notes." in received_contexts[0].messages[-1].content
    assert [tool.name for tool in runtime.agent.registry.schemas] == ["skill_view", "skill_manage", "memory_manage"]
    assert runtime.source_store is None


@pytest.mark.asyncio
async def test_unified_runtime_does_not_repeat_local_attachment_in_later_user_messages(tmp_path: Path):
    received_contexts = []

    async def stream(_model, context, _options):
        received_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    local_context = LocalResearchContext((
        LocalContextItem("L1", "notes.md", 12, "a" * 64, "Local notes."),
    ))
    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, local_context=local_context,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("first")
    await runtime.prompt("second")

    assert "[L1] notes.md" not in received_contexts[1].system_prompt
    assert received_contexts[1].messages[-1].content == "second"
    assert "[L1] notes.md" in received_contexts[1].messages[0].content


@pytest.mark.asyncio
async def test_unified_runtime_restores_local_attachment_from_session_history(tmp_path: Path):
    received_contexts = []

    async def stream(_model, context, _options):
        received_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    session_root = tmp_path / "sessions"
    local_context = LocalResearchContext((
        LocalContextItem("L1", "notes.md", 12, "a" * 64, "Local notes."),
    ))
    original = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, local_context=local_context,
        session_root=session_root, artifact_root=tmp_path / "artifacts",
    )
    session_id = original.session.session_id
    assert session_id is not None

    await original.prompt("first")
    original.session.close()

    restored = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        session_id=session_id, session_root=session_root, artifact_root=tmp_path / "artifacts",
    )
    await restored.prompt("second")

    assert received_contexts[-1].messages[-1].content == "second"
    assert "[L1] notes.md" in received_contexts[-1].messages[0].content
    assert "Local notes." in received_contexts[-1].messages[0].content


def test_unified_runtime_keeps_dynamic_context_out_of_stable_agent_prompt(tmp_path: Path):
    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=lambda *_: None,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    assert "Runtime facts:" not in runtime.agent.system_prompt
    assert "Memory snapshot:" not in runtime.agent.system_prompt


@pytest.mark.asyncio
async def test_unified_runtime_injects_minimal_runtime_facts_without_python_path(tmp_path: Path):
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("hello")

    runtime_context = received_system_prompts[0]
    current_time = next(line for line in runtime_context.splitlines() if line.startswith("- Current time: "))
    assert "Runtime facts:" in runtime_context
    assert datetime.fromisoformat(current_time.removeprefix("- Current time: ")).tzinfo is not None
    assert "- OS:" in runtime_context
    assert "- Python version:" not in runtime_context
    assert "Runtime Python:" not in runtime_context
    assert "PATH=" not in runtime_context
    assert "OPENAI_API_KEY=" not in runtime_context


@pytest.mark.asyncio
async def test_unified_runtime_keeps_source_provenance_out_of_system_context(tmp_path: Path):
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        web_search_backend=FakeSearch(), webpage_fetcher=FakeFetcher(),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )
    assert runtime.source_store is not None
    source = runtime.source_store.register(SearchHit("Example source", "https://example.test/source", "snippet"))
    runtime.source_store.set_content(source.source_id, "private fetched page body")

    await runtime.prompt("summarize")

    provider_context = received_system_prompts[0]
    assert "External source context:" not in provider_context
    assert "[S1] fetched: Example source" not in provider_context
    assert "URL: https://example.test/source" not in provider_context
    assert "private fetched page body" not in provider_context


@pytest.mark.asyncio
async def test_unified_runtime_adds_web_evidence_rules_only_when_web_tools_are_enabled(tmp_path: Path):
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    plain_runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        session_root=tmp_path / "plain-sessions", artifact_root=tmp_path / "plain-artifacts",
    )
    web_runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        web_search_backend=FakeSearch(), webpage_fetcher=FakeFetcher(),
        session_root=tmp_path / "web-sessions", artifact_root=tmp_path / "web-artifacts",
    )

    await plain_runtime.prompt("hello")
    await web_runtime.prompt("hello")

    assert "Web tool guidance:" not in received_system_prompts[0]
    assert "Web tool guidance:" in received_system_prompts[1]
    assert "status=search_only" in received_system_prompts[1]
    assert "status=fetched" in received_system_prompts[1]


@pytest.mark.asyncio
async def test_unified_runtime_freezes_skill_catalog_and_only_injects_metadata(tmp_path: Path):
    skill_root = tmp_path / "skills"
    store = FileSkillStore(skill_root)
    store.create("code-review", "---\nname: code-review\ndescription: Review changes.\n---\n\n# Code Review\n\nFull procedure.")
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, skill_root=skill_root,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )
    created = await runtime.agent.registry.execute(ToolCall("create-skill", "skill_manage", {
        "action": "create",
        "name": "paper-review",
        "content": "---\nname: paper-review\ndescription: Review papers.\n---\n\n# Paper Review\n\nFull paper procedure.",
    }))
    assert created.is_error is False

    await runtime.prompt("first")
    await runtime.prompt("second")

    assert [tool.name for tool in runtime.agent.registry.schemas] == ["skill_view", "skill_manage", "memory_manage"]
    assert "Available Skills:" in received_system_prompts[0]
    assert "code-review: Review changes." in received_system_prompts[0]
    assert "Full procedure." not in received_system_prompts[0]
    assert "paper-review" not in received_system_prompts[1]

    session_id = runtime.session.session_id
    assert session_id is not None
    runtime.session.close()
    next_runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, skill_root=skill_root,
        session_id=session_id, session_root=tmp_path / "sessions", artifact_root=tmp_path / "next-artifacts",
    )
    await next_runtime.prompt("third")
    assert "paper-review: Review papers." in received_system_prompts[-1]


@pytest.mark.asyncio
async def test_unified_runtime_persists_skill_view_as_a_normal_tool_result(tmp_path: Path):
    skill_root = tmp_path / "skills"
    FileSkillStore(skill_root).create(
        "code-review", "---\nname: code-review\ndescription: Review changes.\n---\n\n# Code Review\n\nProcedure.",
    )

    async def stream(_model, context, _options):
        results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
        if not results:
            yield StreamDone(AssistantMessage([
                ToolCall("view-1", "skill_view", {"name": "code-review"}),
            ], stop_reason="tool_calls"))
            return
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    session_root = tmp_path / "sessions"
    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, skill_root=skill_root,
        session_root=session_root, artifact_root=tmp_path / "artifacts",
    )

    assert (await runtime.prompt("review this"))[-1].text == "done"
    skill_result = next(message for message in runtime.agent.messages if isinstance(message, ToolResultMessage))
    assert skill_result.tool_name == "skill_view"
    assert "# Code Review" in skill_result.text
    assert "# Code Review" in next(session_root.glob("*.jsonl")).read_text(encoding="utf-8")
