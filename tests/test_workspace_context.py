from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage
from rova.ai.models import Model
from rova.agent_core.tools import ToolRegistry
from rova.app import workspace as workspace_module
from rova.app.runtime import build_rova_runtime
from rova.app.workspace import AlwaysApprove, AlwaysDeny, DefaultCodingToolPolicy, Workspace, build_controlled_coding_tools


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "read.py").write_text("read me\n", encoding="utf-8")
    (root / "src" / "existing.py").write_text("old value\n", encoding="utf-8")
    return root


def test_workspace_context_records_explicit_file_operations_by_final_state(workspace_root: Path):
    workspace = Workspace(workspace_root)
    context = workspace_module.WorkspaceContext(workspace)

    context.record_read(workspace.resolve("src/read.py"))
    context.record_write(workspace.resolve("src/created.py"), existed_before=False)
    context.record_write(workspace.resolve("src/created.py"), existed_before=True)
    context.record_write(workspace.resolve("src/existing.py"), existed_before=True)
    context.record_edit(workspace.resolve("src/created.py"))

    assert context.files_read == {"src/read.py"}
    assert context.files_created == {"src/created.py"}
    assert context.files_modified == {"src/existing.py"}
    assert context.render_for_provider() == (
        "Current workspace changes:\n"
        "Read:\n"
        "- src/read.py\n"
        "Modified:\n"
        "- src/existing.py\n"
        "Created:\n"
        "- src/created.py"
    )


@pytest.mark.asyncio
async def test_rova_runtime_keeps_workspace_operation_history_in_tool_results_not_system_context(
    workspace_root: Path,
    tmp_path: Path,
):
    shell_created = workspace_root / "shell-created.py"
    shell_command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'shell-created.py\').write_text(\'shell\')"'

    async def stream(model, context, options):
        results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
        if len(results) == 0:
            assert "Current workspace changes:" not in context.system_prompt
            yield StreamDone(AssistantMessage([ToolCall("read", "read", {"path": "src/read.py"})], stop_reason="tool_calls"))
        elif len(results) == 1:
            assert "Current workspace changes:" not in context.system_prompt
            assert results[-1].role == "tool"
            yield StreamDone(AssistantMessage([ToolCall("create", "write", {"path": "src/created.py", "content": "created\n"})], stop_reason="tool_calls"))
        elif len(results) == 2:
            assert "Current workspace changes:" not in context.system_prompt
            assert all(result.role == "tool" for result in results)
            yield StreamDone(AssistantMessage([ToolCall("overwrite", "write", {"path": "src/existing.py", "content": "old value\n"})], stop_reason="tool_calls"))
        elif len(results) == 3:
            assert "Current workspace changes:" not in context.system_prompt
            yield StreamDone(AssistantMessage([ToolCall("edit", "edit", {"path": "src/existing.py", "old_text": "old", "new_text": "new"})], stop_reason="tool_calls"))
        elif len(results) == 4:
            assert "Current workspace changes:" not in context.system_prompt
            yield StreamDone(AssistantMessage([ToolCall("shell", "shell", {"command": shell_command})], stop_reason="tool_calls"))
        elif len(results) == 5:
            assert "shell-created.py" not in context.system_prompt
            yield StreamDone(AssistantMessage([TextBlock("workspace work complete")]))
        else:
            raise AssertionError("unexpected runtime state")

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        approval_handler=AlwaysApprove(),
        session_root=tmp_path / "sessions",
    )

    assert (await runtime.session.prompt("inspect and update files"))[-1].text == "workspace work complete"
    assert runtime.workspace_context.files_read == {"src/read.py"}
    assert runtime.workspace_context.files_created == {"src/created.py"}
    assert runtime.workspace_context.files_modified == {"src/existing.py"}
    assert shell_created.read_text(encoding="utf-8") == "shell"
    assert all("Current workspace changes:" not in str(message) for message in runtime.agent.messages)


@pytest.mark.asyncio
async def test_rova_runtime_does_not_track_denied_or_failed_file_operations(workspace_root: Path, tmp_path: Path):
    async def stream(model, context, options):
        results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
        if len(results) == 0:
            yield StreamDone(AssistantMessage([ToolCall("write-denied", "write", {"path": "src/denied.py", "content": "no"})], stop_reason="tool_calls"))
        elif len(results) == 1:
            assert results[-1].is_error is True
            assert "Current workspace changes:" not in context.system_prompt
            yield StreamDone(AssistantMessage([TextBlock("denial observed")]))
        else:
            raise AssertionError("unexpected runtime state")

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace_root,
        approval_handler=AlwaysDeny(),
        session_root=tmp_path / "sessions",
    )

    assert (await runtime.session.prompt("write a file"))[-1].text == "denial observed"
    assert runtime.workspace_context.files_read == set()
    assert runtime.workspace_context.files_created == set()
    assert runtime.workspace_context.files_modified == set()
    assert not (workspace_root / "src" / "denied.py").exists()


@pytest.mark.asyncio
async def test_workspace_context_keeps_invalid_write_inside_existing_controlled_tool_error_boundary(workspace_root: Path):
    workspace = Workspace(workspace_root)
    workspace_context = workspace_module.WorkspaceContext(workspace)
    registry = ToolRegistry(
        build_controlled_coding_tools(
            workspace,
            DefaultCodingToolPolicy(),
            AlwaysApprove(),
            workspace_context,
        )
    )

    result = await registry.execute(ToolCall("invalid-write", "write", {"path": "../outside.py", "content": "no"}))

    assert result.is_error is True
    assert result.metadata["policy_decision"] == "require_approval"
    assert workspace_context.files_created == set()
    assert workspace_context.files_modified == set()
