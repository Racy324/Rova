import asyncio
import os
import sys
from pathlib import Path

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.app.workspace import (
    CodingToolError,
    Workspace,
    WorkspaceError,
    create_edit_tool,
    create_list_dir_tool,
    create_read_tool,
    create_search_tool,
    create_shell_tool,
    create_write_tool,
)


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "nested").mkdir()
    (root / "src" / "app.py").write_text("first\nsecond\nthird\n", encoding="utf-8")
    (root / "nested" / "match.txt").write_text("needle\nother needle\n", encoding="utf-8")
    (root / ".git" / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01needle")
    return root


def test_workspace_resolves_relative_and_absolute_paths_and_displays_stable_relative_path(workspace_root: Path):
    workspace = Workspace(workspace_root)
    expected = (workspace_root / "src" / "app.py").resolve()
    assert workspace.resolve("src/app.py") == expected
    assert workspace.resolve(str(expected)) == expected
    assert workspace.display_path(expected) == "src/app.py"


def test_workspace_rejects_parent_absolute_and_symlink_escapes(workspace_root: Path):
    workspace = Workspace(workspace_root)
    outside = workspace_root.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="escapes workspace root"):
        workspace.resolve("../secret.txt")
    with pytest.raises(WorkspaceError, match="escapes workspace root"):
        workspace.resolve(str(outside))
    link = workspace_root / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(WorkspaceError, match="escapes workspace root"):
        workspace.resolve("outside-link")


@pytest.mark.asyncio
async def test_read_tool_reads_utf8_files_and_inclusive_line_ranges(workspace_root: Path):
    tool = create_read_tool(Workspace(workspace_root))
    whole = await tool.execute("read-1", {"path": "src/app.py"})
    ranged = await tool.execute("read-2", {"path": "src/app.py", "start_line": 2, "end_line": 3})
    assert whole.content == [TextBlock("src/app.py (lines 1-3)\n1 | first\n2 | second\n3 | third")]
    assert ranged.content == [TextBlock("src/app.py (lines 2-3)\n2 | second\n3 | third")]


@pytest.mark.asyncio
async def test_read_tool_rejects_invalid_ranges_and_invalid_targets(workspace_root: Path):
    tool = create_read_tool(Workspace(workspace_root))
    (workspace_root / "invalid.txt").write_bytes(b"\xff")
    for params, message in [
        ({"path": "src/app.py", "start_line": 0}, "start_line must be at least 1"),
        ({"path": "src/app.py", "end_line": 0}, "end_line must be at least start_line"),
        ({"path": "src/app.py", "start_line": 3, "end_line": 2}, "end_line must be at least start_line"),
        ({"path": "missing.txt"}, "file not found"),
        ({"path": "src"}, "not a file"),
        ({"path": "invalid.txt"}, "UTF-8 text"),
        ({"path": "../secret.txt"}, "escapes workspace root"),
    ]:
        with pytest.raises((CodingToolError, WorkspaceError), match=message):
            await tool.execute("read-error", params)


@pytest.mark.asyncio
async def test_list_dir_tool_lists_one_sorted_level_with_entry_types(workspace_root: Path):
    link = workspace_root / "link"
    try:
        link.symlink_to(workspace_root / "src", target_is_directory=True)
    except OSError:
        link = None
    tool = create_list_dir_tool(Workspace(workspace_root))
    result = await tool.execute("list-1", {})
    lines = result.content[0].text.splitlines()
    assert lines == sorted(lines)
    assert "binary.bin [file]" in lines
    assert "src [directory]" in lines
    assert "nested [directory]" in lines
    assert all("app.py" not in line for line in lines)
    if link is not None:
        assert "link [symlink]" in lines


@pytest.mark.asyncio
async def test_list_dir_tool_rejects_missing_file_and_escape_paths(workspace_root: Path):
    tool = create_list_dir_tool(Workspace(workspace_root))
    for path, message in [("missing", "directory not found"), ("src/app.py", "not a directory"), ("../", "escapes workspace root")]:
        with pytest.raises((CodingToolError, WorkspaceError), match=message):
            await tool.execute("list-error", {"path": path})


@pytest.mark.asyncio
async def test_search_tool_recurses_deterministically_and_treats_query_as_data(workspace_root: Path):
    tool = create_search_tool(Workspace(workspace_root))
    result = await tool.execute("search-1", {"query": "needle"})
    assert result.content[0].text.splitlines() == ["nested/match.txt:1: needle", "nested/match.txt:2: other needle"]
    subtree = await tool.execute("search-2", {"query": "needle", "path": "nested", "max_results": 1})
    assert subtree.content == [TextBlock("nested/match.txt:1: needle")]
    shell_like = await tool.execute("search-3", {"query": "$(echo executed)"})
    assert shell_like.content == [TextBlock("no results")]


@pytest.mark.asyncio
async def test_search_tool_rejects_invalid_options_and_escape(workspace_root: Path):
    tool = create_search_tool(Workspace(workspace_root))
    for params, message in [
        ({"query": ""}, "query must not be empty"),
        ({"query": "needle", "max_results": 0}, "max_results must be at least 1"),
        ({"query": "needle", "path": "../"}, "escapes workspace root"),
    ]:
        with pytest.raises((CodingToolError, WorkspaceError), match=message):
            await tool.execute("search-error", params)


@pytest.mark.asyncio
async def test_write_tool_creates_replaces_utf8_and_rejects_bad_targets(workspace_root: Path):
    tool = create_write_tool(Workspace(workspace_root))
    await tool.execute("write-1", {"path": "created.txt", "content": "café"})
    await tool.execute("write-2", {"path": "created.txt", "content": "updated"})
    assert (workspace_root / "created.txt").read_text(encoding="utf-8") == "updated"
    for path, message in [("missing/child.txt", "parent directory does not exist"), ("src", "not a regular file"), ("../secret.txt", "escapes workspace root")]:
        with pytest.raises((CodingToolError, WorkspaceError), match=message):
            await tool.execute("write-error", {"path": path, "content": "value"})


@pytest.mark.asyncio
async def test_edit_tool_replaces_exact_text_and_preserves_remaining_content(workspace_root: Path):
    path = workspace_root / "edit.txt"
    path.write_text("prefix target suffix", encoding="utf-8")
    tool = create_edit_tool(Workspace(workspace_root))
    result = await tool.execute("edit-1", {"path": "edit.txt", "old_text": "target", "new_text": "replacement"})
    assert result.content == [TextBlock("edit.txt: replacements=1")]
    assert path.read_text(encoding="utf-8") == "prefix replacement suffix"


@pytest.mark.asyncio
async def test_edit_tool_rejects_no_or_ambiguous_matches_and_supports_replace_all(workspace_root: Path):
    path = workspace_root / "edit.txt"
    path.write_text("repeat repeat", encoding="utf-8")
    tool = create_edit_tool(Workspace(workspace_root))
    with pytest.raises(CodingToolError, match="not found"):
        await tool.execute("edit-none", {"path": "edit.txt", "old_text": "missing", "new_text": "x"})
    with pytest.raises(CodingToolError, match="occurs 2 times"):
        await tool.execute("edit-many", {"path": "edit.txt", "old_text": "repeat", "new_text": "x"})
    result = await tool.execute("edit-all", {"path": "edit.txt", "old_text": "repeat", "new_text": "x", "replace_all": True})
    assert result.content == [TextBlock("edit.txt: replacements=2")]
    assert path.read_text(encoding="utf-8") == "x x"
    with pytest.raises(WorkspaceError, match="escapes workspace root"):
        await tool.execute("edit-escape", {"path": "../secret.txt", "old_text": "a", "new_text": "b"})


@pytest.mark.asyncio
async def test_shell_tool_uses_workspace_captures_output_and_nonzero_exit(workspace_root: Path):
    tool = create_shell_tool(Workspace(workspace_root))
    command = f'"{sys.executable}" -c "import os, sys; print(os.getcwd()); print(\'stderr\', file=sys.stderr); raise SystemExit(3)"'
    result = await tool.execute("shell-1", {"command": command, "timeout_seconds": 5})
    text = result.content[0].text.replace("\\", "/")
    assert "exit_code: 3" in text
    assert "timed_out: false" in text
    assert workspace_root.as_posix() in text
    assert "stdout:\n" in text
    assert "stderr:\nstderr" in text
    assert result.metadata == {
        "command": command,
        "exit_code": 3,
        "timed_out": False,
        "outcome": "command_nonzero_exit",
    }


def test_shell_tool_describes_workspace_as_cwd_not_filesystem_sandbox(workspace_root: Path):
    tool = create_shell_tool(Workspace(workspace_root))

    assert tool.tool.description == (
        "Run a local host shell command with the workspace as its working directory. "
        "Shell commands require approval and are not filesystem sandboxed."
    )


@pytest.mark.asyncio
async def test_shell_tool_times_out_without_interactive_stdin(workspace_root: Path):
    tool = create_shell_tool(Workspace(workspace_root))
    command = f'"{sys.executable}" -c "import time; time.sleep(2)"'
    result = await tool.execute("shell-timeout", {"command": command, "timeout_seconds": 1})
    assert "exit_code: null" in result.content[0].text
    assert "timed_out: true" in result.content[0].text
    assert result.metadata == {
        "command": command,
        "exit_code": None,
        "timed_out": True,
        "outcome": "command_timeout",
    }


@pytest.mark.asyncio
async def test_shell_tool_cancellation_terminates_a_started_command(workspace_root: Path):
    tool = create_shell_tool(Workspace(workspace_root))
    started = workspace_root / "started.txt"
    completed = workspace_root / "completed.txt"
    command = (
        f'"{sys.executable}" -c "from pathlib import Path; import time; '
        "Path('started.txt').write_text('started'); time.sleep(0.5); Path('completed.txt').write_text('completed'); time.sleep(1)\""
    )
    task = asyncio.create_task(tool.execute("shell-cancel", {"command": command, "timeout_seconds": 40}))
    for _ in range(100):
        if started.exists():
            break
        await asyncio.sleep(0.01)
    assert started.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.75)

    assert not completed.exists()


@pytest.mark.asyncio
async def test_shell_tool_does_not_block_on_standard_input(workspace_root: Path):
    tool = create_shell_tool(Workspace(workspace_root))
    command = f'"{sys.executable}" -c "import sys; print(repr(sys.stdin.read()))"'
    result = await tool.execute("shell-stdin", {"command": command})
    assert "exit_code: 0" in result.content[0].text
    assert "stdout:\n''" in result.content[0].text


@pytest.mark.asyncio
async def test_shell_tool_strips_provider_keys_from_injected_child_environment_and_results(workspace_root: Path):
    blocked_values = ["fake-openai-key", "fake-Rova-key", "fake-anthropic-key"]
    runtime_variables = {
        name: os.environ.get(name, f"test-{name.lower()}")
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP")
    }
    environment = {
        "OPENAI_API_KEY": blocked_values[0],
        "ROVA_PROVIDER_KEY": blocked_values[1],
        "ANTHROPIC_API_KEY": blocked_values[2],
        **runtime_variables,
    }
    tool = create_shell_tool(Workspace(workspace_root), environment=environment)
    command = (
        f'"{sys.executable}" -c "import os; '
        "print(repr(os.getenv('OPENAI_API_KEY'))); "
        "print(repr(os.getenv('ROVA_PROVIDER_KEY'))); "
        "print(repr(os.getenv('ANTHROPIC_API_KEY'))); "
        f"print({blocked_values[0]!r}); "
        "print(os.environ['SYSTEMROOT']); print(os.environ['WINDIR']); "
        "print(os.environ['COMSPEC']); print(os.environ['TEMP']); print(os.environ['TMP'])\""
    )

    result = await tool.execute("shell-sanitized-environment", {"command": command})

    assert "stdout:\nNone\nNone\nNone" in result.content[0].text
    assert all(value in result.content[0].text for value in runtime_variables.values())
    assert all(value not in result.content[0].text for value in blocked_values)
    assert all(value not in str(result.metadata) for value in blocked_values)


@pytest.mark.asyncio
async def test_read_tool_runs_through_agenttool_registry_without_agent_core_changes(workspace_root: Path):
    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("read-1", "read", {"path": "src/app.py"})], stop_reason="tool_calls"))
            return
        assert any(isinstance(message, ToolResultMessage) and "1 | first" in message.text for message in context.messages)
        yield StreamDone(AssistantMessage([TextBlock("observed real tool result")]))

    agent = Agent(Model(provider="mock"), "", [create_read_tool(Workspace(workspace_root))], stream)
    result = await agent.run([UserMessage("read the file")])
    assert result[-1].text == "observed real tool result"
