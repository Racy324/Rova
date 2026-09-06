from __future__ import annotations

from pathlib import Path

from rova.agent_core.tools import ToolExecutionMode
from rova.app.memory import FileMemoryStore, create_memory_tools
from rova.app.skills import FileSkillStore, create_skill_tools
from rova.app.vision.tool import create_vision_analyze_tool
from rova.app.web.sources import ResearchSourceStore
from rova.app.web.tools import create_fetch_webpage_tool, create_web_search_tool
from rova.app.workspace import Workspace
from rova.app.workspace.controlled_tool import build_coding_tools
from rova.app.workspace.terminal import LocalTerminalBackend


class _VisionClient:
    async def analyze(self, *, question: str, image_data_url: str) -> str:
        return "unused"


class _SearchBackend:
    async def search(self, query: str, max_results: int):
        return []


class _FetchBackend:
    async def fetch(self, url: str):
        return "unused"


def test_builtin_tools_explicitly_declare_their_execution_mode(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace(workspace_root)
    terminal = LocalTerminalBackend(workspace)
    source_store = ResearchSourceStore()
    tools = [
        *build_coding_tools(workspace, terminal_backend=terminal),
        *create_skill_tools(FileSkillStore(tmp_path / "skills")),
        *create_memory_tools(FileMemoryStore(tmp_path / "memory"), max_chars=100),
        create_vision_analyze_tool(workspace, _VisionClient()),
        create_web_search_tool(source_store, _SearchBackend()),
        create_fetch_webpage_tool(source_store, _FetchBackend()),
    ]

    modes = {tool.tool.name: tool.execution_mode for tool in tools}

    assert {name: modes[name] for name in ("read", "list_dir", "search", "skill_view", "vision_analyze", "web_search", "fetch_webpage")} == {
        "read": ToolExecutionMode.PARALLEL,
        "list_dir": ToolExecutionMode.PARALLEL,
        "search": ToolExecutionMode.PARALLEL,
        "skill_view": ToolExecutionMode.PARALLEL,
        "vision_analyze": ToolExecutionMode.PARALLEL,
        "web_search": ToolExecutionMode.PARALLEL,
        "fetch_webpage": ToolExecutionMode.PARALLEL,
    }
    assert {name: modes[name] for name in ("write", "edit", "shell", "memory_manage", "skill_manage")} == {
        "write": ToolExecutionMode.SEQUENTIAL,
        "edit": ToolExecutionMode.SEQUENTIAL,
        "shell": ToolExecutionMode.SEQUENTIAL,
        "memory_manage": ToolExecutionMode.SEQUENTIAL,
        "skill_manage": ToolExecutionMode.SEQUENTIAL,
    }
