from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from rova.agent_core.agent import Agent
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.app.extensions import ContextContribution, ExtensionAPI
from rova.app.runtime import ROVA_SYSTEM_PROMPT, build_rova_runtime


@pytest.fixture(autouse=True)
def _isolate_default_rova_data_dir(monkeypatch, tmp_path: Path) -> None:
    """Default-on review state must not leak from one extension test into another."""
    monkeypatch.setenv("ROVA_DATA_DIR", str(tmp_path / "rova-data"))


def _write_extension(root: Path, name: str, source: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_extension_tool_registration_requires_an_explicit_execution_mode() -> None:
    async def run(_tool_call_id, _params):
        return AgentToolResult([TextBlock("unused")])

    api = ExtensionAPI(())
    with api.extension_setup("mode-check"):
        with pytest.raises(ValueError, match="execution_mode"):
            api.register_tool(AgentTool(Tool("unmarked", "unmarked", {}), run))
        api.register_tool(
            AgentTool(
                Tool("marked", "marked", {}),
                run,
                execution_mode=ToolExecutionMode.PARALLEL,
            )
        )

    assert [tool.tool.name for tool in api.tools] == ["marked"]


@pytest.mark.asyncio
async def test_extension_tool_is_available_to_and_callable_by_agent(tmp_path: Path) -> None:
    extensions = tmp_path / "extensions"
    _write_extension(
        extensions,
        "greeting",
        """
        from rova.ai.messages import TextBlock
        from rova.ai.tools import Tool
        from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

        async def greet(_tool_call_id, params):
            return AgentToolResult([TextBlock(f"hello, {params['name']}")])

        def setup(api):
            api.register_tool(AgentTool(Tool("extension_greet", "Greet a name.", {"name": str}), greet, execution_mode=ToolExecutionMode.PARALLEL))
        """,
    )

    async def stream(_model, context, _options):
        results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
        if not results:
            yield StreamDone(AssistantMessage([ToolCall("greet-1", "extension_greet", {"name": "Rova"})], stop_reason="tool_calls"))
            return
        yield StreamDone(AssistantMessage([TextBlock(results[0].text)]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        extension_roots=(extensions,),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    response = (await runtime.prompt("use the extension"))[-1]

    assert response.text == "hello, Rova"
    assert "extension_greet" in [tool.name for tool in runtime.agent.registry.schemas]
    assert runtime.extension_load_report.loaded == ("greeting",)


@pytest.mark.asyncio
async def test_extension_hook_receives_agent_lifecycle_event_and_context_is_rendered(tmp_path: Path) -> None:
    extensions = tmp_path / "extensions"
    marker = tmp_path / "agent-end.txt"
    _write_extension(
        extensions,
        "observer",
        f"""
        from pathlib import Path
        from rova.app.extensions import ContextContribution

        def on_agent_end(event):
            Path({str(marker)!r}).write_text(event.type, encoding="utf-8")

        def provide_context():
            return ContextContribution("demo", "Extension context fact.")

        def setup(api):
            api.on("agent_end", on_agent_end)
            api.register_context_provider(provide_context)
        """,
    )
    received_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        extension_roots=(extensions,),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert (await runtime.prompt("hello"))[-1].text == "done"
    assert marker.read_text(encoding="utf-8") == "agent_end"
    assert "Extension context:" in received_prompts[0]
    assert "Name: demo" in received_prompts[0]
    assert "Extension context fact." in received_prompts[0]
    assert runtime.agent.system_prompt == ROVA_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_failing_extension_handler_does_not_block_later_handler(tmp_path: Path) -> None:
    extensions = tmp_path / "extensions"
    marker = tmp_path / "second-handler.txt"
    _write_extension(
        extensions,
        "a_failing_handler",
        """
        def setup(api):
            def fail(_event):
                raise RuntimeError("intentional hook failure")
            api.on("agent_end", fail)
        """,
    )
    _write_extension(
        extensions,
        "b_working_handler",
        f"""
        from pathlib import Path
        def setup(api):
            api.on("agent_end", lambda _event: Path({str(marker)!r}).write_text("called", encoding="utf-8"))
        """,
    )

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, extension_roots=(extensions,),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("hello")

    assert marker.read_text(encoding="utf-8") == "called"
    assert [(issue.extension_name, issue.phase) for issue in runtime.extension_runtime_issues] == [
        ("a_failing_handler", "event"),
    ]


@pytest.mark.asyncio
async def test_extension_setup_rolls_back_on_base_exception_while_preserving_propagation() -> None:
    class SetupInterrupted(BaseException):
        pass

    async def run(_id, _params):
        return AgentToolResult([TextBlock("unused")])

    hook_calls: list[str] = []
    api = ExtensionAPI(())
    with pytest.raises(SetupInterrupted):
        with api.extension_setup("interrupted"):
            api.register_tool(AgentTool(Tool("rolled_back_tool", "rolled back", {}), run, execution_mode=ToolExecutionMode.PARALLEL))
            api.on("agent_end", lambda _event: hook_calls.append("called"))
            api.register_context_provider(lambda: ContextContribution("rolled-back", "must not survive"))
            raise SetupInterrupted()

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model(provider="mock"), "", [], stream)
    api.bind_event_hooks(agent)
    await agent.run([UserMessage("hello")])

    assert api.tools == ()
    assert hook_calls == []
    assert api.render_context_sections() == []


@pytest.mark.asyncio
async def test_failed_extension_rolls_back_tool_hook_and_context_before_loading_neighbors(tmp_path: Path) -> None:
    extensions = tmp_path / "extensions"
    partial_hook_marker = tmp_path / "partial-hook.txt"
    received_prompts: list[str] = []
    _write_extension(extensions, "a_broken_import", "raise RuntimeError('broken import')")
    _write_extension(
        extensions,
        "b_partial_duplicate",
        f"""
        from pathlib import Path
        from rova.ai.messages import TextBlock
        from rova.ai.tools import Tool
        from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode
        from rova.app.extensions import ContextContribution

        async def run(_id, _params):
            return AgentToolResult([TextBlock("unused")])

        def partial_hook(_event):
            Path({str(partial_hook_marker)!r}).write_text("called", encoding="utf-8")

        def partial_context():
            return ContextContribution("partial", "partial context must not survive")

        def setup(api):
            api.register_tool(AgentTool(Tool("temporary_extension_tool", "temporary", {{}}), run, execution_mode=ToolExecutionMode.PARALLEL))
            api.on("agent_end", partial_hook)
            api.register_context_provider(partial_context)
            api.register_tool(AgentTool(Tool("skill_view", "duplicate", {{}}), run, execution_mode=ToolExecutionMode.PARALLEL))
        """,
    )
    _write_extension(
        extensions,
        "c_working",
        """
        from rova.ai.messages import TextBlock
        from rova.ai.tools import Tool
        from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode
        async def run(_id, _params):
            return AgentToolResult([TextBlock("available")])
        def setup(api):
            api.register_tool(AgentTool(Tool("available_extension_tool", "available", {}), run, execution_mode=ToolExecutionMode.PARALLEL))
        """,
    )

    async def stream(_model, context, _options):
        received_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, extension_roots=(extensions,),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    names = [tool.name for tool in runtime.agent.registry.schemas]
    assert "available_extension_tool" in names
    assert "temporary_extension_tool" not in names
    assert runtime.extension_load_report.loaded == ("c_working",)
    assert [(issue.extension_name, issue.phase) for issue in runtime.extension_load_report.issues] == [
        ("a_broken_import", "import"),
        ("b_partial_duplicate", "setup"),
    ]
    assert "duplicate tool name: skill_view" in runtime.extension_load_report.issues[1].message
    assert (await runtime.prompt("hello"))[-1].text == "done"
    assert not partial_hook_marker.exists()
    assert "partial context must not survive" not in received_prompts[0]


def test_default_loader_discovers_user_extensions_before_workspace_extensions(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ROVA_DATA_DIR", str(data_root))
    source = """
        from rova.ai.messages import TextBlock
        from rova.ai.tools import Tool
        from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode
        async def run(_id, _params):
            return AgentToolResult([TextBlock("ok")])
        def setup(api):
            api.register_tool(AgentTool(Tool(TOOL_NAME, TOOL_NAME, {}), run, execution_mode=ToolExecutionMode.PARALLEL))
    """
    _write_extension(data_root / "extensions", "a_user", source.replace("TOOL_NAME", '"user_extension_tool"'))
    _write_extension(workspace / ".rova" / "extensions", "a_workspace", source.replace("TOOL_NAME", '"workspace_extension_tool"'))

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=workspace,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    assert runtime.extension_load_report.loaded == ("a_user", "a_workspace")
    names = [tool.name for tool in runtime.agent.registry.schemas]
    assert names.index("user_extension_tool") < names.index("workspace_extension_tool")


@pytest.mark.asyncio
async def test_extension_hook_isolation_does_not_swallow_direct_agent_subscriber_failure(tmp_path: Path) -> None:
    extensions = tmp_path / "extensions"
    extension_marker = tmp_path / "extension-hook.txt"
    _write_extension(
        extensions,
        "observer",
        f"""
        from pathlib import Path

        def extension_hook(_event):
            Path({str(extension_marker)!r}).write_text("called", encoding="utf-8")

        def setup(api):
            api.on("agent_end", extension_hook)
        """,
    )

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, extension_roots=(extensions,),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    def direct_subscriber(event):
        if event.type == "agent_end":
            raise RuntimeError("direct subscriber failed")

    runtime.agent.subscribe(direct_subscriber)

    with pytest.raises(RuntimeError, match="direct subscriber failed"):
        await runtime.prompt("hello")

    assert extension_marker.read_text(encoding="utf-8") == "called"


@pytest.mark.asyncio
async def test_context_contributors_are_ordered_dynamic_and_recomputed_per_provider_request(tmp_path: Path) -> None:
    extensions = tmp_path / "extensions"
    _write_extension(
        extensions,
        "contributors",
        """
        from rova.app.extensions import ContextContribution

        counts = {"alpha": 0, "beta": 0}

        def alpha():
            counts["alpha"] += 1
            return ContextContribution("alpha", f"alpha-{counts['alpha']}")

        def beta():
            counts["beta"] += 1
            return ContextContribution("beta", f"beta-{counts['beta']}")

        def setup(api):
            api.register_context_provider(alpha)
            api.register_context_provider(beta)
        """,
    )
    received_contexts = []

    async def stream(_model, context, _options):
        received_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, extension_roots=(extensions,),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("first")
    await runtime.prompt("second")

    first, second = received_contexts
    assert first.system_prompt.index("Name: alpha") < first.system_prompt.index("Name: beta")
    assert "alpha-1" in first.system_prompt and "beta-1" in first.system_prompt
    assert "alpha-2" in second.system_prompt and "beta-2" in second.system_prompt
    assert runtime.agent.system_prompt == ROVA_SYSTEM_PROMPT
    assert [message.content for message in runtime.agent.messages if message.role == "user"] == ["first", "second"]
    assert [tool.name for tool in runtime.agent.registry.schemas] == ["skill_view", "skill_manage", "memory_manage"]


@pytest.mark.asyncio
async def test_same_named_files_and_repeated_non_tool_registrations_are_append_only(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    workspace = tmp_path / "workspace"
    hook_marker = tmp_path / "hook-calls.txt"
    workspace.mkdir()
    monkeypatch.setenv("ROVA_DATA_DIR", str(data_root))
    _write_extension(
        data_root / "extensions",
        "shared",
        """
        from rova.ai.messages import TextBlock
        from rova.ai.tools import Tool
        from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

        async def run(_id, _params):
            return AgentToolResult([TextBlock("user")])

        def setup(api):
            api.register_tool(AgentTool(Tool("user_shared_tool", "user", {}), run, execution_mode=ToolExecutionMode.PARALLEL))
        """,
    )
    _write_extension(
        workspace / ".rova" / "extensions",
        "shared",
        f"""
        from pathlib import Path
        from rova.app.extensions import ContextContribution

        def hook(_event):
            with Path({str(hook_marker)!r}).open("a", encoding="utf-8") as handle:
                handle.write("x")

        def context():
            return ContextContribution("duplicate", "one registration")

        def setup(api):
            api.on("agent_end", hook)
            api.on("agent_end", hook)
            api.register_context_provider(context)
            api.register_context_provider(context)
        """,
    )
    received_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=workspace,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("hello")

    assert runtime.extension_load_report.loaded == ("shared", "shared")
    assert "user_shared_tool" in [tool.name for tool in runtime.agent.registry.schemas]
    assert hook_marker.read_text(encoding="utf-8") == "xx"
    assert received_prompts[0].count("Name: duplicate") == 2


def test_later_same_named_file_with_duplicate_tool_rolls_back_its_setup(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ROVA_DATA_DIR", str(data_root))
    source = """
        from rova.ai.messages import TextBlock
        from rova.ai.tools import Tool
        from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionMode

        async def run(_id, _params):
            return AgentToolResult([TextBlock("ok")])

        def setup(api):
            api.register_tool(AgentTool(Tool("shared_tool", "shared", {}), run, execution_mode=ToolExecutionMode.PARALLEL))
    """
    _write_extension(data_root / "extensions", "shared", source)
    _write_extension(workspace / ".rova" / "extensions", "shared", source)

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=workspace,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    assert runtime.extension_load_report.loaded == ("shared",)
    assert [(issue.extension_name, issue.phase) for issue in runtime.extension_load_report.issues] == [
        ("shared", "setup"),
    ]
    assert [tool.name for tool in runtime.agent.registry.schemas].count("shared_tool") == 1
    assert "duplicate tool name: shared_tool" in runtime.extension_load_report.issues[0].message


@pytest.mark.asyncio
async def test_no_extensions_preserves_existing_tool_set_and_context(tmp_path: Path) -> None:
    received_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, extension_roots=(),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    assert (await runtime.prompt("hello"))[-1].text == "done"
    assert [tool.name for tool in runtime.agent.registry.schemas] == ["skill_view", "skill_manage", "memory_manage"]
    assert "Extension context:" not in received_prompts[0]
    assert runtime.extension_load_report.loaded == ()
    assert runtime.extension_load_report.issues == ()
