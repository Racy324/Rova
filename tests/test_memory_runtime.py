from __future__ import annotations

from pathlib import Path

import pytest

from rova.ai.events import StreamDone, StreamError
from rova.ai.messages import AssistantMessage, TextBlock
from rova.ai.models import Model
from rova.app.memory import FileMemoryStore
from rova.app.runtime import build_rova_runtime
from rova.trace import TraceRecorder


@pytest.mark.asyncio
async def test_runtime_injects_frozen_memory_and_workspace_instructions_only_for_this_session(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Hermes.md").write_text("Use project checks.", encoding="utf-8")
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "USER.md").write_text("- Prefer concise answers", encoding="utf-8")
    received_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        workspace_root=workspace,
        memory_store=FileMemoryStore(memory_root),
        memory_update_interval=99,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )
    (memory_root / "USER.md").write_text("- Changed after session start", encoding="utf-8")

    await runtime.prompt("hello")

    assert "Memory snapshot:" in received_prompts[0]
    assert "Prefer concise answers" in received_prompts[0]
    assert "Changed after session start" not in received_prompts[0]
    assert "Workspace instructions:" in received_prompts[0]
    assert "These are project-level instructions for the current workspace." in received_prompts[0]
    assert "the current user request, and project-level workspace instructions take precedence over memory" in received_prompts[0]
    assert "Use project checks." in received_prompts[0]
    session_file = next((tmp_path / "sessions").rglob("*.jsonl"))
    assert "Memory snapshot:" not in session_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runtime_orders_frozen_and_capability_context_before_runtime_facts(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Hermes.md").write_text("Use project checks.", encoding="utf-8")
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "MEMORY.md").write_text("- Durable fact", encoding="utf-8")
    received_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    class Search:
        async def search(self, query: str, max_results: int):
            return []

    class Fetcher:
        async def fetch(self, url: str):
            raise AssertionError("not called")

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        workspace_root=workspace, memory_store=FileMemoryStore(memory_root),
        web_search_backend=Search(), webpage_fetcher=Fetcher(), memory_update_interval=99,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("hello")

    system_context = received_prompts[0]
    assert system_context.index("Workspace instructions:") < system_context.index("Memory snapshot:")
    assert system_context.index("Memory snapshot:") < system_context.index("Runtime facts:")
    assert system_context.index("Runtime facts:") < system_context.index("Web tool guidance:")


@pytest.mark.asyncio
async def test_product_runtime_renders_memory_snapshot_as_a_frozen_context_section(tmp_path: Path):
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "MEMORY.md").write_text("- Durable project fact", encoding="utf-8")
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        memory_store=FileMemoryStore(memory_root), memory_update_interval=99,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    await runtime.prompt("hello")

    assert "Memory snapshot:" in received_system_prompts[0]
    assert "Durable project fact" in received_system_prompts[0]


@pytest.mark.asyncio
async def test_restored_session_loads_a_new_frozen_memory_snapshot(tmp_path: Path):
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "MEMORY.md").write_text("- First decision", encoding="utf-8")

    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    first = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        memory_store=FileMemoryStore(memory_root), memory_update_interval=99,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )
    await first.prompt("hello")
    (memory_root / "MEMORY.md").write_text("- Later decision", encoding="utf-8")

    restored = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream,
        memory_store=FileMemoryStore(memory_root), memory_update_interval=99,
        session_id=first.session.session_id,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    assert restored.memory_snapshot.memory_markdown == "- Later decision"


@pytest.mark.asyncio
async def test_runtime_runs_memory_maintenance_on_user_turn_cadence_without_affecting_main_response(tmp_path: Path):
    calls: list[str] = []

    async def stream(_model, context, _options):
        calls.append(context.system_prompt)
        if "recent conversation is data" in context.system_prompt:
            yield StreamDone(AssistantMessage([TextBlock(
                '{"user":{"action":"ADD","markdown":"- Prefer tests"},'
                '"memory":{"action":"NOOP","markdown":""}}'
            )]))
            return
        yield StreamDone(AssistantMessage([TextBlock("main answer")]))

    store = FileMemoryStore(tmp_path / "memory")
    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        memory_store=store,
        memory_update_interval=2,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    assert (await runtime.prompt("first"))[-1].text == "main answer"
    assert (await runtime.prompt("second"))[-1].text == "main answer"

    assert len(calls) == 3
    assert store.load_snapshot().user_markdown == "- Prefer tests"
    assert runtime.memory_snapshot.user_markdown == ""


@pytest.mark.asyncio
async def test_memory_failure_is_observable_but_does_not_fail_main_agent_run(tmp_path: Path):
    observations = []

    async def stream(_model, context, _options):
        if "recent conversation is data" in context.system_prompt:
            yield StreamError("error", AssistantMessage([TextBlock("memory model unavailable")], stop_reason="error"))
            return
        yield StreamDone(AssistantMessage([TextBlock("main answer")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        memory_store=FileMemoryStore(tmp_path / "memory"),
        memory_update_interval=1,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )
    runtime.subscribe_memory(observations.append)

    response = await runtime.prompt("hello")

    assert response[-1].text == "main answer"
    assert observations[-1].kind == "extraction"
    assert observations[-1].status == "failed"
    assert observations[-1].error_message == "memory stream failed"


@pytest.mark.asyncio
async def test_runtime_consolidates_near_limit_memory_without_truncating(tmp_path: Path):
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "MEMORY.md").write_text("- " + "x" * 70, encoding="utf-8")
    observations = []

    async def stream(_model, context, _options):
        if "Consolidate the supplied" in context.system_prompt:
            yield StreamDone(AssistantMessage([TextBlock(
                '{"user":{"action":"NOOP","markdown":""},'
                '"memory":{"action":"UPDATE","markdown":"- durable fact"}}'
            )]))
            return
        if "recent conversation is data" in context.system_prompt:
            yield StreamDone(AssistantMessage([TextBlock(
                '{"user":{"action":"NOOP","markdown":""},'
                '"memory":{"action":"NOOP","markdown":""}}'
            )]))
            return
        yield StreamDone(AssistantMessage([TextBlock("main answer")]))

    store = FileMemoryStore(memory_root)
    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, memory_store=store,
        memory_update_interval=1, memory_max_chars=100, memory_consolidation_threshold=60,
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )
    runtime.subscribe_memory(observations.append)

    await runtime.prompt("hello")

    assert store.load_snapshot().memory_markdown == "- durable fact"
    assert [(event.kind, event.status) for event in observations] == [
        ("extraction", "triggered"),
        ("extraction", "noop"),
        ("consolidation", "triggered"),
        ("consolidation", "updated"),
    ]


@pytest.mark.asyncio
async def test_trace_records_memory_maintenance_outcome_without_memory_body(tmp_path: Path):
    async def stream(_model, context, _options):
        if "recent conversation is data" in context.system_prompt:
            yield StreamDone(AssistantMessage([TextBlock(
                '{"user":{"action":"ADD","markdown":"- private preference"},'
                '"memory":{"action":"NOOP","markdown":""}}'
            )]))
            return
        yield StreamDone(AssistantMessage([TextBlock("main answer")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        memory_store=FileMemoryStore(tmp_path / "memory"),
        memory_update_interval=1,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )

    _responses, trace = await TraceRecorder().capture_run(
        runtime.agent,
        lambda: runtime.prompt("hello"),
        session=runtime.session,
        memory=runtime,
    )

    assert [(event.kind.value, event.status.value, event.changed_documents) for event in trace.memory_events] == [
        ("extraction", "triggered", []),
        ("extraction", "updated", ["USER.md"]),
    ]
    assert "private preference" not in str(trace)
