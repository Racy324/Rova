from __future__ import annotations

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock
from rova.ai.models import Model
from rova.app.runtime import build_rova_runtime
from rova.trace import JsonlTraceStore


@pytest.mark.asyncio
async def test_runtime_prompt_persists_one_main_agent_run_trace(tmp_path) -> None:
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    trace_root = tmp_path / "traces"
    runtime = build_rova_runtime(
        model=Model("mock"),
        stream_fn=stream,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
        trace_root=trace_root,
        experience_review_enabled=False,
    )
    try:
        responses = await runtime.prompt("hello")
    finally:
        await runtime.close()

    traces = JsonlTraceStore(trace_root / "runs.jsonl").load_all()
    assert [response.text for response in responses] == ["done"]
    assert len(traces) == 1
    trace = traces[0]
    assert trace.session_id == runtime.session.session_id
    assert trace.input_entry_id is not None
    assert trace.input_message == "hello"
    assert len(trace.steps) == 1
    assert trace.steps[0].usage.estimated_input_tokens is not None
    assert "Runtime facts:" in runtime.agent.last_context.system_prompt
