from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from rova.agent_core.tool_output import (
    ArtifactReference,
    ProcessedToolOutput,
    ToolOutputLimits,
    ToolOutputProcessor,
)
from rova.agent_session.agent_session import AgentSession
from rova.artifacts import FileArtifactStore
from rova.agent_core.agent import Agent
from rova.agent_core.tools import AgentTool, AgentToolResult
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.trace import TraceRecorder


class MemoryArtifactStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def write_text(self, raw_text: str, **kwargs) -> ArtifactReference:
        self.calls.append({"raw_text": raw_text, **kwargs})
        return ArtifactReference(
            artifact_id=f"artifact-{len(self.calls)}",
            media_type="text/plain; charset=utf-8",
            byte_count=len(raw_text.encode("utf-8")),
            sha256="a" * 64,
            created_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
            run_id=kwargs["run_id"],
            session_id=kwargs["session_id"],
        )


def test_output_at_limits_is_unchanged_and_persisted():
    store = MemoryArtifactStore()
    processor = ToolOutputProcessor(store)

    processed = processor.process("read-1", "read", "one\ntwo", {}, is_error=False)

    assert processed.preview == [TextBlock("one\ntwo")]
    assert processed.metadata.truncated is False
    assert processed.metadata.externalized is True
    assert processed.metadata.preview_truncated is False
    assert processed.metadata.artifact_ref == "artifact-1"
    assert processed.metadata.original_size_chars == len("one\ntwo")
    assert processed.metadata.preview_size_chars == len("one\ntwo")
    assert processed.metadata.original_byte_count == len("one\ntwo".encode("utf-8"))
    assert store.calls[0]["raw_text"] == "one\ntwo"


def test_named_strategies_emit_bounded_deterministic_preview():
    raw = "one\ntwo\nthree\nfour\nfive\nsix"
    processor = ToolOutputProcessor(MemoryArtifactStore(), ToolOutputLimits(max_lines=3, max_bytes=80))

    read = processor.process("read-1", "read", raw, {}, is_error=False)
    shell = processor.process("shell-1", "shell", raw, {}, is_error=False)
    search = processor.process("search-1", "search", raw, {}, is_error=False)

    assert read.metadata.strategy == "head"
    assert shell.metadata.strategy == "tail"
    assert search.metadata.strategy == "head_tail"
    assert all(item.metadata.truncated for item in (read, shell, search))
    assert all(item.metadata.externalized is True for item in (read, shell, search))
    assert all(item.metadata.preview_truncated is True for item in (read, shell, search))
    assert "one" in read.preview[0].text and "six" not in read.preview[0].text
    assert "six" in shell.preview[0].text and "one" not in shell.preview[0].text
    assert "one" in search.preview[0].text and "six" in search.preview[0].text
    assert all(item.preview[0].text.count("[tool output truncated:") == 1 for item in (read, shell, search))
    assert all(item.metadata.preview_byte_count <= 80 for item in (read, shell, search))


def test_overlong_unicode_line_is_utf8_safe_and_records_partial_line():
    raw = "你" * 100
    store = MemoryArtifactStore()
    processor = ToolOutputProcessor(store, ToolOutputLimits(max_lines=5, max_bytes=100))

    processed = processor.process("read-1", "read", raw, {}, is_error=False)

    preview = processed.preview[0].text
    assert preview.encode("utf-8").decode("utf-8") == preview
    assert processed.metadata.preview_byte_count <= 100
    assert processed.metadata.truncated_line_count == 1
    assert store.calls[0]["raw_text"] == raw


def test_multiple_blocks_and_errors_preserve_raw_join_and_error_flag():
    store = MemoryArtifactStore()
    processor = ToolOutputProcessor(store)

    processed = processor.process("error-1", "other", [TextBlock("one"), TextBlock("two")], {"outcome": "tool_execution_error"}, is_error=True)

    assert processed.preview == [TextBlock("onetwo")]
    assert store.calls[0]["raw_text"] == "onetwo"
    assert store.calls[0]["is_error"] is True


@pytest.mark.asyncio
async def test_registry_sends_only_preview_to_agent_and_trace():
    store = MemoryArtifactStore()
    processor = ToolOutputProcessor(store, ToolOutputLimits(max_lines=2, max_bytes=100))

    async def execute(tool_call_id, params):
        return AgentToolResult([TextBlock("first\nsecond\nsecret omitted\nfourth")])

    tool = AgentTool(Tool("read", "read", {}), execute)

    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("read-1", "read", {})], stop_reason="tool_calls"))
            return
        result = next(message for message in context.messages if isinstance(message, ToolResultMessage))
        assert "secret omitted" not in result.text
        assert result.metadata["tool_output"]["artifact"]["artifact_id"] == "artifact-1"
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model("mock"), "", [tool], stream, tool_output_processor=processor)
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("read")]))

    execution = trace.tool_executions[0]
    assert "secret omitted" not in execution.result
    assert execution.metadata["tool_output"]["artifact"]["artifact_id"] == "artifact-1"
    assert store.calls[0]["raw_text"].endswith("secret omitted\nfourth")


@pytest.mark.asyncio
async def test_traced_durable_session_persists_preview_and_binds_artifact_identities(tmp_path):
    async def execute(tool_call_id, params):
        return AgentToolResult([TextBlock("first\nsecond\nthird\nfourth")])

    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("read-1", "read", {})], stop_reason="tool_calls"))
            return
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    processor = ToolOutputProcessor(FileArtifactStore(tmp_path / "artifacts"), ToolOutputLimits(max_lines=2, max_bytes=100))
    agent = Agent(Model("mock"), "", [AgentTool(Tool("read", "read", {}), execute)], stream, tool_output_processor=processor)
    session = AgentSession.create(agent, session_root=tmp_path / "sessions")
    _, trace = await TraceRecorder().capture_run(
        agent,
        lambda: session.prompt("read"),
        session_id=session.session_id,
    )

    tool_message = next(message for message in agent.messages if isinstance(message, ToolResultMessage))
    assert "third" not in tool_message.text
    metadata = tool_message.metadata["tool_output"]
    envelope = json.loads(next((tmp_path / "artifacts").glob("*.json")).read_text(encoding="utf-8"))
    assert envelope["raw_output"] == "first\nsecond\nthird\nfourth"
    assert envelope["run_id"] == trace.run_id
    assert envelope["session_id"] == session.session_id
    assert metadata["artifact"]["artifact_id"] == envelope["artifact_id"]
