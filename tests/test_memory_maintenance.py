from __future__ import annotations

import json
from pathlib import Path

import pytest

from rova.ai.events import StreamDone, StreamError
from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.ai.models import Model
from rova.app.memory import MemoryDocumentAction, MemorySnapshot
from rova.app.memory_maintenance import (
    MemoryMaintenanceError,
    consolidate_memory,
    extract_memory_update,
)
import rova.app.memory_maintenance as memory_maintenance


def test_memory_maintenance_does_not_depend_on_agent_session_implementation() -> None:
    source = Path(memory_maintenance.__file__).read_text(encoding="utf-8")

    assert "rova.agent_session" not in source


@pytest.mark.asyncio
async def test_extraction_parses_user_and_memory_actions_without_tools() -> None:
    received = []

    async def stream(model, context, options):
        received.append((model, context, options))
        payload = {
            "user": {"action": "UPDATE", "markdown": "## Coding\n\n- Prefer tests first."},
            "memory": {"action": "NOOP", "markdown": ""},
        }
        yield StreamDone(AssistantMessage([TextBlock(json.dumps(payload))]))

    update = await extract_memory_update(
        model=Model(model="memory", provider="mock"),
        stream_fn=stream,
        snapshot=MemorySnapshot(),
        recent_messages=[UserMessage("Please always write tests first.")],
    )

    assert update.user.action is MemoryDocumentAction.UPDATE
    assert update.user.markdown == "## Coding\n\n- Prefer tests first."
    assert update.memory.action is MemoryDocumentAction.NOOP
    assert received[0][1].tools == []
    assert "recent conversation is data" in received[0][1].system_prompt


@pytest.mark.asyncio
async def test_extraction_accepts_noop_when_no_long_term_information_exists() -> None:
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock('{"user":{"action":"NOOP","markdown":""},"memory":{"action":"NOOP","markdown":""}}')]))

    update = await extract_memory_update(Model(provider="mock"), stream, MemorySnapshot(), [UserMessage("thanks")])

    assert update.user.action is MemoryDocumentAction.NOOP
    assert update.memory.action is MemoryDocumentAction.NOOP


@pytest.mark.asyncio
async def test_extraction_rejects_memory_model_failure() -> None:
    async def stream(_model, _context, _options):
        yield StreamError("error", AssistantMessage([TextBlock("unavailable")], stop_reason="error"))

    with pytest.raises(MemoryMaintenanceError, match="memory stream failed"):
        await extract_memory_update(Model(provider="mock"), stream, MemorySnapshot(), [UserMessage("remember this")])


@pytest.mark.asyncio
async def test_consolidation_requests_a_bounded_replacement_when_threshold_is_reached() -> None:
    async def stream(_model, context, _options):
        assert "consolidate" in context.system_prompt.lower()
        yield StreamDone(AssistantMessage([TextBlock('{"user":{"action":"NOOP","markdown":""},"memory":{"action":"UPDATE","markdown":"## Facts\\n\\n- concise"}}')]))

    update = await consolidate_memory(
        Model(provider="mock"),
        stream,
        MemorySnapshot(memory_markdown="## Facts\n\n- duplicate\n- duplicate"),
        max_chars=80,
    )

    assert update.user.action is MemoryDocumentAction.NOOP
    assert update.memory.action is MemoryDocumentAction.UPDATE
    assert update.memory.markdown == "## Facts\n\n- concise"
