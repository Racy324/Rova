import pytest

from rova.agent_core.types import StreamFn
from rova.agent_session.compaction import (
    COMPACTION_SUMMARY_INSTRUCTION,
    generate_compaction_summary,
    serialize_conversation,
)
from rova.agent_session.context_builder import COMPACTION_SUMMARY_PREAMBLE
from rova.agent_session.summarization import (
    SummarizationError,
    SummarizationRequest,
    summarize_with_stream,
)
from rova.ai.events import Start, StreamDone, StreamError, TextDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model


def test_serialize_conversation_preserves_roles_tool_data_and_error_state_deterministically():
    transcript = serialize_conversation(
        [
            UserMessage("find the bug"),
            AssistantMessage(
                [TextBlock("I will inspect it."), ToolCall("call-2", "read_file", {"b": 2, "a": 1})],
                stop_reason="tool_calls",
            ),
            ToolResultMessage("call-2", "read_file", [TextBlock("contents")], is_error=True),
        ]
    )

    assert "USER:\nfind the bug" in transcript
    assert "ASSISTANT:\nI will inspect it." in transcript
    assert "TOOL CALL:\nid=call-2\nname=read_file\narguments={\"a\":1,\"b\":2}" in transcript
    assert "TOOL RESULT:\nid=call-2\nname=read_file\nis_error=true\ncontent=contents" in transcript


def test_serialize_conversation_omits_partial_assistant_messages():
    assert serialize_conversation(
        [AssistantMessage([TextBlock("partial")], partial=True), UserMessage("durable")]
    ) == "USER:\ndurable"


@pytest.mark.asyncio
async def test_summarizer_uses_isolated_no_tools_context_and_returns_stripped_final_text():
    observed = {}

    async def stream(model, context, options):
        observed["model"] = model
        observed["context"] = context
        yield Start(AssistantMessage([], partial=True))
        yield TextDelta("partial", AssistantMessage([TextBlock("partial")], partial=True))
        yield StreamDone(AssistantMessage([TextBlock("  concise summary  ")]))

    result = await summarize_with_stream(
        Model("model", max_tokens=100),
        stream,
        SummarizationRequest("summarize only", "historical transcript"),
    )

    assert result == "concise summary"
    assert observed["context"].tools == []
    assert observed["context"].messages == [UserMessage("summarize only\n\nhistorical transcript")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream_factory, error_text",
    [
        (
            lambda: _error_stream(),
            "provider failure",
        ),
        (
            lambda: _empty_done_stream(),
            "empty",
        ),
        (
            lambda: _tool_done_stream(),
            "tool calls",
        ),
        (
            lambda: _no_done_stream(),
            "without StreamDone",
        ),
    ],
)
async def test_summarizer_rejects_invalid_terminal_streams(stream_factory, error_text):
    async def stream(model, context, options):
        async for event in stream_factory():
            yield event

    with pytest.raises(SummarizationError, match=error_text):
        await summarize_with_stream(Model(), stream, SummarizationRequest("instruction", "content"))


@pytest.mark.asyncio
async def test_summarizer_uses_derived_model_only_when_request_max_tokens_is_provided():
    observed_models = []

    async def stream(model, context, options):
        observed_models.append(model)
        yield StreamDone(AssistantMessage([TextBlock("summary")]))

    original = Model("model", max_tokens=100)
    result = await summarize_with_stream(original, stream, SummarizationRequest("instruction", "content", max_tokens=17))

    assert result == "summary"
    assert original.max_tokens == 100
    assert observed_models == [Model("model", max_tokens=17)]


@pytest.mark.asyncio
async def test_generate_compaction_summary_builds_one_structured_request_for_normal_history():
    requests = []

    async def summarize(request):
        requests.append(request)
        return "S1"

    result = await generate_compaction_summary(
        historical_messages=[UserMessage("goal")],
        summarize=summarize,
        max_tokens=42,
    )

    assert result == "S1"
    assert len(requests) == 1
    assert requests[0].instruction == COMPACTION_SUMMARY_INSTRUCTION
    assert requests[0].max_tokens == 42
    assert "HISTORICAL CONVERSATION:\nUSER:\ngoal" in requests[0].content
    for heading in ("Goal", "Constraints", "Progress / Results", "Key Decisions", "Next Steps", "Critical Context"):
        assert f"## {heading}" in requests[0].instruction


@pytest.mark.asyncio
async def test_generate_compaction_summary_rejects_empty_injected_summary_result():
    async def summarize(request):
        return "   "

    with pytest.raises(SummarizationError, match="empty"):
        await generate_compaction_summary(
            historical_messages=[UserMessage("goal")],
            summarize=summarize,
        )


@pytest.mark.asyncio
async def test_generate_compaction_summary_merges_previous_summary_turn_prefix_and_trusted_context_once():
    requests = []

    async def summarize(request):
        requests.append(request)
        return "S2"

    result = await generate_compaction_summary(
        historical_messages=[UserMessage(f"{COMPACTION_SUMMARY_PREAMBLE}\n<SUMMARY>\nS1\n</SUMMARY>"), UserMessage("M3")],
        turn_prefix_messages=[UserMessage("prefix")],
        additional_context="trusted harness context",
        summarize=summarize,
    )

    assert result == "S2"
    assert len(requests) == 1
    assert "S1" in requests[0].content
    assert "HISTORICAL CONVERSATION:" in requests[0].content
    assert "CURRENT TURN PREFIX:" in requests[0].content
    assert "USER:\nprefix" in requests[0].content
    assert "TRUSTED HARNESS CONTEXT:\ntrusted harness context" in requests[0].content


async def _error_stream():
    yield StreamError("error", AssistantMessage([TextBlock("provider failure")], stop_reason="error"))


async def _empty_done_stream():
    yield StreamDone(AssistantMessage([TextBlock("   ")]))


async def _tool_done_stream():
    yield StreamDone(AssistantMessage([ToolCall("call-1", "tool", {})], stop_reason="tool_calls"))


async def _no_done_stream():
    yield Start(AssistantMessage([], partial=True))
