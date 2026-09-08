import asyncio
import json

import httpx
import pytest

import rova.ai.providers.openai_compatible as openai_compatible
from rova.ai.context import Context
from rova.ai.events import Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from rova.ai.models import Model
from rova.ai.providers.openai_compatible import OpenAICompatibleProvider, ProviderRequestError
from rova.agent_core.agent import Agent
from tests.tool_helpers import make_test_calc_tool


class FakeStreamingHttpClient:
    def __init__(self, lines, error=None):
        self.lines = lines
        self.error = error
        self.calls = []

    async def stream_lines(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        for line in self.lines:
            yield line
        if self.error:
            raise self.error


class TrackingStreamingHttpClient:
    def __init__(self, streams, *, require_previous_stream_closed=False):
        self.streams = streams
        self.require_previous_stream_closed = require_previous_stream_closed
        self.opened: list[int] = []
        self.closed: list[int] = []

    def stream_lines(self, url, headers, payload):
        index = len(self.opened)
        if self.require_previous_stream_closed and index and (index - 1) not in self.closed:
            raise AssertionError(f"stream {index - 1} was not closed before stream {index} opened")
        self.opened.append(index)
        return self._stream_lines(index)

    async def _stream_lines(self, index):
        try:
            for line in self.streams[index]:
                yield line
        finally:
            self.closed.append(index)


class BlockingStreamingHttpClient:
    def __init__(self):
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = 0

    def stream_lines(self, url, headers, payload):
        return self._stream_lines()

    async def _stream_lines(self):
        try:
            yield sse_chunk({"content": "partial"})
            yield ""
            self.blocked.set()
            await self.release.wait()
        finally:
            self.closed += 1


class TrackingSseStream:
    def __init__(self, stream):
        self.stream = stream
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.stream.__anext__()

    async def aclose(self):
        self.close_count += 1
        await self.stream.aclose()


def context():
    return Context("", [UserMessage("hello")])


def sse_chunk(delta, finish_reason=None):
    return f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': finish_reason}]})}"


@pytest.mark.asyncio
async def test_streaming_translator_yields_incremental_text_and_final_message():
    client = FakeStreamingHttpClient([
        sse_chunk({"content": chr(0x4F60)}), "",
        sse_chunk({"content": chr(0x597D)}), "",
        sse_chunk({}, "stop"), "",
        "data: [DONE]", "",
    ])
    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]
    expected_text = chr(0x4F60) + chr(0x597D)
    assert [type(event) for event in events] == [Start, TextDelta, TextDelta, StreamDone]
    assert [event.partial.text for event in events[1:3]] == [chr(0x4F60), expected_text]
    assert events[-1].message.text == expected_text
    assert events[-1].message.partial is False


@pytest.mark.asyncio
async def test_streaming_translator_preserves_usage_from_terminal_usage_only_chunk():
    client = FakeStreamingHttpClient([
        sse_chunk({"content": "done"}), "",
        sse_chunk({}, "stop"), "",
        'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}', "",
        "data: [DONE]", "",
    ])

    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]

    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.usage == Usage(10, 3, 13)


@pytest.mark.asyncio
async def test_streaming_translator_requests_terminal_usage_chunk():
    client = FakeStreamingHttpClient([
        sse_chunk({"content": "done"}, "stop"), "",
        "data: [DONE]", "",
    ])

    _ = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]

    assert client.calls[0][2]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_streaming_translator_closes_transport_before_exposing_text_done():
    client = TrackingStreamingHttpClient([[
        sse_chunk({"content": "done"}), "",
        sse_chunk({}, "stop"), "",
        "data: [DONE]", "",
    ]])
    provider = OpenAICompatibleProvider("key", client)

    async for event in provider.stream(Model(provider="openai_compatible"), context()):
        if isinstance(event, StreamDone):
            assert client.closed == [0]


@pytest.mark.asyncio
async def test_streaming_translator_closes_transport_before_exposing_tool_call_done():
    client = TrackingStreamingHttpClient([[
        sse_chunk({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "calc", "arguments": '{"expression":"1+1"}'}}]}), "",
        sse_chunk({}, "tool_calls"), "",
        "data: [DONE]", "",
    ]])
    provider = OpenAICompatibleProvider("key", client)

    async for event in provider.stream(Model(provider="openai_compatible"), context()):
        if isinstance(event, StreamDone):
            assert event.message.stop_reason == "tool_calls"
            assert client.closed == [0]


@pytest.mark.asyncio
async def test_streaming_translator_accepts_dashscope_reasoning_content_and_empty_terminal_content():
    client = FakeStreamingHttpClient([
        sse_chunk({"content": None, "reasoning_content": "thinking"}), "",
        sse_chunk({"content": "OK", "reasoning_content": None}), "",
        sse_chunk({"content": "", "reasoning_content": None}, "stop"), "",
        "data: [DONE]", "",
    ])
    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]

    assert not any(isinstance(event, StreamError) for event in events)
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.text == "OK"
    assert events[-1].message.stop_reason == "stop"


@pytest.mark.asyncio
async def test_streaming_translator_accumulates_multiple_tool_calls_by_index():
    client = FakeStreamingHttpClient([
        sse_chunk({"tool_calls": [{"index": 1, "id": "call-2", "function": {"name": "cal"}}]}), "",
        sse_chunk({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "calc", "arguments": '{"expression":"1+1"}'}}]}), "",
        sse_chunk({"tool_calls": [{"index": 1, "function": {"name": "c", "arguments": '{"expression":"2'}}]}), "",
        sse_chunk({"tool_calls": [{"index": 1, "function": {"arguments": ' * 3"}'}}]}), "",
        sse_chunk({}, "tool_calls"), "",
        "data: [DONE]", "",
    ])
    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]
    deltas = [event for event in events if isinstance(event, ToolCallDelta)]
    final = events[-1].message
    assert [delta.index for delta in deltas] == [1, 0, 1, 1]
    assert final.stop_reason == "tool_calls"
    assert final.tool_calls == [
        ToolCall(id="call-1", name="calc", arguments={"expression": "1+1"}),
        ToolCall(id="call-2", name="calc", arguments={"expression": "2 * 3"}),
    ]


@pytest.mark.asyncio
async def test_streaming_translator_accepts_dashscope_tool_call_continuation_fragments():
    client = FakeStreamingHttpClient([
        sse_chunk({"content": None, "reasoning_content": "thinking"}), "",
        sse_chunk({"tool_calls": [{"index": 0, "id": "call_real", "type": "function", "function": {"name": "get_test_value", "arguments": ""}}]}), "",
        sse_chunk({"tool_calls": [{"index": 0, "id": "", "type": "function", "function": {"name": "", "arguments": "{}"}}]}), "",
        sse_chunk({}, "tool_calls"), "",
        "data: [DONE]", "",
    ])
    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]

    assert not any(isinstance(event, StreamError) for event in events)
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.tool_calls == [
        ToolCall(id="call_real", name="get_test_value", arguments={})
    ]
    assert events[-1].message.stop_reason == "tool_calls"


@pytest.mark.asyncio
async def test_streaming_translator_accepts_null_tool_call_id_continuation_fragment():
    client = FakeStreamingHttpClient([
        sse_chunk({"tool_calls": [{"index": 0, "id": "call_real", "function": {"name": "calc", "arguments": ""}}]}), "",
        sse_chunk({"tool_calls": [{"index": 0, "id": None, "function": {"name": "", "arguments": "{}"}}]}), "",
        sse_chunk({}, "tool_calls"), "",
        "data: [DONE]", "",
    ])

    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]

    assert not any(isinstance(event, StreamError) for event in events)
    assert events[-1].message.tool_calls == [ToolCall(id="call_real", name="calc", arguments={})]


@pytest.mark.asyncio
async def test_streaming_translator_accepts_null_tool_call_function_continuation_fragments():
    client = FakeStreamingHttpClient([
        sse_chunk({"tool_calls": [{"index": 0, "id": "call_real", "function": {"name": "calc", "arguments": "{}"}}]}), "",
        sse_chunk({"tool_calls": [{"index": 0, "function": {"name": None, "arguments": None}}]}), "",
        sse_chunk({}, "tool_calls"), "",
        "data: [DONE]", "",
    ])

    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]

    assert not any(isinstance(event, StreamError) for event in events)
    assert events[-1].message.tool_calls == [ToolCall(id="call_real", name="calc", arguments={})]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason, expected", [("stop", "stop"), ("length", "length")])
async def test_streaming_translator_uses_existing_finish_reason_mapping(finish_reason, expected):
    client = FakeStreamingHttpClient([sse_chunk({"content": "done"}, finish_reason), "", "data: [DONE]", ""])
    events = [event async for event in OpenAICompatibleProvider("key", client).stream(Model(provider="openai_compatible"), context())]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.stop_reason == expected


@pytest.mark.asyncio
async def test_streaming_translator_converts_invalid_sse_and_network_failures_to_stream_error():
    malformed = FakeStreamingHttpClient(["data: {bad json", ""])
    malformed_events = [event async for event in OpenAICompatibleProvider("key", malformed).stream(Model(provider="openai_compatible"), context())]
    failed = FakeStreamingHttpClient([], ProviderRequestError("connection lost"))
    failed_events = [event async for event in OpenAICompatibleProvider("key", failed).stream(Model(provider="openai_compatible"), context())]
    assert isinstance(malformed_events[-1], StreamError)
    assert isinstance(failed_events[-1], StreamError)


@pytest.mark.asyncio
async def test_streaming_translator_closes_transport_before_exposing_parse_error():
    client = TrackingStreamingHttpClient([["data: {not-json", ""]])
    provider = OpenAICompatibleProvider("key", client)

    async for event in provider.stream(Model(provider="openai_compatible"), context()):
        if isinstance(event, StreamError):
            assert client.closed == [0]


@pytest.mark.asyncio
async def test_streaming_translator_explicit_close_closes_sse_and_transport_once(monkeypatch):
    client = TrackingStreamingHttpClient([[
        sse_chunk({"content": "partial"}), "",
    ]])
    sse_streams = []
    original_sse_data = openai_compatible._sse_data

    def tracking_sse_data(lines):
        sse_stream = TrackingSseStream(original_sse_data(lines))
        sse_streams.append(sse_stream)
        return sse_stream

    monkeypatch.setattr(openai_compatible, "_sse_data", tracking_sse_data)
    provider_iterator = OpenAICompatibleProvider("key", client).stream(
        Model(provider="openai_compatible"),
        context(),
    )

    assert isinstance(await provider_iterator.__anext__(), Start)
    await provider_iterator.aclose()

    assert sse_streams[0].close_count == 1
    assert client.closed == [0]


@pytest.mark.asyncio
async def test_streaming_translator_cancellation_closes_transport_once():
    client = BlockingStreamingHttpClient()
    provider_iterator = OpenAICompatibleProvider("key", client).stream(
        Model(provider="openai_compatible"),
        context(),
    )

    assert isinstance(await provider_iterator.__anext__(), Start)
    assert isinstance(await provider_iterator.__anext__(), TextDelta)
    pending = asyncio.create_task(provider_iterator.__anext__())
    await client.blocked.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await asyncio.sleep(0)

    assert client.closed == 1


@pytest.mark.asyncio
async def test_streaming_translator_does_not_leak_httpx_errors():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    status_error = httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
    events = [event async for event in OpenAICompatibleProvider("key", FakeStreamingHttpClient([], status_error)).stream(Model(provider="openai_compatible"), context())]
    assert isinstance(events[-1], StreamError)
    assert "503" in events[-1].error.text


def _http_status_error(status_code: int, *, provider_code: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    payload = {"error": {"code": provider_code}} if provider_code is not None else None
    response = httpx.Response(status_code, request=request, json=payload)
    return httpx.HTTPStatusError(str(status_code), request=request, response=response)


async def _terminal_stream_error(error: BaseException | None = None, *, lines=None) -> StreamError:
    client = FakeStreamingHttpClient([] if lines is None else lines, error)
    events = [
        event
        async for event in OpenAICompatibleProvider("key", client).stream(
            Model(provider="openai_compatible"),
            context(),
        )
    ]
    assert isinstance(events[-1], StreamError)
    return events[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 529, 500, 503])
async def test_streaming_translator_classifies_retryable_http_statuses_as_transient(status_code):
    event = await _terminal_stream_error(_http_status_error(status_code))

    assert event.failure is not None
    assert event.failure.classification == "transient"
    assert event.failure.category == "transient"
    assert event.failure.retryable is True
    assert event.failure.status_code == status_code


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadTimeout("timed out"), httpx.ConnectError("connection reset")])
async def test_streaming_translator_classifies_timeout_and_connection_as_transient(error):
    event = await _terminal_stream_error(error)

    assert event.failure is not None
    assert event.failure.classification == "transient"
    assert event.failure.retryable is True


@pytest.mark.asyncio
async def test_streaming_translator_classifies_incomplete_stream_as_transient():
    event = await _terminal_stream_error()

    assert event.failure is not None
    assert event.failure.classification == "transient"
    assert event.failure.code == "stream_interrupted"
    assert event.failure.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 400, 422])
async def test_streaming_translator_classifies_auth_and_invalid_requests_as_permanent(status_code):
    event = await _terminal_stream_error(_http_status_error(status_code))

    assert event.failure is not None
    assert event.failure.classification == "permanent"
    assert event.failure.retryable is False
    assert event.failure.status_code == status_code


@pytest.mark.asyncio
async def test_streaming_translator_preserves_typed_context_overflow():
    event = await _terminal_stream_error(
        _http_status_error(400, provider_code="context_length_exceeded")
    )

    assert event.failure is not None
    assert event.failure.classification == "context_overflow"
    assert event.failure.retryable is False
    assert event.failure.code == "context_length_exceeded"


@pytest.mark.asyncio
async def test_streaming_translator_classifies_invalid_sse_as_permanent():
    event = await _terminal_stream_error(lines=["data: {bad json", ""])

    assert event.failure is not None
    assert event.failure.classification == "permanent"
    assert event.failure.retryable is False


@pytest.mark.asyncio
async def test_streaming_translator_classifies_unknown_stream_error_as_unclassified():
    class UnknownStreamError(Exception):
        pass

    event = await _terminal_stream_error(UnknownStreamError("unexpected"))

    assert event.failure is not None
    assert event.failure.classification == "unclassified"
    assert event.failure.retryable is False


@pytest.mark.asyncio
async def test_streaming_translator_propagates_cancellation_without_provider_failure():
    with pytest.raises(asyncio.CancelledError):
        await _terminal_stream_error(asyncio.CancelledError())


async def text_stream(model, provider_context, options):
    yield Start(AssistantMessage([], partial=True))
    yield TextDelta(chr(0x4F60), AssistantMessage([TextBlock(chr(0x4F60))], partial=True))
    yield TextDelta(chr(0x597D), AssistantMessage([TextBlock(chr(0x4F60) + chr(0x597D))], partial=True))
    yield StreamDone(AssistantMessage([TextBlock(chr(0x4F60) + chr(0x597D))]))


@pytest.mark.asyncio
async def test_agent_bridges_text_stream_without_committing_partial_messages():
    agent = Agent(Model(provider="mock"), "", [], text_stream)
    observed_history_at_end = []

    def listener(event):
        if event.type == "message_end":
            observed_history_at_end.append(agent.messages[-1])

    agent.subscribe(listener)
    result = await agent.run([UserMessage("hello")])
    expected_text = chr(0x4F60) + chr(0x597D)
    assert [event.type for event in agent.events] == ["agent_start", "turn_start", "message_start", "message_update", "message_update", "message_end", "turn_end", "agent_end"]
    assert result[-1].text == expected_text
    assert [message.text for message in agent.messages if isinstance(message, AssistantMessage)] == [expected_text]
    assert observed_history_at_end == [result[-1]]
    assert len(agent.last_context.messages) == 1


async def delta_before_start_stream(model, provider_context, options):
    yield TextDelta("first", AssistantMessage([TextBlock("first")], partial=True))
    yield StreamDone(AssistantMessage([TextBlock("first")]))


@pytest.mark.asyncio
async def test_agent_synthesizes_one_start_for_delta_before_start():
    agent = Agent(Model(provider="mock"), "", [], delta_before_start_stream)
    await agent.run([UserMessage("hello")])
    assert [event.type for event in agent.events].count("message_start") == 1
    assert [event.type for event in agent.events].count("message_update") == 1
    assert [event.type for event in agent.events].count("message_end") == 1


async def startless_done_stream(model, provider_context, options):
    yield StreamDone(AssistantMessage([TextBlock("final")]))


@pytest.mark.asyncio
async def test_agent_commits_startless_terminal_before_message_end():
    agent = Agent(Model(provider="mock"), "", [], startless_done_stream)
    observed_history_at_end = []
    agent.subscribe(lambda event: observed_history_at_end.append(agent.messages[-1]) if event.type == "message_end" else None)
    await agent.run([UserMessage("hello")])
    assert [event.type for event in agent.events].count("message_start") == 1
    assert observed_history_at_end[-1].text == "final"


async def streamed_tool_call(model, provider_context, options):
    if any(isinstance(message, ToolResultMessage) for message in provider_context.messages):
        yield Start(AssistantMessage([], partial=True))
        yield TextDelta("2 * 3 = 6", AssistantMessage([TextBlock("2 * 3 = 6")], partial=True))
        yield StreamDone(AssistantMessage([TextBlock("2 * 3 = 6")]))
        return
    yield Start(AssistantMessage([], partial=True))
    yield ToolCallDelta(0, AssistantMessage([], partial=True), id_fragment="call-1", name_fragment="calc", arguments_fragment='{"expression":"2 * 3"}')
    yield StreamDone(AssistantMessage([ToolCall("call-1", "calc", {"expression": "2 * 3"})], stop_reason="tool_calls"))


@pytest.mark.asyncio
async def test_agent_executes_streamed_tool_only_after_final_tool_call():
    agent = Agent(Model(provider="mock"), "", [make_test_calc_tool()], streamed_tool_call)
    result = await agent.run([UserMessage("calculate")])
    event_types = [event.type for event in agent.events]
    assert event_types.index("message_end") < event_types.index("tool_execution_start")
    assert result[-1].text == "2 * 3 = 6"


@pytest.mark.asyncio
async def test_agent_opens_next_provider_stream_only_after_tool_call_transport_closes():
    client = TrackingStreamingHttpClient(
        [
            [
                sse_chunk({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "calc", "arguments": '{"expression":"1+1"}'}}]}), "",
                sse_chunk({}, "tool_calls"), "",
                "data: [DONE]", "",
            ],
            [
                sse_chunk({"content": "2"}), "",
                sse_chunk({}, "stop"), "",
                "data: [DONE]", "",
            ],
        ],
        require_previous_stream_closed=True,
    )

    async def provider_stream(model, provider_context, options):
        async for event in OpenAICompatibleProvider("key", client).stream(model, provider_context, options):
            yield event

    result = await Agent(Model(provider="openai_compatible"), "", [make_test_calc_tool()], provider_stream).run([UserMessage("calculate")])

    assert result[-1].text == "2"
    assert client.opened == [0, 1]
    assert client.closed == [0, 1]


async def failed_stream(model, provider_context, options):
    yield Start(AssistantMessage([], partial=True))
    yield TextDelta("partial", AssistantMessage([TextBlock("partial")], partial=True))
    yield StreamError("error", AssistantMessage([TextBlock("failed")], partial=True, stop_reason="error"))


@pytest.mark.asyncio
async def test_agent_commits_only_final_error_after_stream_failure():
    agent = Agent(Model(provider="mock"), "", [], failed_stream)
    await agent.run([UserMessage("hello")])
    assistants = [message for message in agent.messages if isinstance(message, AssistantMessage)]
    assert len(assistants) == 1
    assert assistants[0].text == "failed"
    assert assistants[0].partial is False
    assert assistants[0].stop_reason == "error"
