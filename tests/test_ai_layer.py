import json

import pytest
import httpx

from rova.ai.context import Context
from rova.ai._env import resolve_api_key
from rova.ai.events import ProviderFailure, StreamDone, StreamError
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.providers.openai_compatible import ContextOverflowError, HttpxStreamingHttpClient, OpenAICompatibleProvider, ProviderRequestError, estimate_provider_input_tokens, from_provider_response, to_provider_messages, to_provider_tools
from rova.ai.stream import stream_simple
from rova.ai.tools import Tool, validate_tool_arguments
from rova.agent_core.agent import Agent
from tests.tool_helpers import make_test_calc_tool
from rova.app.settings import AppSettings


class FakeHttpClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def stream_lines(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        if self.error:
            raise self.error
        if not isinstance(self.response, dict):
            yield f"data: {json.dumps(self.response)}"
            yield ""
            yield "data: [DONE]"
            yield ""
            return
        choices = self.response.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        delta = {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }
        yield "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": choice.get("finish_reason")} ]})
        yield ""
        yield "data: [DONE]"
        yield ""


def make_context(messages=None, tools=None):
    return Context("system instruction", messages or [UserMessage("hello")], tools or [])


def test_provider_input_estimate_uses_the_canonical_serialized_messages_and_tools():
    model = Model(provider="openai_compatible", model="test")
    base = make_context(messages=[UserMessage("hello")])
    with_tool = make_context(
        messages=[UserMessage("hello")],
        tools=[Tool("search", "Search documents", {"query": str})],
    )
    with_system_context = Context("system instruction\n\nMemory snapshot: x" * 10, list(base.messages), list(base.tools))

    assert estimate_provider_input_tokens(model, with_tool) > estimate_provider_input_tokens(model, base)
    assert estimate_provider_input_tokens(model, with_system_context) > estimate_provider_input_tokens(model, base)


@pytest.mark.asyncio
async def test_agent_applies_product_context_preparer_before_each_provider_step():
    prepared_contexts = []

    async def prepare(context):
        return Context(f"{context.system_prompt}\nprepared", list(context.messages), list(context.tools))

    async def stream(_model, context, _options):
        prepared_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model(), "base", [], stream, context_preparer=prepare)

    await agent.run([UserMessage("hello")])

    assert [context.system_prompt for context in prepared_contexts] == ["base\nprepared"]


@pytest.mark.asyncio
async def test_agent_retries_one_uncommitted_provider_step_after_typed_context_overflow():
    attempts = 0
    recovered_contexts = []

    async def recover(context):
        recovered_contexts.append(context)
        return Context("compacted", list(context.messages), list(context.tools))

    async def stream(_model, context, _options):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield StreamError(
                "error",
                AssistantMessage([TextBlock("overflow")], stop_reason="error"),
                failure=ProviderFailure("context_overflow", 400, "context_length_exceeded"),
            )
            return
        assert context.system_prompt == "compacted"
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model(), "base", [], stream, context_overflow_recovery=recover)

    messages = await agent.run([UserMessage("hello")])

    assert [message.text for message in messages] == ["done"]
    assert attempts == 2
    assert len(recovered_contexts) == 1


@pytest.mark.asyncio
async def test_agent_stops_after_one_typed_context_overflow_retry():
    attempts = 0

    async def recover(context):
        return context

    async def stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        yield StreamError(
            "error",
            AssistantMessage([TextBlock("overflow")], stop_reason="error"),
            failure=ProviderFailure("context_overflow", 400, "context_length_exceeded"),
        )

    agent = Agent(Model(), "base", [], stream, context_overflow_recovery=recover)
    messages = await agent.run([UserMessage("hello")])

    assert attempts == 2
    assert messages[-1].stop_reason == "error"
    assert agent.events[-1].termination_reason.value == "context_overflow"


def test_app_settings_builds_model_without_api_key():
    settings = AppSettings.from_env({"ROVA_PROVIDER": "openai_compatible", "ROVA_MODEL": "deepseek-chat", "ROVA_BASE_URL": "https://example.test/v1"})
    model = settings.to_model()
    assert model == Model(provider="openai_compatible", model="deepseek-chat", base_url="https://example.test/v1")
    assert not hasattr(model, "api_key")


def test_credential_mapping_only_exposes_routable_openai_compatible_provider():
    with pytest.raises(ValueError, match="No API key environment variable"):
        resolve_api_key("openai", {})
    assert resolve_api_key("openai_compatible", {"OPENAI_API_KEY": "test-key"}) == "test-key"


@pytest.mark.asyncio
async def test_stream_simple_routes_mock_model_without_network():
    events = [event async for event in stream_simple(Model(provider="mock"), make_context(), None)]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.text == "收到：hello"


@pytest.mark.asyncio
async def test_stream_simple_routes_openai_compatible_model(monkeypatch):
    seen = {}

    class StubProvider:
        def __init__(self, api_key, *, timeout_seconds):
            seen["api_key"] = api_key
            seen["timeout_seconds"] = timeout_seconds

        async def stream(self, model, context, options):
            seen["model"] = model
            yield StreamDone(AssistantMessage(content=[TextBlock("translated")]))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("rova.ai.stream.OpenAICompatibleProvider", StubProvider)
    model = Model(provider="openai_compatible", model="test-model", base_url="https://example.test/v1")
    events = [event async for event in stream_simple(model, make_context(), None)]
    assert events[-1].message.text == "translated"
    assert seen == {"api_key": "test-key", "timeout_seconds": 60.0, "model": model}


@pytest.mark.asyncio
async def test_stream_simple_passes_model_provider_timeout_to_provider(monkeypatch):
    seen = {}

    class StubProvider:
        def __init__(self, api_key, *, timeout_seconds):
            seen["api_key"] = api_key
            seen["timeout_seconds"] = timeout_seconds

        async def stream(self, model, context, options):
            yield StreamDone(AssistantMessage(content=[TextBlock("translated")]))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("rova.ai.stream.OpenAICompatibleProvider", StubProvider)
    model = Model(provider="openai_compatible", model="test-model", provider_timeout=120.0)

    events = [event async for event in stream_simple(model, make_context(), None)]

    assert events[-1].message.text == "translated"
    assert seen == {"api_key": "test-key", "timeout_seconds": 120.0}


@pytest.mark.asyncio
async def test_http_client_uses_configured_timeout(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_lines(self):
            yield "data: [DONE]"

    class Client:
        def __init__(self, *, timeout):
            seen["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("rova.ai.providers.openai_compatible.httpx.AsyncClient", Client)
    lines = [line async for line in HttpxStreamingHttpClient(timeout_seconds=12.5).stream_lines("https://example.test", {}, {})]

    assert lines == ["data: [DONE]"]
    assert seen["timeout"] == 12.5


@pytest.mark.asyncio
async def test_stream_simple_returns_stream_error_for_unsupported_provider():
    events = [event async for event in stream_simple(Model(provider="unsupported"), make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "Unsupported provider" in events[-1].error.text


@pytest.mark.asyncio
async def test_stream_simple_returns_stream_error_when_openai_credential_is_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("rova.ai._env.load_project_env", lambda env, dotenv_path=None: {})
    events = [event async for event in stream_simple(Model(provider="openai_compatible", model="test"), make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "Missing API credential: OPENAI_API_KEY" in events[-1].error.text


def test_openai_request_mapping_preserves_messages_tool_ids_and_schemas():
    call = ToolCall(id="call-123", name="calc", arguments={"expression": "2 * 3"})
    tool_result = ToolResultMessage("call-123", "calc", [TextBlock("6")])
    messages = [UserMessage("calculate"), AssistantMessage([TextBlock("working"), call], stop_reason="tool_calls"), tool_result]
    provider_messages = to_provider_messages("system instruction", messages)
    provider_tools = to_provider_tools([Tool("calc", "Evaluate arithmetic", {"expression": str})])

    assert provider_messages[0] == {"role": "system", "content": "system instruction"}
    assert provider_messages[1] == {"role": "user", "content": "calculate"}
    assert provider_messages[2]["role"] == "assistant"
    assert provider_messages[2]["content"] == "working"
    assert provider_messages[2]["tool_calls"] == [{"id": "call-123", "type": "function", "function": {"name": "calc", "arguments": '{"expression": "2 * 3"}'}}]
    assert provider_messages[3] == {"role": "tool", "tool_call_id": "call-123", "content": "6"}
    assert provider_tools == [{"type": "function", "function": {"name": "calc", "description": "Evaluate arithmetic", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}}}]


def test_tool_defaults_to_all_parameters_required_for_backward_compatibility():
    tool = Tool("pair", "Pair values", {"a": str, "b": int})
    with pytest.raises(ValueError, match="missing required argument: b"):
        validate_tool_arguments(tool, {"a": "value"})
    assert to_provider_tools([tool])[0]["function"]["parameters"]["required"] == ["a", "b"]


def test_tool_preserves_full_json_schema_without_legacy_conversion():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "filters": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    provider_tool = to_provider_tools([Tool("search", "Search", input_schema=schema)])[0]

    assert provider_tool["function"]["parameters"] == schema


def test_tool_required_field_allows_and_validates_optional_parameters():
    tool = Tool("pair", "Pair values", {"a": str, "b": int}, required=("a",))
    assert validate_tool_arguments(tool, {"a": "value"}) == {"a": "value"}
    with pytest.raises(ValueError, match="b must be int"):
        validate_tool_arguments(tool, {"a": "value", "b": "wrong"})
    assert to_provider_tools([tool])[0]["function"]["parameters"]["required"] == ["a"]


def test_tool_supports_an_empty_required_tuple():
    tool = Tool("optional", "Optional values", {"a": str}, required=())
    assert validate_tool_arguments(tool, {}) == {}
    assert to_provider_tools([tool])[0]["function"]["parameters"]["required"] == []


@pytest.mark.parametrize("required", [("missing",), ("a", "a")])
def test_tool_rejects_invalid_required_parameter_definitions(required):
    with pytest.raises(ValueError):
        Tool("invalid", "Invalid tool", {"a": str}, required=required)


def test_openai_response_mapping_returns_text_and_parsed_tool_call_arguments():
    response = {"choices": [{"finish_reason": "tool_calls", "message": {"content": "I will calculate.", "tool_calls": [{"id": "call-123", "type": "function", "function": {"name": "calc", "arguments": '{"expression": "123 * 456"}'}}]}}]}
    message = from_provider_response(response)
    assert message.text == "I will calculate."
    assert message.stop_reason == "tool_calls"
    assert message.tool_calls == [ToolCall(id="call-123", name="calc", arguments={"expression": "123 * 456"})]


@pytest.mark.asyncio
async def test_openai_translator_converts_malformed_tool_arguments_to_stream_error():
    response = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-123", "function": {"name": "calc", "arguments": "{bad json"}}]}}]}
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(response)).stream(Model(provider="openai_compatible", model="test", base_url="https://example.test/v1"), make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "Invalid provider response" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_converts_non_object_response_to_stream_error():
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient([])).stream(Model(provider="openai_compatible", model="test", base_url="https://example.test/v1"), make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "Invalid provider response" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_converts_http_failure_to_stream_error():
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(error=ProviderRequestError("network unavailable"))).stream(Model(provider="openai_compatible", model="test", base_url="https://example.test/v1"), make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "network unavailable" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_marks_only_typed_context_overflow_as_recoverable():
    events = [
        event
        async for event in OpenAICompatibleProvider(
            "test-key",
            FakeHttpClient(error=ContextOverflowError(status_code=400, code="context_length_exceeded")),
        ).stream(Model(provider="openai_compatible", model="test"), make_context(), None)
    ]

    assert isinstance(events[-1], StreamError)
    assert events[-1].failure is not None
    assert events[-1].failure.classification == "context_overflow"


@pytest.mark.asyncio
async def test_openai_translator_keeps_a_nonempty_type_when_provider_error_message_is_empty():
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(error=ProviderRequestError(""))).stream(Model(provider="openai_compatible", model="test"), make_context(), None)]

    assert isinstance(events[-1], StreamError)
    assert events[-1].error.text == "ProviderRequestError"


@pytest.mark.asyncio
async def test_openai_translator_includes_empty_timeout_cause_type():
    error = ProviderRequestError("")
    error.__cause__ = httpx.ReadTimeout("")

    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(error=error)).stream(Model(provider="openai_compatible", model="test"), make_context(), None)]

    assert isinstance(events[-1], StreamError)
    assert "ProviderRequestError" in events[-1].error.text
    assert "ReadTimeout" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_includes_http_status_without_response_body():
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(429, request=request)
    error = httpx.HTTPStatusError("", request=request, response=response)

    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(error=error)).stream(Model(provider="openai_compatible", model="test"), make_context(), None)]

    assert isinstance(events[-1], StreamError)
    assert "HTTPStatusError" in events[-1].error.text
    assert "HTTP status=429" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_redacts_key_authorization_and_url_query_from_error():
    error = ProviderRequestError("Authorization: Bearer top-secret request=https://provider.example/v1?token=query-secret")

    events = [event async for event in OpenAICompatibleProvider("top-secret", FakeHttpClient(error=error)).stream(Model(provider="openai_compatible", model="test"), make_context(), None)]

    assert isinstance(events[-1], StreamError)
    assert "top-secret" not in events[-1].error.text
    assert "query-secret" not in events[-1].error.text
    assert "[REDACTED]" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_keeps_unrecognized_programming_error_unwrapped():
    with pytest.raises(TypeError, match="programming bug"):
        [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(error=TypeError("programming bug"))).stream(Model(provider="openai_compatible", model="test"), make_context(), None)]


@pytest.mark.asyncio
async def test_openai_translator_converts_request_translation_value_error_to_stream_error():
    context = Context("", [object()])
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient()).stream(Model(provider="openai_compatible"), context, None)]
    assert isinstance(events[-1], StreamError)
    assert "Invalid provider request" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_converts_request_translation_type_error_to_stream_error():
    context = make_context(tools=[Tool("bad", "", {"value": []})])
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient()).stream(Model(provider="openai_compatible"), context, None)]
    assert isinstance(events[-1], StreamError)
    assert "Invalid provider request" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_converts_request_json_serialization_error_to_stream_error():
    model = Model(provider="openai_compatible", temperature=object())
    events = [event async for event in OpenAICompatibleProvider("test-key").stream(model, make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "serialization" in events[-1].error.text


@pytest.mark.asyncio
async def test_openai_translator_posts_translated_request_and_returns_text():
    client = FakeHttpClient({"choices": [{"finish_reason": "stop", "message": {"content": "hello back"}}]})
    model = Model(provider="openai_compatible", model="test", base_url="https://example.test/v1")
    events = [event async for event in OpenAICompatibleProvider("test-key", client).stream(model, make_context(), None)]
    url, headers, payload = client.calls[0]
    assert url == "https://example.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "test"
    assert payload["messages"][-1] == {"role": "user", "content": "hello"}
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.text == "hello back"
    assert events[-1].message.stop_reason == "stop"


@pytest.mark.asyncio
async def test_openai_translator_preserves_length_finish_reason():
    response = {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}]}
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(response)).stream(Model(provider="openai_compatible"), make_context(), None)]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.stop_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", [None, "content_filter"])
async def test_openai_translator_rejects_unknown_finish_reason(finish_reason):
    response = {"choices": [{"finish_reason": finish_reason, "message": {"content": "not accepted"}}]}
    events = [event async for event in OpenAICompatibleProvider("test-key", FakeHttpClient(response)).stream(Model(provider="openai_compatible"), make_context(), None)]
    assert isinstance(events[-1], StreamError)
    assert "Invalid provider response" in events[-1].error.text


@pytest.mark.asyncio
async def test_agent_loop_uses_stream_simple_without_agent_core_provider_dependency():
    agent = Agent(Model(provider="mock"), "", [make_test_calc_tool()], stream_simple)
    messages = await agent.run([UserMessage("calculate 2 * 3")])
    assert messages[-1].text == "2 * 3 = 6"
