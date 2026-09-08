from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from ..context import Context
from ..events import AssistantMessageEvent, ProviderFailure, Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from ..messages import AssistantMessage, Message, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from ..models import Model
from ..tools import Tool


class ProviderRequestError(Exception):
    """An external HTTP failure represented by the AI event contract."""

    def __init__(self, message: str, *, failure: ProviderFailure | None = None) -> None:
        super().__init__(message)
        self.failure = failure


class ContextOverflowError(ProviderRequestError):
    """A Provider adapter verified a context-length rejection structurally."""

    def __init__(self, *, status_code: int, code: str) -> None:
        super().__init__(f"Provider rejected request context (HTTP {status_code}, code={code})")
        self.status_code = status_code
        self.code = code


class StreamingHttpClient(Protocol):
    def stream_lines(self, url: str, headers: dict[str, str], payload: dict) -> AsyncIterator[str]: ...


class HttpxStreamingHttpClient:
    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        self._timeout_seconds = timeout_seconds

    def stream_lines(self, url: str, headers: dict[str, str], payload: dict) -> AsyncIterator[str]:
        return self._stream_lines(url, headers, payload)

    async def _stream_lines(self, url: str, headers: dict[str, str], payload: dict) -> AsyncIterator[str]:
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            raise ProviderRequestError(
                f"Provider request serialization failed: {error}",
                failure=ProviderFailure("permanent", code="invalid_request", message="Provider request serialization failed"),
            ) from error
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                async with client.stream("POST", url, content=body, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yield line
        except httpx.HTTPStatusError as error:
            failure = _provider_failure_for_exception(error)
            if failure.classification == "context_overflow":
                raise ContextOverflowError(status_code=error.response.status_code, code=failure.code or "context_overflow") from error
            raise ProviderRequestError(str(error), failure=failure) from error
        except httpx.HTTPError as error:
            raise ProviderRequestError(str(error), failure=_provider_failure_for_exception(error)) from error


class OpenAICompatibleProvider:
    """Translate Rova's internal AI contract to streaming Chat Completions SSE."""

    def __init__(
        self,
        api_key: str,
        http_client: StreamingHttpClient | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client or HttpxStreamingHttpClient(timeout_seconds=timeout_seconds)

    async def stream(self, model: Model, context: Context, options: object | None = None) -> AsyncIterator[AssistantMessageEvent]:
        try:
            payload = to_provider_request(model, context)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            yield _stream_error(
                f"Invalid provider request: {error}",
                failure=ProviderFailure("permanent", code="invalid_request", message="Invalid provider request"),
            )
            return
        try:
            json.dumps(payload)
        except (TypeError, ValueError, OverflowError) as error:
            yield _stream_error(
                f"Provider request serialization failed: {error}",
                failure=ProviderFailure("permanent", code="invalid_request", message="Provider request serialization failed"),
            )
            return

        accumulator = _StreamingAccumulator()
        terminal_event: StreamDone | StreamError | None = None
        sse_stream = _sse_data(self._http_client.stream_lines(
            _chat_completions_url(model),
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            payload,
        ))
        try:
            async for data in sse_stream:
                if data == "[DONE]":
                    try:
                        terminal_event = StreamDone(accumulator.finalize())
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                        terminal_event = _stream_error(
                            f"Invalid provider response: {error}",
                            failure=ProviderFailure("permanent", code="invalid_response", message="Invalid provider response"),
                        )
                    break
                try:
                    for event in accumulator.consume(data):
                        yield event
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    terminal_event = _stream_error(
                        f"Invalid provider response: {error}",
                        failure=ProviderFailure("permanent", code="invalid_response", message="Invalid provider response"),
                    )
                    break
        except (ProviderRequestError, httpx.HTTPError) as error:
            terminal_event = _stream_error(
                _provider_error_summary(error, self._api_key),
                failure=_provider_failure_for_exception(error),
            )
        except (TypeError, AttributeError, AssertionError):
            raise
        except Exception as error:
            terminal_event = _stream_error(
                _provider_error_summary(error, self._api_key),
                failure=ProviderFailure("unclassified", code="stream_error", message=type(error).__name__),
            )
        finally:
            await _close_if_supported(sse_stream)

        if terminal_event is None:
            terminal_event = _stream_error(
                "Provider stream ended before [DONE]",
                failure=ProviderFailure("transient", code="stream_interrupted", message="Provider stream ended before completion"),
            )
        yield terminal_event


def to_provider_request(model: Model, context: Context) -> dict:
    payload = {
        "model": model.model,
        "messages": to_provider_messages(context.system_prompt, context.messages),
        "stream": True,
        # OpenAI-compatible streaming APIs otherwise commonly omit the final
        # usage-only chunk. Providers that do not implement this extension can
        # still respond normally; usage remains optional in Rova's contract.
        "stream_options": {"include_usage": True},
    }
    tools = to_provider_tools(context.tools)
    if tools:
        payload["tools"] = tools
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    if model.max_tokens is not None:
        payload["max_tokens"] = model.max_tokens
    return payload


def estimate_provider_input_tokens(model: Model, context: Context) -> int:
    """Deterministically estimate provider input from its canonical request form.

    This is deliberately an estimate, not a tokenizer claim or Provider usage
    substitute. It serializes exactly the messages and tool schemas that this
    adapter will send, including role and function-call framing.
    """
    payload = to_provider_request(model, context)
    input_payload = {"messages": payload["messages"]}
    if "tools" in payload:
        input_payload["tools"] = payload["tools"]
    encoded = json.dumps(input_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return max(1, (len(encoded) + 3) // 4)


def to_provider_messages(system_prompt: str, messages: Sequence[Message]) -> list[dict]:
    provider_messages: list[dict] = []
    if system_prompt:
        provider_messages.append({"role": "system", "content": system_prompt})
    for message in messages:
        if isinstance(message, UserMessage):
            provider_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            provider_messages.append(_assistant_to_provider_message(message))
        elif isinstance(message, ToolResultMessage):
            provider_messages.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text})
        else:
            raise ValueError(f"Unsupported Rova message: {type(message).__name__}")
    return provider_messages


def _assistant_to_provider_message(message: AssistantMessage) -> dict:
    text = message.text
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
        }
        for call in message.tool_calls
    ]
    provider_message = {"role": "assistant", "content": text or None}
    if tool_calls:
        provider_message["tool_calls"] = tool_calls
    return provider_message


def to_provider_tools(tools: Sequence[Tool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema if tool.input_schema is not None else {
                    "type": "object",
                    "properties": {
                        name: {"type": _json_schema_type(value_type)}
                        for name, value_type in tool.parameters.items()
                    },
                    "required": list(tool.parameters) if tool.required is None else list(tool.required),
                    "additionalProperties": False,
                },
            },
        }
        for tool in tools
    ]


def from_provider_response(response: dict) -> AssistantMessage:
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    choice = choices[0]
    provider_message = choice.get("message")
    if not isinstance(provider_message, dict):
        raise ValueError("response has no assistant message")
    content = provider_message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("assistant content must be text or null")
    blocks = [TextBlock(content)] if content else []
    provider_tool_calls = provider_message.get("tool_calls", [])
    if not isinstance(provider_tool_calls, list):
        raise ValueError("assistant tool_calls must be a list")
    tool_calls = [_tool_call_from_provider(item) for item in provider_tool_calls]
    blocks.extend(tool_calls)
    if not blocks:
        raise ValueError("response has no convertible assistant content")
    return AssistantMessage(
        content=blocks,
        stop_reason=_stop_reason(choice.get("finish_reason"), bool(tool_calls)),
        usage=_usage_from_provider(response.get("usage")),
    )


async def _sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data_lines: list[str] = []
    try:
        async for line in lines:
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
                continue
            if line.startswith(("event:", "id:", "retry:")):
                continue
            raise ProviderRequestError(
                f"Provider SSE protocol error: invalid SSE line: {line!r}",
                failure=ProviderFailure("permanent", code="invalid_sse", message="Invalid provider SSE response"),
            )
        if data_lines:
            yield "\n".join(data_lines)
    finally:
        await _close_if_supported(lines)


@dataclass
class _ToolCallFragments:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamingAccumulator:
    text_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCallFragments] = field(default_factory=dict)
    finish_reason: str | None = None
    started: bool = False
    usage: Usage | None = None

    def consume(self, data: str) -> list[AssistantMessageEvent]:
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("stream chunk must be an object")
        if "usage" in chunk:
            usage = _usage_from_provider(chunk["usage"])
            if usage is not None:
                self.usage = usage
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise ValueError("stream chunk has no choices")
        if not choices:
            if "usage" in chunk:
                return []
            raise ValueError("stream chunk has no choices")
        if not isinstance(choices[0], dict):
            raise ValueError("stream chunk has no choices")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ValueError("stream choice has no delta")
        events: list[AssistantMessageEvent] = []
        if not self.started:
            events.append(Start(self._partial_message()))
            self.started = True
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError("stream content must be text or null")
            if content:
                self.text_parts.append(content)
                events.append(TextDelta(content, self._partial_message()))
        tool_call_deltas = delta.get("tool_calls")
        if tool_call_deltas is not None:
            if not isinstance(tool_call_deltas, list):
                raise ValueError("stream tool_calls must be a list")
            for raw_delta in tool_call_deltas:
                event = self._consume_tool_call_delta(raw_delta)
                events.append(event)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str):
                raise ValueError("stream finish_reason must be text or null")
            self.finish_reason = finish_reason
        return events

    def finalize(self) -> AssistantMessage:
        blocks: list[TextBlock | ToolCall] = [TextBlock("".join(self.text_parts))] if self.text_parts else []
        calls = [self._tool_call(index, fragments) for index, fragments in sorted(self.tool_calls.items())]
        blocks.extend(calls)
        if not blocks:
            raise ValueError("stream has no convertible assistant content")
        return AssistantMessage(
            content=blocks,
            stop_reason=_stop_reason(self.finish_reason, bool(calls)),
            usage=self.usage,
        )

    def _consume_tool_call_delta(self, raw_delta: object) -> ToolCallDelta:
        if not isinstance(raw_delta, dict):
            raise ValueError("stream tool call delta must be an object")
        index = raw_delta.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("stream tool call delta is missing integer index")
        fragments = self.tool_calls.setdefault(index, _ToolCallFragments())
        id_fragment = raw_delta.get("id", "")
        if not isinstance(id_fragment, str):
            raise ValueError("stream tool call id fragment must be text")
        function = raw_delta.get("function", {})
        if not isinstance(function, dict):
            raise ValueError("stream tool call function must be an object")
        name_fragment = function.get("name", "")
        arguments_fragment = function.get("arguments", "")
        if not isinstance(name_fragment, str) or not isinstance(arguments_fragment, str):
            raise ValueError("stream tool call fragments must be text")
        fragments.call_id += id_fragment
        fragments.name += name_fragment
        fragments.arguments += arguments_fragment
        return ToolCallDelta(index, self._partial_message(), id_fragment, name_fragment, arguments_fragment)

    def _partial_message(self) -> AssistantMessage:
        content = [TextBlock("".join(self.text_parts))] if self.text_parts else []
        return AssistantMessage(content=content, partial=True)

    @staticmethod
    def _tool_call(index: int, fragments: _ToolCallFragments) -> ToolCall:
        if not fragments.call_id or not fragments.name or not fragments.arguments:
            raise ValueError(f"stream tool call {index} is incomplete")
        try:
            arguments = json.loads(fragments.arguments)
        except json.JSONDecodeError as error:
            raise ValueError("tool call arguments are not valid JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must decode to an object")
        return ToolCall(fragments.call_id, fragments.name, arguments)


def _tool_call_from_provider(value: object) -> ToolCall:
    if not isinstance(value, dict):
        raise ValueError("tool call must be an object")
    call_id = value.get("id")
    function = value.get("function")
    if not isinstance(call_id, str) or not isinstance(function, dict):
        raise ValueError("tool call is missing id or function")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, str):
        raise ValueError("tool call is missing name or JSON arguments")
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValueError("tool call arguments are not valid JSON") from error
    if not isinstance(parsed_arguments, dict):
        raise ValueError("tool call arguments must decode to an object")
    return ToolCall(id=call_id, name=name, arguments=parsed_arguments)


def _usage_from_provider(value: object) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("usage must be an object or null")
    return Usage(
        input_tokens=_usage_count(value, "prompt_tokens"),
        output_tokens=_usage_count(value, "completion_tokens"),
        total_tokens=_usage_count(value, "total_tokens"),
    )


def _usage_count(value: dict, key: str) -> int:
    count = value.get(key)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"usage.{key} must be a non-negative integer")
    return count


def _stop_reason(finish_reason: object, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    if finish_reason == "stop":
        return "stop"
    if finish_reason == "length":
        return "length"
    raise ValueError(f"unsupported finish_reason: {finish_reason!r}")


def _chat_completions_url(model: Model) -> str:
    base_url = model.base_url or "https://api.openai.com/v1"
    return f"{base_url.rstrip('/')}/chat/completions"


def _json_schema_type(value_type: type) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(value_type, "string")


_RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}


def _provider_failure_for_exception(error: BaseException) -> ProviderFailure:
    """Map only structurally known transport failures to stable adapter facts."""
    if isinstance(error, ContextOverflowError):
        return ProviderFailure(
            "context_overflow",
            error.status_code,
            error.code,
            "Provider rejected the request context",
        )
    if isinstance(error, ProviderRequestError) and error.failure is not None:
        return error.failure
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        overflow_code = _structured_context_overflow_code(error.response)
        if overflow_code is not None:
            return ProviderFailure(
                "context_overflow",
                status_code,
                overflow_code,
                "Provider rejected the request context",
            )
        code = _structured_provider_error_code(error.response) or f"http_{status_code}"
        if status_code in _RETRYABLE_HTTP_STATUS_CODES:
            return ProviderFailure("transient", status_code, code, f"Provider returned HTTP {status_code}")
        if 400 <= status_code < 500:
            return ProviderFailure("permanent", status_code, code, f"Provider returned HTTP {status_code}")
        return ProviderFailure("unclassified", status_code, code, f"Provider returned HTTP {status_code}")
    if isinstance(error, httpx.TimeoutException):
        return ProviderFailure("transient", code="timeout", message="Provider request timed out")
    if isinstance(error, (httpx.NetworkError, httpx.ProtocolError)):
        return ProviderFailure("transient", code="network_interruption", message="Provider network stream interrupted")
    if isinstance(error, ProviderRequestError):
        return ProviderFailure("unclassified", code="provider_request_error", message="Unclassified provider request failure")
    return ProviderFailure("unclassified", code="stream_error", message=type(error).__name__)


def _structured_context_overflow_code(response: httpx.Response) -> str | None:
    """Recognize only documented structured codes, never free-form error text."""
    if response.status_code not in {400, 413}:
        return None
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if code in {"context_length_exceeded", "context_window_exceeded"}:
        return str(code)
    return None


def _structured_provider_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, (str, int)) and not isinstance(code, bool):
        return str(code)
    return None


def _stream_error(text: str, *, failure: ProviderFailure | None = None) -> StreamError:
    return StreamError("error", AssistantMessage(content=[TextBlock(text)], stop_reason="error"), failure=failure)


def _provider_error_summary(error: BaseException, api_key: str) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(parts) < 4:
        seen.add(id(current))
        parts.append(_describe_provider_error(current, api_key))
        current = current.__cause__ or current.__context__
    return "; caused by ".join(parts)[:800]


def _describe_provider_error(error: BaseException, api_key: str) -> str:
    details: list[str] = []
    message = _redact_provider_text(str(error), api_key).strip()
    if message:
        details.append(message)
    if isinstance(error, httpx.HTTPStatusError):
        details.append(f"HTTP status={error.response.status_code}")
    return f"{type(error).__name__}: {', '.join(details)}" if details else type(error).__name__


def _redact_provider_text(value: str, api_key: str) -> str:
    if api_key:
        value = value.replace(api_key, "[REDACTED]")
    value = re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)\S+", r"\1[REDACTED]", value)
    value = re.sub(r"([?&][A-Za-z0-9_.-]+=)[^&#\s]+", r"\1[REDACTED]", value)
    return value


async def _close_if_supported(iterator: AsyncIterator[object]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()
