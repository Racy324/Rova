from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from ..context import Context
from ..events import AssistantMessageEvent, Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from ..messages import AssistantMessage, Message, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from ..models import Model
from ..tools import Tool


class ProviderRequestError(Exception):
    """An external HTTP failure represented by the AI event contract."""


class StreamingHttpClient(Protocol):
    def stream_lines(self, url: str, headers: dict[str, str], payload: dict) -> AsyncIterator[str]: ...


class HttpxStreamingHttpClient:
    def stream_lines(self, url: str, headers: dict[str, str], payload: dict) -> AsyncIterator[str]:
        return self._stream_lines(url, headers, payload)

    async def _stream_lines(self, url: str, headers: dict[str, str], payload: dict) -> AsyncIterator[str]:
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            raise ProviderRequestError(f"Provider request serialization failed: {error}") from error
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("POST", url, content=body, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yield line
        except httpx.HTTPError as error:
            raise ProviderRequestError(str(error)) from error


class OpenAICompatibleProvider:
    """Translate Rova's internal AI contract to streaming Chat Completions SSE."""

    def __init__(self, api_key: str, http_client: StreamingHttpClient | None = None) -> None:
        self._api_key = api_key
        self._http_client = http_client or HttpxStreamingHttpClient()

    async def stream(self, model: Model, context: Context, options: object | None = None) -> AsyncIterator[AssistantMessageEvent]:
        try:
            payload = to_provider_request(model, context)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            yield _stream_error(f"Invalid provider request: {error}")
            return
        try:
            json.dumps(payload)
        except (TypeError, ValueError, OverflowError) as error:
            yield _stream_error(f"Provider request serialization failed: {error}")
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
                        terminal_event = _stream_error(f"Invalid provider response: {error}")
                    break
                try:
                    for event in accumulator.consume(data):
                        yield event
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    terminal_event = _stream_error(f"Invalid provider response: {error}")
                    break
        except (ProviderRequestError, httpx.HTTPError) as error:
            terminal_event = _stream_error(_provider_error_summary(error, self._api_key))
        finally:
            await _close_if_supported(sse_stream)

        if terminal_event is None:
            terminal_event = _stream_error("Provider stream ended before [DONE]")
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
                "parameters": {
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
            raise ProviderRequestError(f"Provider SSE protocol error: invalid SSE line: {line!r}")
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


def _stream_error(text: str) -> StreamError:
    return StreamError("error", AssistantMessage(content=[TextBlock(text)], stop_reason="error"))


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
