from __future__ import annotations

from typing import Any

from rova.ai.messages import AssistantMessage, Message, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage


class MessageSerializationError(ValueError):
    pass


def message_to_dict(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        if message.partial:
            raise MessageSerializationError("partial assistant messages are not durable")
        data = {
            "role": "assistant",
            "content": [_content_block_to_dict(block) for block in message.content],
            "stop_reason": message.stop_reason,
            "partial": False,
        }
        if message.usage is not None:
            data["usage"] = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.total_tokens,
            }
        return data
    if isinstance(message, ToolResultMessage):
        data = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": [_content_block_to_dict(block) for block in message.content],
            "is_error": message.is_error,
        }
        if message.metadata:
            data["metadata"] = message.metadata
        return data
    raise MessageSerializationError(f"unsupported message type: {type(message).__name__}")


def message_from_dict(data: object) -> Message:
    if not isinstance(data, dict):
        raise MessageSerializationError("message must be an object")
    role = data.get("role")
    if role == "user":
        return UserMessage(_string(data, "content"))
    if role == "assistant":
        partial = data.get("partial", False)
        if partial is not False:
            raise MessageSerializationError("partial assistant messages cannot be loaded")
        stop_reason = _string(data, "stop_reason")
        if stop_reason not in {"stop", "tool_calls", "length", "error", "aborted"}:
            raise MessageSerializationError(f"unsupported stop_reason: {stop_reason}")
        return AssistantMessage(
            _content_blocks_from_dict(data.get("content")),
            stop_reason=stop_reason,
            partial=False,
            usage=_usage(data.get("usage")),
        )
    if role == "tool":
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise MessageSerializationError("tool metadata must be an object")
        return ToolResultMessage(
            _string(data, "tool_call_id"),
            _string(data, "tool_name"),
            _text_blocks_from_dict(data.get("content")),
            is_error=_bool(data, "is_error"),
            metadata=metadata,
        )
    raise MessageSerializationError(f"unsupported message role: {role!r}")


def _content_block_to_dict(block: TextBlock | ToolCall) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCall):
        return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}
    raise MessageSerializationError(f"unsupported content block: {type(block).__name__}")


def _content_blocks_from_dict(value: object) -> list[TextBlock | ToolCall]:
    if not isinstance(value, list):
        raise MessageSerializationError("content must be a list")
    blocks: list[TextBlock | ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise MessageSerializationError("content block must be an object")
        if item.get("type") == "text":
            blocks.append(TextBlock(_string(item, "text")))
        elif item.get("type") == "toolCall":
            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                raise MessageSerializationError("toolCall arguments must be an object")
            blocks.append(ToolCall(_string(item, "id"), _string(item, "name"), arguments))
        else:
            raise MessageSerializationError(f"unsupported content block type: {item.get('type')!r}")
    return blocks


def _text_blocks_from_dict(value: object) -> list[TextBlock]:
    blocks = _content_blocks_from_dict(value)
    if any(not isinstance(block, TextBlock) for block in blocks):
        raise MessageSerializationError("tool result content may only contain text blocks")
    return blocks


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise MessageSerializationError(f"{key} must be a string")
    return value


def _bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise MessageSerializationError(f"{key} must be a boolean")
    return value


def _usage(value: object) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MessageSerializationError("usage must be an object")
    try:
        return Usage(
            input_tokens=value.get("input_tokens"),
            output_tokens=value.get("output_tokens"),
            total_tokens=value.get("total_tokens"),
        )
    except ValueError as error:
        raise MessageSerializationError(str(error)) from error
