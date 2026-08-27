from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    type: Literal["toolCall"] = "toolCall"


@dataclass
class UserMessage:
    content: str
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass
class AssistantMessage:
    content: list[TextBlock | ToolCall] = field(default_factory=list)
    stop_reason: Literal["stop", "tool_calls", "length", "error", "aborted"] = "stop"
    partial: bool = False
    role: Literal["assistant"] = "assistant"
    usage: Usage | None = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [block for block in self.content if isinstance(block, ToolCall)]

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[TextBlock]
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    role: Literal["tool"] = "tool"

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content)


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]
