from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from .messages import AssistantMessage


@dataclass
class Start:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass
class TextDelta:
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass
class ToolCallDelta:
    index: int
    partial: AssistantMessage
    id_fragment: str = ""
    name_fragment: str = ""
    arguments_fragment: str = ""
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass
class StreamDone:
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass
class StreamError:
    reason: Literal["error", "aborted"]
    error: AssistantMessage
    type: Literal["error"] = "error"


AssistantMessageEvent = Union[Start, TextDelta, ToolCallDelta, StreamDone, StreamError]
