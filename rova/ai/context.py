from __future__ import annotations

from dataclasses import dataclass, field

from .messages import Message
from .tools import Tool


@dataclass
class Context:
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
