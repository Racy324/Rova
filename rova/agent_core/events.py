from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rova.ai.events import AssistantMessageEvent
from rova.ai.messages import Message


class AgentTerminationReason(Enum):
    FINAL_RESPONSE = "final_response"
    MAX_TURNS = "max_turns"
    PROVIDER_ERROR = "provider_error"
    ABORTED = "aborted"


@dataclass
class AgentEvent:
    type: str
    message: Message | None = None
    assistant_message_event: AssistantMessageEvent | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    batch_id: str | None = None
    call_index: int | None = None
    batch_mode: str | None = None
    execution_mode: str | None = None
    execution_state: str | None = None
    outcome: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    is_error: bool = False
    termination_reason: AgentTerminationReason | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
