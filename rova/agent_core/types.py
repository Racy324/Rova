from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from rova.ai.context import Context
from rova.ai.events import AssistantMessageEvent
from rova.ai.models import Model


class StreamFn(Protocol):
    def __call__(self, model: Model, context: Context, options: object | None = None) -> AsyncIterator[AssistantMessageEvent]: ...
