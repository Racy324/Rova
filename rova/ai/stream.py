from __future__ import annotations

from collections.abc import AsyncIterator

from ._env import resolve_api_key
from .context import Context
from .events import AssistantMessageEvent, StreamError
from .messages import AssistantMessage, TextBlock
from .mock import MockProvider
from .models import Model
from .providers.openai_compatible import OpenAICompatibleProvider


async def stream_simple(model: Model, context: Context, options: object | None = None) -> AsyncIterator[AssistantMessageEvent]:
    if model.provider == "mock":
        async for event in MockProvider()(model, context, options):
            yield event
        return
    if model.provider == "openai_compatible":
        try:
            api_key = resolve_api_key(model.provider)
        except ValueError as error:
            yield _stream_error(str(error))
            return
        async for event in OpenAICompatibleProvider(api_key).stream(model, context, options):
            yield event
        return
    yield _stream_error(f"Unsupported provider: {model.provider}")


def _stream_error(text: str) -> StreamError:
    return StreamError("error", AssistantMessage(content=[TextBlock(text)], stop_reason="error"))
