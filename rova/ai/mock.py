from __future__ import annotations

import re
from collections.abc import AsyncIterator

from .context import Context
from .events import AssistantMessageEvent, StreamDone
from .messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from .models import Model


class MockProvider:
    """A deterministic, network-free provider for the Phase 1 probe and tests."""

    def __init__(self, *, always_tool_call: bool = False, usage: Usage | None = None) -> None:
        self.always_tool_call = always_tool_call
        self.usage = usage

    async def __call__(self, model: Model, context: Context, options: object | None = None) -> AsyncIterator[AssistantMessageEvent]:
        if self.always_tool_call:
            yield self._calc_call("1 + 1")
            return
        latest = context.messages[-1]
        if isinstance(latest, ToolResultMessage):
            expression = self._latest_expression(context)
            yield StreamDone(AssistantMessage(content=[TextBlock(f"{expression} = {latest.text}")], usage=self.usage))
            return
        user = next((message for message in reversed(context.messages) if isinstance(message, UserMessage)), None)
        if user is not None:
            expression = self._expression_from(user.content)
            if expression:
                yield self._calc_call(expression)
                return
            yield StreamDone(AssistantMessage(content=[TextBlock(f"收到：{user.content}")], usage=self.usage))
            return
        yield StreamDone(AssistantMessage(content=[TextBlock("收到")], usage=self.usage))

    @staticmethod
    def _expression_from(text: str) -> str | None:
        match = re.search(r"(\d+\s*[-+*/]\s*\d+)", text)
        return match.group(1) if match else None

    @staticmethod
    def _latest_expression(context: Context) -> str:
        for message in reversed(context.messages):
            if isinstance(message, AssistantMessage) and message.tool_calls:
                return str(message.tool_calls[-1].arguments["expression"])
        return "calculation"

    def _calc_call(self, expression: str) -> StreamDone:
        return StreamDone(
            AssistantMessage(
                content=[ToolCall(id="calc-1", name="calc", arguments={"expression": expression})],
                stop_reason="tool_calls",
                usage=self.usage,
            )
        )
