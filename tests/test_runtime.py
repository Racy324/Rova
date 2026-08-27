import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from rova.ai.mock import MockProvider
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.agent_core.tools import AgentTool
from tests.tool_helpers import make_test_calc_tool


@pytest.mark.asyncio
async def test_regular_text_returns_final_assistant_message():
    agent = Agent(Model("mock"), "You are helpful.", [], MockProvider())
    messages = await agent.run([UserMessage("你好")])
    assert messages[-1].text == "收到：你好"
    assert messages[-1].stop_reason == "stop"


@pytest.mark.asyncio
async def test_calc_tool_result_is_appended_and_used_by_next_provider_turn():
    agent = Agent(Model("mock"), "You are helpful.", [make_test_calc_tool()], MockProvider())
    messages = await agent.run([UserMessage("帮我计算 123 * 456")])
    assert messages[-1].text == "123 * 456 = 56088"
    assert any(isinstance(m, ToolResultMessage) and m.content[0].text == "56088" for m in agent.messages)
    assert any(isinstance(m, ToolResultMessage) and m.content[0].text == "56088" for m in agent.last_context.messages)
    assert [event.type for event in agent.events] == ["agent_start", "turn_start", "message_start", "message_end", "tool_execution_start", "tool_execution_end", "turn_end", "turn_start", "message_start", "message_end", "turn_end", "agent_end"]


@pytest.mark.asyncio
async def test_each_provider_turn_receives_a_fresh_context_snapshot_from_agent_state():
    contexts = []

    async def record_context_stream(model, context, options):
        contexts.append(context)
        if len(contexts) == 1:
            yield StreamDone(AssistantMessage(content=[ToolCall(id="calc-1", name="calc", arguments={"expression": "2 * 3"})], stop_reason="tool_calls"))
        else:
            yield StreamDone(AssistantMessage(content=[], stop_reason="stop"))

    agent = Agent(Model("mock"), "", [make_test_calc_tool()], record_context_stream)
    await agent.run([UserMessage("calculate")])

    assert contexts[0] is not contexts[1]
    assert not any(isinstance(message, ToolResultMessage) for message in contexts[0].messages)
    assert any(isinstance(message, ToolResultMessage) and message.text == "6" for message in contexts[1].messages)
    assert any(isinstance(message, ToolResultMessage) and message.text == "6" for message in agent.messages)

    agent.last_context.messages.append(UserMessage("outside mutation"))
    assert not any(isinstance(message, UserMessage) and message.content == "outside mutation" for message in agent.messages)


async def unknown_tool_stream(model, context, options):
    yield StreamDone(AssistantMessage(content=[ToolCall(id="missing-1", name="missing", arguments={})], stop_reason="tool_calls"))


@pytest.mark.asyncio
async def test_unknown_tool_becomes_tool_error_instead_of_crashing():
    agent = Agent(Model("mock"), "", [], unknown_tool_stream, max_turns=2)
    await agent.run([UserMessage("call something")])
    result = next(m for m in agent.last_context.messages if isinstance(m, ToolResultMessage))
    assert result.is_error is True
    assert "Unknown tool: missing" in result.content[0].text


async def invalid_calc_stream(model, context, options):
    yield StreamDone(AssistantMessage(content=[ToolCall(id="bad-1", name="calc", arguments={"expression": 3})], stop_reason="tool_calls"))


@pytest.mark.asyncio
async def test_invalid_tool_arguments_become_tool_error():
    agent = Agent(Model("mock"), "", [make_test_calc_tool()], invalid_calc_stream, max_turns=2)
    await agent.run([UserMessage("bad calc")])
    result = next(m for m in agent.last_context.messages if isinstance(m, ToolResultMessage))
    assert result.is_error is True
    assert "expression must be str" in result.content[0].text


@pytest.mark.asyncio
async def test_max_turns_stops_provider_that_repeats_tool_calls():
    agent = Agent(Model("mock"), "", [make_test_calc_tool()], MockProvider(always_tool_call=True), max_turns=2)
    messages = await agent.run([UserMessage("loop")])
    assert messages[-1].stop_reason == "error"
    assert messages[-1].text == "Maximum turns (2) reached"


def test_registry_projects_only_schema_into_context():
    calc = make_test_calc_tool()
    agent = Agent(Model("mock"), "", [calc], MockProvider())
    context = agent.create_context_snapshot()
    assert context.tools == [calc.tool]
    assert not hasattr(context.tools[0], "execute")
    assert isinstance(calc, AgentTool)
