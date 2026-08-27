import pytest

from rova.agent_session.serialization import message_from_dict, message_to_dict
from rova.ai.context import Context
from rova.ai.messages import AssistantMessage, TextBlock, Usage, UserMessage
from rova.ai.mock import MockProvider
from rova.ai.models import Model
from rova.ai.providers.openai_compatible import from_provider_response
from rova.app.settings import AppSettings


def test_usage_requires_non_negative_integer_counts():
    assert Usage(input_tokens=10, output_tokens=3, total_tokens=13) == Usage(10, 3, 13)

    with pytest.raises(ValueError, match="input_tokens"):
        Usage(input_tokens=-1, output_tokens=0, total_tokens=0)

    with pytest.raises(ValueError, match="output_tokens"):
        Usage(input_tokens=0, output_tokens=True, total_tokens=0)


def test_assistant_message_usage_is_optional_and_preserved():
    assert AssistantMessage([TextBlock("answer")]).usage is None

    usage = Usage(10, 3, 13)
    assert AssistantMessage([TextBlock("answer")], usage=usage).usage == usage


def test_openai_non_stream_response_maps_provider_usage():
    message = from_provider_response(
        {
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
        }
    )

    assert message.usage == Usage(10, 3, 13)


def test_openai_non_stream_response_allows_missing_usage():
    message = from_provider_response(
        {"choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}
    )

    assert message.usage is None


def test_assistant_usage_round_trips_and_old_json_defaults_to_none():
    message = AssistantMessage([TextBlock("answer")], usage=Usage(10, 3, 13))
    assert message_from_dict(message_to_dict(message)) == message

    restored = message_from_dict(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "old answer"}],
            "stop_reason": "stop",
            "partial": False,
        }
    )
    assert isinstance(restored, AssistantMessage)
    assert restored.usage is None


def test_model_context_window_is_optional_positive_and_independent_from_max_tokens():
    assert Model().context_window is None
    assert Model(max_tokens=8_192, context_window=128_000).context_window == 128_000

    with pytest.raises(ValueError, match="context_window"):
        Model(context_window=0)


def test_app_settings_wires_context_window_to_model():
    settings = AppSettings.from_env({"ROVA_CONTEXT_WINDOW": "128000"})
    assert settings.to_model().context_window == 128_000


@pytest.mark.asyncio
async def test_mock_provider_can_emit_deterministic_usage():
    provider = MockProvider(usage=Usage(10, 3, 13))
    events = [event async for event in provider(Model(), Context("", [UserMessage("hello")]))]

    assert events[-1].message.usage == Usage(10, 3, 13)
