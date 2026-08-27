import pytest

from rova.agent_session.agent_session import (
    AgentSession,
    CompactionError,
    CompactionHeadroomWarning,
    PreRunContextTooLarge,
)
from rova.agent_session.compaction import CompactionPolicy
from rova.agent_session.session_store import JsonlSessionStore
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from tests.tool_helpers import make_test_calc_tool


class FixedEstimator:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    def estimate_messages(self, messages):
        self.calls.append(tuple(messages))
        return 0 if not messages else self.tokens


def high_usage(text="final"):
    return AssistantMessage([TextBlock(text)], usage=Usage(70, 20, 90))


@pytest.mark.asyncio
async def test_post_compaction_pressure_below_threshold_has_no_maintenance_diagnostic(tmp_path):
    async def stream(model, context, options):
        yield StreamDone(high_usage())

    agent = Agent(Model("mock", context_window=100, max_tokens=40), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 10),
        token_estimator=FixedEstimator(40),
        summary_fn=lambda request: _summary("S1"),
    )

    await session.prompt("U1")

    assert session.last_maintenance_error is None


@pytest.mark.asyncio
async def test_post_compaction_high_pressure_is_diagnostic_without_recursive_attempt(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        yield StreamDone(high_usage())

    async def summarize(request):
        summary_calls.append(request)
        return "S1"

    agent = Agent(Model("mock", context_window=100, max_tokens=40), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 10),
        token_estimator=FixedEstimator(80),
        summary_fn=summarize,
    )

    await session.prompt("U1")

    assert isinstance(session.last_maintenance_error, CompactionHeadroomWarning)
    assert len(summary_calls) == 1
    assert sum(type(entry).__name__ == "CompactionEntry" for entry in session._durable_session.entries) == 1


@pytest.mark.asyncio
async def test_summary_input_that_cannot_fit_with_output_budget_fails_before_summary_provider(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    async def summarize(request):
        summary_calls.append(request)
        return "must not be called"

    agent = Agent(Model("mock", context_window=50, max_tokens=40), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 5),
        token_estimator=FixedEstimator(35),
        summary_fn=summarize,
    )
    session._durable_session.append(UserMessage("historical"))
    agent.messages[:] = [UserMessage("historical")]
    session.persisted_message_count = 1

    with pytest.raises(CompactionError, match="summary input"):
        await session.compact_at(None)

    assert summary_calls == []
    assert [type(entry).__name__ for entry in session._durable_session.entries] == ["MessageEntry"]


@pytest.mark.asyncio
async def test_summary_input_that_fits_invokes_provider_and_persists_compaction(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    async def summarize(request):
        summary_calls.append(request)
        return "S1"

    agent = Agent(Model("mock", context_window=50, max_tokens=40), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 5),
        token_estimator=FixedEstimator(30),
        summary_fn=summarize,
    )
    session._durable_session.append(UserMessage("historical"))
    agent.messages[:] = [UserMessage("historical")]
    session.persisted_message_count = 1

    await session.compact_at(None)

    assert len(summary_calls) == 1
    assert type(session._durable_session.entries[-1]).__name__ == "CompactionEntry"


@pytest.mark.asyncio
async def test_summary_input_at_the_reserved_output_boundary_fits(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    async def summarize(request):
        summary_calls.append(request)
        return "S1"

    agent = Agent(Model("mock", context_window=50, max_tokens=40), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 5),
        token_estimator=FixedEstimator(34),
        summary_fn=summarize,
    )
    session._durable_session.append(UserMessage("historical"))
    agent.messages[:] = [UserMessage("historical")]
    session.persisted_message_count = 1

    await session.compact_at(None)

    assert len(summary_calls) == 1


@pytest.mark.asyncio
async def test_unbounded_manual_summary_is_rejected_before_provider_when_context_is_known(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    async def summarize(request):
        summary_calls.append(request)
        return "must not be called"

    agent = Agent(Model("mock", context_window=50), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        token_estimator=FixedEstimator(49),
        summary_fn=summarize,
    )
    session._durable_session.append(UserMessage("historical"))
    agent.messages[:] = [UserMessage("historical")]
    session.persisted_message_count = 1

    with pytest.raises(CompactionError, match="output budget"):
        await session.compact_at(None)

    assert summary_calls == []


@pytest.mark.asyncio
async def test_pre_run_obviously_oversized_user_context_is_not_sent_to_provider(tmp_path):
    provider_calls = []

    async def stream(model, context, options):
        provider_calls.append(context)
        yield StreamDone(AssistantMessage([TextBlock("must not be called")]))

    agent = Agent(Model("mock", context_window=50), "", [], stream)
    session = AgentSession.create(agent, session_root=tmp_path, token_estimator=FixedEstimator(51))

    with pytest.raises(PreRunContextTooLarge, match="before provider"):
        await session.prompt("oversized")

    assert provider_calls == []
    assert session._durable_session.physical_messages == [UserMessage("oversized")]
    assert session.faulted is False


@pytest.mark.asyncio
async def test_latest_current_run_usage_wins_over_earlier_tool_call_usage(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([TextBlock("final")], usage=Usage(30, 40, 70)))
            return
        yield StreamDone(
            AssistantMessage(
                [ToolCall("calc-1", "calc", {"expression": "1 + 1"})],
                stop_reason="tool_calls",
                usage=Usage(70, 20, 90),
            )
        )

    async def summarize(request):
        summary_calls.append(request)
        return "must not be called"

    agent = Agent(Model("mock", context_window=100, max_tokens=40), "", [make_test_calc_tool()], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 10),
        token_estimator=FixedEstimator(40),
        summary_fn=summarize,
    )

    await session.prompt("calculate")

    assert summary_calls == []
    assert session.last_maintenance_error is None


@pytest.mark.asyncio
async def test_maintenance_error_clears_at_the_start_of_the_next_prompt(tmp_path):
    run_count = 0

    async def stream(model, context, options):
        nonlocal run_count
        run_count += 1
        yield StreamDone(high_usage() if run_count == 1 else AssistantMessage([TextBlock("low")], usage=Usage(5, 5, 10)))

    async def summarize(request):
        raise RuntimeError("summary unavailable")

    agent = Agent(Model("mock", context_window=100, max_tokens=40), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 10),
        token_estimator=FixedEstimator(40),
        summary_fn=summarize,
    )

    await session.prompt("first")
    assert isinstance(session.last_maintenance_error, RuntimeError)
    await session.prompt("second")
    assert session.last_maintenance_error is None


@pytest.mark.asyncio
async def test_manual_compaction_on_selected_branch_preserves_raw_sibling_branch(tmp_path):
    durable = JsonlSessionStore(tmp_path).create()
    e1 = durable.append(UserMessage("M1"))
    e2 = durable.append(UserMessage("M2"))
    e3 = durable.append(UserMessage("sibling raw"))

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    agent = Agent(Model("mock"), "", [], stream)
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path, summary_fn=lambda request: _summary("S1"))
    session.branch(e2)
    compaction = await session.compact_at(None)
    session.close()

    compacted_agent = Agent(Model("mock"), "", [], stream)
    compacted = AgentSession.load(compacted_agent, durable.session_id, session_root=tmp_path, leaf_id=compaction.entry_id)
    assert compacted_agent.messages[0].content.endswith("S1\n</SUMMARY>")
    with pytest.raises(CompactionError, match="visible raw provenance"):
        await compacted.compact_at(e3)
    compacted.close()

    sibling_agent = Agent(Model("mock"), "", [], stream)
    sibling = AgentSession.load(sibling_agent, durable.session_id, session_root=tmp_path, leaf_id=e3)
    assert sibling_agent.messages == [UserMessage("M1"), UserMessage("M2"), UserMessage("sibling raw")]
    sibling.close()
    assert e1 in durable.by_id


@pytest.mark.asyncio
async def test_repeated_automatic_compaction_reloads_latest_summary_and_next_context_only(tmp_path):
    contexts = []
    summary_count = 0

    async def stream(model, context, options):
        contexts.append(context)
        yield StreamDone(high_usage(f"A{len(contexts)}"))

    async def summarize(request):
        nonlocal summary_count
        summary_count += 1
        return f"S{summary_count}"

    model = Model("mock", context_window=100, max_tokens=40)
    agent = Agent(model, "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 10),
        token_estimator=FixedEstimator(40),
        summary_fn=summarize,
    )
    await session.prompt("U1")
    await session.prompt("U2")
    before_reload = list(agent.messages)
    session_id = session.session_id
    session.close()

    restored_agent = Agent(model, "", [], stream)
    restored = AgentSession.load(
        restored_agent,
        session_id,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(20, 10),
        token_estimator=FixedEstimator(40),
        summary_fn=summarize,
    )

    assert restored_agent.messages == before_reload
    assert "S2" in restored_agent.messages[0].content
    assert "S1" not in restored_agent.messages[0].content
    await restored.prompt("U3")
    next_context = contexts[-1]
    assert "S2" in next_context.messages[0].content
    assert "S1" not in next_context.messages[0].content
    assert UserMessage("U3") in next_context.messages
    restored.close()


async def _summary(text):
    return text
