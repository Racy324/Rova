import pytest

import rova.agent_session.agent_session as agent_session_module
from rova.agent_session.agent_session import AgentSession, CompactionError, SessionPersistenceError
from rova.agent_session.compaction import (
    CompactionPolicy,
    ContextPressure,
    ConservativeTokenEstimator,
    should_compact,
    summary_max_tokens,
)
from rova.agent_session.context_builder import COMPACTION_SUMMARY_PREAMBLE
from rova.agent_session.session_store import JsonlSessionStore, SessionStoreError
from rova.agent_session.summarization import SummarizationError
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from tests.tool_helpers import make_test_calc_tool
from rova.app.settings import AppSettings


class CountingEstimator:
    def __init__(self, tokens_per_message=10):
        self.tokens_per_message = tokens_per_message
        self.calls = []

    def estimate_messages(self, messages):
        self.calls.append(tuple(messages))
        return len(messages) * self.tokens_per_message


def high_usage_message(text="final"):
    return AssistantMessage([TextBlock(text)], usage=Usage(80, 10, 90))


def test_compaction_policy_and_summary_budget_validation_are_independent():
    policy = CompactionPolicy(reserve_tokens=20, keep_recent_tokens=7)

    assert policy.reserve_tokens == 20
    assert policy.keep_recent_tokens == 7
    assert summary_max_tokens(Model(max_tokens=12), policy) == 12
    assert summary_max_tokens(Model(max_tokens=100), policy) == 16
    with pytest.raises(ValueError, match="reserve"):
        CompactionPolicy(0, 0)
    with pytest.raises(ValueError, match="keep_recent"):
        CompactionPolicy(1, -1)


def test_should_compact_uses_context_window_threshold_and_marks_pressure_source_explicitly():
    model = Model(context_window=100)
    policy = CompactionPolicy(reserve_tokens=10, keep_recent_tokens=20)

    assert ContextPressure(89, "reported").source == "reported"
    assert should_compact(pressure=ContextPressure(89, "reported"), model=model, policy=policy) is False
    assert should_compact(pressure=ContextPressure(90, "estimated"), model=model, policy=policy) is True
    assert should_compact(pressure=ContextPressure(1, "reported"), model=Model(), policy=policy) is False
    with pytest.raises(ValueError, match="context_window"):
        should_compact(pressure=ContextPressure(1, "reported"), model=Model(context_window=20), policy=CompactionPolicy(20, 1))


def test_agent_session_rejects_model_impossible_compaction_policy_at_binding_time(tmp_path):
    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    with pytest.raises(ValueError, match="context_window"):
        AgentSession.create(
            Agent(Model("mock", context_window=20), "", [], stream),
            session_root=tmp_path,
            compaction_policy=CompactionPolicy(10, 20),
        )

    with pytest.raises(ValueError, match="available context"):
        AgentSession.create(
            Agent(Model("mock", context_window=20), "", [], stream),
            session_root=tmp_path,
            compaction_policy=CompactionPolicy(10, 10),
        )


@pytest.mark.asyncio
async def test_automatic_compaction_uses_current_run_reported_usage_then_rebuilds_logical_runtime(tmp_path):
    summary_requests = []

    async def stream(model, context, options):
        yield StreamDone(high_usage_message())

    async def summarize(request):
        summary_requests.append(request)
        return "S1"

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=summarize,
    )

    result = await session.prompt("U1")
    durable = session._durable_session

    assert result[-1].text == "final"
    assert len(summary_requests) == 1
    assert any(type(entry).__name__ == "CompactionEntry" for entry in durable.entries)
    assert agent.messages[0].content.startswith(COMPACTION_SUMMARY_PREAMBLE)
    assert agent.messages[1] == AssistantMessage([TextBlock("final")], usage=Usage(80, 10, 90))
    assert session.persisted_message_count == len(agent.messages) == 2
    assert all(
        not (isinstance(message, UserMessage) and message.content.startswith(COMPACTION_SUMMARY_PREAMBLE))
        for message in durable.physical_messages
    )


@pytest.mark.asyncio
async def test_stale_historical_usage_is_not_reused_when_current_run_has_no_usage(tmp_path):
    durable = JsonlSessionStore(tmp_path).create()
    durable.append(UserMessage("old"))
    durable.append(AssistantMessage([TextBlock("old final")], usage=Usage(80, 10, 90)))

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("current final")]))

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.load(
        agent,
        durable.session_id,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(tokens_per_message=1),
        summary_fn=lambda request: _summary("unused"),
    )

    await session.prompt("current")

    assert not any(type(entry).__name__ == "CompactionEntry" for entry in session._durable_session.entries)
    assert session.last_maintenance_error is None


@pytest.mark.asyncio
async def test_estimated_pressure_fallback_and_keep_recent_budget_are_used_for_automatic_plan(tmp_path, monkeypatch):
    observed_budgets = []

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    async def summarize(request):
        return "S1"

    original_plan = agent_session_module.find_compaction_plan

    def recording_plan(projection, *, retained_token_budget, token_estimator):
        observed_budgets.append(retained_token_budget)
        return original_plan(projection, retained_token_budget=retained_token_budget, token_estimator=token_estimator)

    monkeypatch.setattr(agent_session_module, "find_compaction_plan", recording_plan)
    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 7),
        token_estimator=CountingEstimator(tokens_per_message=50),
        summary_fn=summarize,
    )

    await session.prompt("U1")

    assert observed_budgets == [7]
    assert any(type(entry).__name__ == "CompactionEntry" for entry in session._durable_session.entries)


@pytest.mark.asyncio
async def test_automatic_non_durable_planning_and_summary_failures_are_maintenance_warnings(tmp_path):
    async def stream(model, context, options):
        yield StreamDone(high_usage_message())

    async def failed_summary(request):
        raise SummarizationError("summary unavailable")

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=failed_summary,
    )

    result = await session.prompt("U1")

    assert result[-1].text == "final"
    assert isinstance(session.last_maintenance_error, SummarizationError)
    assert not any(type(entry).__name__ == "CompactionEntry" for entry in session._durable_session.entries)
    assert agent.messages == [UserMessage("U1"), high_usage_message()]


@pytest.mark.asyncio
async def test_context_window_none_policy_absent_and_no_useful_plan_disable_or_warn_without_durable_change(tmp_path):
    summary_calls = []

    async def stream(model, context, options):
        yield StreamDone(high_usage_message())

    async def summarize(request):
        summary_calls.append(request)
        return "unused"

    disabled_agent = Agent(Model("mock"), "", [], stream)
    disabled = AgentSession.create(
        disabled_agent,
        session_root=tmp_path / "disabled",
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=summarize,
    )
    await disabled.prompt("U1")
    assert summary_calls == []

    no_plan_agent = Agent(Model("mock", context_window=100), "", [], stream)
    no_plan = AgentSession.create(
        no_plan_agent,
        session_root=tmp_path / "no-plan",
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(tokens_per_message=1),
        summary_fn=summarize,
    )
    await no_plan.prompt("U1")
    assert isinstance(no_plan.last_maintenance_error, CompactionError)
    assert not any(type(entry).__name__ == "CompactionEntry" for entry in no_plan._durable_session.entries)


@pytest.mark.asyncio
async def test_automatic_compaction_runs_after_tool_loop_and_only_once_per_agent_run(tmp_path):
    summary_events = []

    async def stream(model, context, options):
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(high_usage_message("final"))
            return
        yield StreamDone(
            AssistantMessage(
                [ToolCall("calc-1", "calc", {"expression": "1 + 1"})],
                stop_reason="tool_calls",
                usage=Usage(70, 20, 90),
            )
        )

    agent = Agent(Model("mock", context_window=100), "", [make_test_calc_tool()], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=lambda request: _record_summary(summary_events, agent, request),
    )

    await session.prompt("calculate 1 + 1")

    assert len(summary_events) == 1
    event_types = [event.type for event in agent.events]
    assert event_types.index("agent_end") < event_types.index("compaction_started")
    assert any(isinstance(message, ToolResultMessage) for message in session._durable_session.physical_messages)


@pytest.mark.asyncio
async def test_repeated_automatic_compaction_replaces_prior_summary_and_next_provider_context_uses_it(tmp_path):
    contexts = []
    summary_requests = []

    async def stream(model, context, options):
        contexts.append(context)
        yield StreamDone(high_usage_message(f"A{len(contexts)}"))

    async def summarize(request):
        summary_requests.append(request)
        return f"S{len(summary_requests)}"

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=summarize,
    )

    await session.prompt("U1")
    await session.prompt("U2")

    assert len(summary_requests) == 2
    assert "S1" in summary_requests[1].content
    assert contexts[1].messages[0].content.startswith(COMPACTION_SUMMARY_PREAMBLE)
    assert "S2" in agent.messages[0].content
    assert "S1" not in agent.messages[0].content
    assert session.persisted_message_count == len(agent.messages)


@pytest.mark.asyncio
async def test_rebuild_failure_after_durable_compaction_faults_session_without_rollback(tmp_path, monkeypatch):
    async def stream(model, context, options):
        yield StreamDone(high_usage_message())

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=lambda request: _summary("S1"),
    )
    monkeypatch.setattr(
        agent_session_module,
        "build_session_messages",
        lambda entries: (_ for _ in ()).throw(RuntimeError("rebuild failed")),
    )

    with pytest.raises(SessionPersistenceError, match="compaction"):
        await session.prompt("U1")

    assert session.faulted is True
    assert any(type(entry).__name__ == "CompactionEntry" for entry in session._durable_session.entries)


@pytest.mark.asyncio
async def test_automatic_compaction_durable_append_failure_faults_session(tmp_path, monkeypatch):
    async def stream(model, context, options):
        yield StreamDone(high_usage_message())

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=CountingEstimator(),
        summary_fn=lambda request: _summary("S1"),
    )
    monkeypatch.setattr(session._durable_session.store, "append_compaction", lambda *_: (_ for _ in ()).throw(SessionStoreError("disk full")))

    with pytest.raises(SessionPersistenceError, match="compaction"):
        await session.prompt("U1")

    assert session.faulted is True


@pytest.mark.asyncio
async def test_manual_compact_at_uses_safe_split_and_rejects_prompt_active(tmp_path):
    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    agent = Agent(Model("mock"), "", [], stream)
    session = AgentSession.create(agent, session_root=tmp_path, summary_fn=lambda request: _summary("S1"))
    user_id = session._durable_session.append(UserMessage("U1"))
    assistant_id = session._durable_session.append(AssistantMessage([TextBlock("A1")]))
    agent.messages[:] = [UserMessage("U1"), AssistantMessage([TextBlock("A1")])]
    session.persisted_message_count = 2

    entry = await session.compact_at(assistant_id)

    assert entry.first_kept_entry_id == assistant_id
    assert agent.messages[0].content.startswith(COMPACTION_SUMMARY_PREAMBLE)
    assert agent.messages[1] == AssistantMessage([TextBlock("A1")])
    session._prompt_active = True
    with pytest.raises(CompactionError, match="prompt is active"):
        await session.compact_at(user_id)


@pytest.mark.asyncio
async def test_compaction_observer_failure_does_not_change_manual_compaction_semantics(tmp_path):
    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    agent = Agent(Model("mock"), "", [], stream)
    session = AgentSession.create(agent, session_root=tmp_path, summary_fn=lambda request: _summary("S1"))
    session._durable_session.append(UserMessage("U1"))
    assistant_id = session._durable_session.append(AssistantMessage([TextBlock("A1")]))
    agent.messages[:] = [UserMessage("U1"), AssistantMessage([TextBlock("A1")])]
    session.persisted_message_count = 2

    def broken_observer(event):
        raise RuntimeError("observer failed")

    session.subscribe_maintenance(broken_observer)
    entry = await session.compact_at(assistant_id)

    assert entry.first_kept_entry_id == assistant_id
    assert session.faulted is False


@pytest.mark.asyncio
async def test_manual_compact_at_rejects_unsafe_tool_boundary_and_supports_full_history(tmp_path):
    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    agent = Agent(Model("mock"), "", [], stream)
    session = AgentSession.create(agent, session_root=tmp_path, summary_fn=lambda request: _summary("S1"))
    session._durable_session.append(UserMessage("U1"))
    call_id = session._durable_session.append(
        AssistantMessage([ToolCall("call-1", "tool", {})], stop_reason="tool_calls")
    )
    result_id = session._durable_session.append(ToolResultMessage("call-1", "tool", [TextBlock("result")]))
    agent.messages[:] = session._durable_session.physical_messages
    session.persisted_message_count = len(agent.messages)

    with pytest.raises(CompactionError, match="protocol-safe"):
        await session.compact_at(result_id)

    entry = await session.compact_at(None)
    assert entry.first_kept_entry_id is None
    assert call_id in session._durable_session.by_id


def test_app_settings_wires_optional_compaction_policy_only_when_both_values_are_configured():
    enabled = AppSettings.from_env(
        {
            "ROVA_COMPACTION_RESERVE_TOKENS": "16",
            "ROVA_COMPACTION_KEEP_RECENT_TOKENS": "8",
        }
    )

    assert enabled.to_compaction_policy() == CompactionPolicy(16, 8)
    assert AppSettings.from_env({}).to_compaction_policy() is None
    with pytest.raises(ValueError, match="both"):
        AppSettings.from_env({"ROVA_COMPACTION_RESERVE_TOKENS": "16"}).to_compaction_policy()


async def _summary(text):
    return text


async def _record_summary(events, agent, request):
    events.append((agent.events[-1].type, request))
    return "S1"
