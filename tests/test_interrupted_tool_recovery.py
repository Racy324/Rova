from __future__ import annotations

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.agent_core.agent import Agent
from rova.agent_core.events import AgentEvent
from rova.agent_core.tools import AgentTool, AgentToolResult
from rova.agent_session.agent_session import AgentSession, SessionPersistenceError, SessionRecoveryError
from rova.agent_session.execution_journal import ToolExecutionJournal
from rova.agent_session.session_store import JsonlSessionStore, SessionStoreError


def _tool_stream():
    async def stream(_model, context, _options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"))
            return
        yield StreamDone(AssistantMessage([TextBlock("done")]))
    return stream


@pytest.mark.asyncio
async def test_started_record_is_durable_before_executor_launch(tmp_path) -> None:
    observed_states: list[str] = []
    session: AgentSession

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        records = session._execution_journal.load()
        observed_states[:] = [record.state for record in records]
        assert observed_states == ["started"]
        assert records[0].assistant_entry_id is not None
        return AgentToolResult([TextBlock("ok")])

    agent = Agent(Model("mock"), "", [AgentTool(Tool("sample", "sample", {}), execute)], _tool_stream())
    session = AgentSession.create(agent, session_root=tmp_path)

    await session.prompt("run")

    assert observed_states == ["started"]


@pytest.mark.asyncio
async def test_completed_receipt_is_durable_before_tool_result_commit(tmp_path, monkeypatch) -> None:
    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        return AgentToolResult([TextBlock("canonical result")], {"outcome": "success"})

    agent = Agent(Model("mock"), "", [AgentTool(Tool("sample", "sample", {}), execute)], _tool_stream())
    session = AgentSession.create(agent, session_root=tmp_path)
    original_append = session._durable_session.store.append_message

    def fail_tool_result(session_id, parent_id, message):
        if isinstance(message, ToolResultMessage):
            raise SessionStoreError("simulated crash before tool result commit")
        return original_append(session_id, parent_id, message)

    monkeypatch.setattr(session._durable_session.store, "append_message", fail_tool_result)

    with pytest.raises(SessionPersistenceError, match="failed to persist"):
        await session.prompt("run")

    records = session._execution_journal.load()
    completed = [record for record in records if record.state == "completed"]
    assert len(completed) == 1
    assert completed[0].receipt is not None
    assert completed[0].receipt.content == "canonical result"
    assert JsonlSessionStore(tmp_path).load(session.session_id).messages == [
        UserMessage("run"),
        AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"),
    ]


@pytest.mark.asyncio
async def test_load_reconciles_unknown_and_completed_calls_without_reexecuting_tools(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    assistant_entry_id = durable.append(
        AssistantMessage(
            [ToolCall("a", "sample", {}), ToolCall("b", "sample", {})],
            stop_reason="tool_calls",
        )
    )
    journal = ToolExecutionJournal(tmp_path, durable.session_id)
    journal.append(
        AgentEvent(
            "tool_execution_state", tool_call_id="a", tool_name="sample", batch_id="batch",
            call_index=0, batch_mode="parallel", execution_mode="parallel", execution_state="started",
        ),
        assistant_entry_id=assistant_entry_id,
    )
    journal.append(
        AgentEvent(
            "tool_execution_state", tool_call_id="b", tool_name="sample", batch_id="batch",
            call_index=1, batch_mode="parallel", execution_mode="parallel", execution_state="completed",
            outcome="success", result="restored result", is_error=False, metadata={"outcome": "success"},
        ),
        assistant_entry_id=assistant_entry_id,
    )
    executions = 0

    async def execute(_call_id: str, _arguments: dict) -> AgentToolResult:
        nonlocal executions
        executions += 1
        return AgentToolResult([TextBlock("must not run")])

    agent = Agent(Model("mock"), "", [AgentTool(Tool("sample", "sample", {}), execute)], _tool_stream())
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    results = [message for message in agent.messages if isinstance(message, ToolResultMessage)]
    assert [(result.tool_call_id, result.text) for result in results] == [
        ("a", "The previous tool execution was interrupted before a durable result was recorded. Its side effects may have occurred. Inspect the current state before taking a new action."),
        ("b", "restored result"),
    ]
    assert results[0].metadata["outcome"] == "execution_interrupted"
    assert results[0].metadata["side_effects_unknown"] is True
    assert results[1].metadata == {"outcome": "success"}
    assert executions == 0
    assert session.recovery_report.recovered_count == 2


def test_legacy_completed_without_receipt_is_recovered_as_side_effects_unknown(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    durable.append(AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"))
    journal = ToolExecutionJournal(tmp_path, durable.session_id)
    journal.path.write_text(
        '{"batch_id":"batch","tool_call_id":"call-1","call_index":0,"tool_name":"sample","batch_mode":"parallel","execution_mode":"parallel","state":"completed","outcome":"success","timestamp":"2026-09-07T00:00:00+00:00"}\n',
        encoding="utf-8",
    )

    agent = Agent(Model("mock"), "", [], _tool_stream())
    AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    result = next(message for message in agent.messages if isinstance(message, ToolResultMessage))
    assert result.metadata["outcome"] == "execution_interrupted"
    assert result.metadata["side_effects_unknown"] is True


def test_recovery_report_is_available_without_an_agent_run(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    durable.append(AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"))

    from rova.app.runtime import build_rova_runtime

    runtime = build_rova_runtime(
        model=Model("mock"),
        stream_fn=_tool_stream(),
        session_root=tmp_path,
        session_id=durable.session_id,
    )

    assert runtime.recovery_report is runtime.session.recovery_report
    assert runtime.recovery_report.recovered_count == 1
    assert runtime.recovery_report.items[0].tool_name == "sample"


def test_recovery_is_idempotent_after_durable_result_append(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    durable.append(AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"))

    first_agent = Agent(Model("mock"), "", [], _tool_stream())
    first = AgentSession.load(first_agent, durable.session_id, session_root=tmp_path)
    second_agent = Agent(Model("mock"), "", [], _tool_stream())
    second = AgentSession.load(second_agent, durable.session_id, session_root=tmp_path)

    assert first.recovery_report.recovered_count == 1
    assert second.recovery_report.recovered_count == 0
    restored = JsonlSessionStore(tmp_path).load(durable.session_id)
    assert sum(isinstance(message, ToolResultMessage) for message in restored.messages) == 1


def test_recovery_report_keeps_source_call_index_without_a_journal_record(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    durable.append(AssistantMessage([
        ToolCall("first", "sample", {}),
        ToolCall("second", "sample", {}),
    ], stop_reason="tool_calls"))
    agent = Agent(Model("mock"), "", [], _tool_stream())

    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    assert [(item.tool_name, item.call_index) for item in session.recovery_report.items] == [
        ("sample", 0),
        ("sample", 1),
    ]


def test_recovery_is_branch_scoped(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    root_id = durable.append(UserMessage("run"))
    first_assistant_id = durable.append(AssistantMessage([ToolCall("first", "sample", {})], stop_reason="tool_calls"))
    durable.branch(root_id)
    second_assistant_id = durable.append(AssistantMessage([ToolCall("second", "sample", {})], stop_reason="tool_calls"))
    journal = ToolExecutionJournal(tmp_path, durable.session_id)
    journal.append(AgentEvent(
        "tool_execution_state", tool_call_id="second", tool_name="sample", batch_id="batch",
        call_index=0, batch_mode="parallel", execution_mode="parallel", execution_state="completed",
        outcome="success", result="second result", is_error=False, metadata={"outcome": "success"},
    ), assistant_entry_id=second_assistant_id)

    first_agent = Agent(Model("mock"), "", [], _tool_stream())
    AgentSession.load(first_agent, durable.session_id, session_root=tmp_path, leaf_id=first_assistant_id)
    assert [message.tool_call_id for message in first_agent.messages if isinstance(message, ToolResultMessage)] == ["first"]

    second_agent = Agent(Model("mock"), "", [], _tool_stream())
    AgentSession.load(second_agent, durable.session_id, session_root=tmp_path, leaf_id=second_assistant_id)
    second_results = [message for message in second_agent.messages if isinstance(message, ToolResultMessage)]
    assert [(message.tool_call_id, message.text) for message in second_results] == [("second", "second result")]


def test_recovery_rejects_out_of_order_session_tool_results_without_mutating_agent(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    durable.append(AssistantMessage([
        ToolCall("first", "sample", {}),
        ToolCall("second", "sample", {}),
    ], stop_reason="tool_calls"))
    durable.append(ToolResultMessage("second", "sample", [TextBlock("out of order")]))
    agent = Agent(Model("mock"), "", [], _tool_stream())

    with pytest.raises(SessionRecoveryError, match="next unresolved ToolCall"):
        AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    assert agent.messages == []
    AgentSession.create(agent, session_root=tmp_path)


def test_recovery_rejects_selected_journal_identity_mismatch_without_mutating_agent(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    assistant_entry_id = durable.append(AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"))
    journal = ToolExecutionJournal(tmp_path, durable.session_id)
    journal.append(AgentEvent(
        "tool_execution_state", tool_call_id="call-1", tool_name="sample", batch_id="batch",
        call_index=0, batch_mode="parallel", execution_mode="parallel", execution_state="started",
    ), assistant_entry_id=assistant_entry_id)
    journal_path = journal.path
    journal_path.write_text(journal_path.read_text(encoding="utf-8").replace('"tool_name":"sample"', '"tool_name":"other"'), encoding="utf-8")
    agent = Agent(Model("mock"), "", [], _tool_stream())

    with pytest.raises(SessionRecoveryError, match="does not match"):
        AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    assert agent.messages == []
    AgentSession.create(agent, session_root=tmp_path)


def test_existing_session_tool_result_skips_irrelevant_journal_state(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("run"))
    assistant_entry_id = durable.append(AssistantMessage([ToolCall("call-1", "sample", {})], stop_reason="tool_calls"))
    durable.append(ToolResultMessage("call-1", "sample", [TextBlock("session authority")]))
    journal = ToolExecutionJournal(tmp_path, durable.session_id)
    journal.append(AgentEvent(
        "tool_execution_state", tool_call_id="call-1", tool_name="sample", batch_id="batch",
        call_index=0, batch_mode="parallel", execution_mode="parallel", execution_state="started",
    ), assistant_entry_id=assistant_entry_id)
    journal.path.write_text(journal.path.read_text(encoding="utf-8").replace('"tool_name":"sample"', '"tool_name":"other"'), encoding="utf-8")
    agent = Agent(Model("mock"), "", [], _tool_stream())

    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    assert session.recovery_report.recovered_count == 0
    assert [message.text for message in agent.messages if isinstance(message, ToolResultMessage)] == ["session authority"]
