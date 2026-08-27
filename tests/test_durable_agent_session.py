import asyncio

import pytest

import rova.agent_session.agent_session as agent_session_module
from rova.ai.events import Start, StreamDone, TextDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.agent_session.agent_session import AgentSession, SessionBranchError, SessionIncompleteError, SessionPersistenceError
from rova.agent_session.session_store import JsonlSessionStore, SessionStoreError
from tests.tool_helpers import make_test_calc_tool


async def two_turn_calc_stream(model, context, options):
    if any(isinstance(message, ToolResultMessage) for message in context.messages):
        yield StreamDone(AssistantMessage([TextBlock("2 + 2 = 4")]))
    else:
        yield StreamDone(
            AssistantMessage(
                [ToolCall("call-1", "calc", {"expression": "2 + 2"})],
                stop_reason="tool_calls",
            )
        )


@pytest.mark.asyncio
async def test_durable_session_resumes_committed_history_and_builds_next_context_from_it(tmp_path):
    first_agent = Agent(Model("mock"), "", [make_test_calc_tool()], two_turn_calc_stream)
    first_session = AgentSession.create(first_agent, session_root=tmp_path)

    await first_session.prompt("calculate 2 + 2")

    resumed_contexts = []

    async def resumed_stream(model, context, options):
        resumed_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("continued")]))

    resumed_agent = Agent(Model("mock"), "", [make_test_calc_tool()], resumed_stream)
    resumed_session = AgentSession.load(resumed_agent, first_session.session_id, session_root=tmp_path)

    await resumed_session.prompt("what next?")

    assert resumed_agent.messages[:4] == first_agent.messages
    assert resumed_contexts[0].messages == resumed_agent.messages[:-1]
    assert any(isinstance(message, ToolResultMessage) and message.text == "4" for message in resumed_contexts[0].messages)
    assert resumed_contexts[0].messages[-1] == UserMessage("what next?")


async def partial_text_stream(model, context, options):
    yield Start(AssistantMessage([], partial=True))
    yield TextDelta("par", AssistantMessage([TextBlock("par")], partial=True))
    yield StreamDone(AssistantMessage([TextBlock("partial became final")]))


@pytest.mark.asyncio
async def test_durable_session_persists_only_final_streaming_message_once(tmp_path):
    agent = Agent(Model("mock"), "", [], partial_text_stream)
    session = AgentSession.create(agent, session_root=tmp_path)

    await session.prompt("hello")

    loaded = JsonlSessionStore(tmp_path).load(session.session_id)
    assert loaded.messages == [UserMessage("hello"), AssistantMessage([TextBlock("partial became final")])]
    assert session.persisted_message_count == len(agent.messages) == 2


@pytest.mark.asyncio
async def test_resume_with_one_of_multiple_tool_calls_missing_result_blocks_new_provider_run(tmp_path):
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("two tools"))
    durable.append(
        AssistantMessage(
            [ToolCall("call-1", "calc", {"expression": "1 + 1"}), ToolCall("call-2", "calc", {"expression": "2 + 2"})],
            stop_reason="tool_calls",
        )
    )
    durable.append(ToolResultMessage("call-1", "calc", [TextBlock("2")]))
    agent = Agent(Model("mock"), "", [make_test_calc_tool()], two_turn_calc_stream)
    resumed = AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    with pytest.raises(SessionIncompleteError, match="incomplete"):
        await resumed.prompt("continue")

    assert agent.messages == durable.messages


@pytest.mark.asyncio
async def test_persistence_failure_after_assistant_commit_faults_session_and_stops_run(tmp_path, monkeypatch):
    async def final_stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.create(agent, session_root=tmp_path)

    def fail_after_user(session_id, parent_id, message):
        if isinstance(message, AssistantMessage):
            raise SessionStoreError("disk full")
        return original_append(session_id, parent_id, message)

    original_append = session._durable_session.store.append_message
    monkeypatch.setattr(session._durable_session.store, "append_message", fail_after_user)

    with pytest.raises(SessionPersistenceError, match="failed to persist"):
        await session.prompt("hello")
    with pytest.raises(SessionPersistenceError, match="faulted"):
        await session.prompt("try again")

    assert session.faulted is True
    assert agent.messages == [UserMessage("hello"), AssistantMessage([TextBlock("final")])]
    assert JsonlSessionStore(tmp_path).load(session.session_id).messages == [UserMessage("hello")]


@pytest.mark.asyncio
async def test_user_message_is_durable_before_provider_failure(tmp_path):
    async def failing_stream(model, context, options):
        raise RuntimeError("provider crashed")
        yield  # pragma: no cover

    agent = Agent(Model("mock"), "", [], failing_stream)
    session = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(RuntimeError, match="provider crashed"):
        await session.prompt("keep this")

    assert JsonlSessionStore(tmp_path).load(session.session_id).messages == [UserMessage("keep this")]


def test_new_durable_session_rejects_agent_that_already_has_runtime_history(tmp_path):
    agent = Agent(Model("mock"), "", [], two_turn_calc_stream)
    agent.messages.append(UserMessage("existing transient history"))

    with pytest.raises(ValueError, match="empty Agent"):
        AgentSession.create(agent, session_root=tmp_path)


def test_agent_allows_exactly_one_durable_session_binding_until_it_is_closed(tmp_path):
    agent = Agent(Model("mock"), "", [], two_turn_calc_stream)
    first = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(RuntimeError, match="already has a durable AgentSession"):
        AgentSession.create(agent, session_root=tmp_path)

    first.close()
    replacement = AgentSession.create(agent, session_root=tmp_path)
    replacement.close()


def test_load_requires_an_agent_without_existing_runtime_history(tmp_path):
    durable = JsonlSessionStore(tmp_path).create()
    durable.append(UserMessage("persisted"))
    agent = Agent(Model("mock"), "", [], two_turn_calc_stream)
    agent.messages.append(UserMessage("existing runtime history"))

    with pytest.raises(ValueError, match="fresh Agent"):
        AgentSession.load(agent, durable.session_id, session_root=tmp_path)


@pytest.mark.asyncio
async def test_non_json_tool_arguments_fault_session_after_assistant_is_committed(tmp_path):
    async def non_json_tool_stream(model, context, options):
        yield StreamDone(
            AssistantMessage(
                [ToolCall("call-1", "invalid", {"not_json": {1}})],
                stop_reason="tool_calls",
            )
        )

    agent = Agent(Model("mock"), "", [], non_json_tool_stream)
    session = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(SessionPersistenceError, match="failed to persist"):
        await session.prompt("call invalid")

    assert session.faulted is True
    assert agent.messages == [
        UserMessage("call invalid"),
        AssistantMessage([ToolCall("call-1", "invalid", {"not_json": {1}})], stop_reason="tool_calls"),
    ]


@pytest.mark.asyncio
async def test_branch_rebuilds_agent_messages_and_next_prompt_appends_from_selected_leaf(tmp_path):
    contexts = []

    async def final_stream(model, context, options):
        contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock(f"reply-{len(contexts)}")]))

    agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.create(agent, session_root=tmp_path)
    await session.prompt("first")
    root = session._durable_session.entries[0].entry_id
    old_leaf = session._durable_session.leaf_id
    transcript_path = tmp_path / f"{session.session_id}.jsonl"
    before_branch = transcript_path.read_bytes()

    session.branch(root)

    assert agent.messages == [UserMessage("first")]
    assert session.persisted_message_count == 1
    assert transcript_path.read_bytes() == before_branch
    await session.prompt("second")

    entries = session._durable_session.entries
    assert [entry.parent_id for entry in entries] == [None, root, root, entries[2].entry_id]
    assert old_leaf in session._durable_session.by_id
    assert contexts[-1].messages == [UserMessage("first"), UserMessage("second")]


@pytest.mark.asyncio
async def test_load_selected_leaf_restores_only_that_branch_path(tmp_path):
    async def final_stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    first_agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.create(first_agent, session_root=tmp_path)
    await session.prompt("first")
    root = session._durable_session.entries[0].entry_id
    old_leaf = session._durable_session.leaf_id
    session.branch(root)
    await session.prompt("second")
    new_leaf = session._durable_session.leaf_id

    old_agent = Agent(Model("mock"), "", [], final_stream)
    old_session = AgentSession.load(old_agent, session.session_id, session_root=tmp_path, leaf_id=old_leaf)
    new_agent = Agent(Model("mock"), "", [], final_stream)
    new_session = AgentSession.load(new_agent, session.session_id, session_root=tmp_path, leaf_id=new_leaf)
    default_agent = Agent(Model("mock"), "", [], final_stream)
    default_session = AgentSession.load(default_agent, session.session_id, session_root=tmp_path)

    assert old_agent.messages == [UserMessage("first"), AssistantMessage([TextBlock("final")])]
    assert new_agent.messages == [UserMessage("first"), UserMessage("second"), AssistantMessage([TextBlock("final")])]
    assert old_session.persisted_message_count == 2
    assert new_session.persisted_message_count == 3
    assert default_agent.messages == new_agent.messages
    assert default_session.persisted_message_count == 3

    unknown_agent = Agent(Model("mock"), "", [], final_stream)
    with pytest.raises(SessionStoreError, match="unknown session entry"):
        AgentSession.load(unknown_agent, session.session_id, session_root=tmp_path, leaf_id="missing")


@pytest.mark.asyncio
async def test_branch_rejects_while_prompt_active_then_allows_navigation_after_completion(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_stream(model, context, options):
        started.set()
        await release.wait()
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model("mock"), "", [], blocking_stream)
    session = AgentSession.create(agent, session_root=tmp_path)
    prompt_task = asyncio.create_task(session.prompt("first"))
    await started.wait()
    root = session._durable_session.entries[0].entry_id

    with pytest.raises(SessionBranchError, match="prompt is active"):
        session.branch(root)
    with pytest.raises(SessionBranchError, match="already active"):
        await session.prompt("overlap")

    release.set()
    await prompt_task
    session.branch(root)


@pytest.mark.asyncio
async def test_close_rejects_while_prompt_active_then_succeeds_after_prompt_finishes(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_stream(model, context, options):
        started.set()
        await release.wait()
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model("mock"), "", [], blocking_stream)
    session = AgentSession.create(agent, session_root=tmp_path)
    prompt_task = asyncio.create_task(session.prompt("first"))
    await started.wait()

    with pytest.raises(RuntimeError, match="prompt is active"):
        session.close()

    release.set()
    await prompt_task
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        await session.prompt("after close")


@pytest.mark.asyncio
async def test_faulted_session_rejects_branch_but_can_be_reloaded_on_a_fresh_agent(tmp_path, monkeypatch):
    async def final_stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.create(agent, session_root=tmp_path)
    root = None
    original_append = session._durable_session.store.append_message

    def fail_assistant(session_id, parent_id, message):
        if isinstance(message, AssistantMessage):
            raise SessionStoreError("disk full")
        return original_append(session_id, parent_id, message)

    monkeypatch.setattr(session._durable_session.store, "append_message", fail_assistant)
    with pytest.raises(SessionPersistenceError):
        await session.prompt("first")
    root = session._durable_session.leaf_id

    with pytest.raises(SessionPersistenceError, match="faulted"):
        session.branch(root)

    session.close()
    fresh_agent = Agent(Model("mock"), "", [], final_stream)
    loaded = AgentSession.load(fresh_agent, session.session_id, session_root=tmp_path)
    assert loaded.faulted is False


@pytest.mark.asyncio
async def test_provider_exception_releases_prompt_guard_for_later_branch(tmp_path):
    async def failing_stream(model, context, options):
        raise RuntimeError("provider crashed")
        yield  # pragma: no cover

    agent = Agent(Model("mock"), "", [], failing_stream)
    session = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(RuntimeError, match="provider crashed"):
        await session.prompt("first")

    root = session._durable_session.leaf_id
    session.branch(root)
    assert agent.messages == [UserMessage("first")]


@pytest.mark.asyncio
async def test_branch_to_incomplete_tool_call_path_blocks_prompt_until_complete_branch_is_selected(tmp_path):
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    root = durable.append(UserMessage("root"))
    incomplete_leaf = durable.append(
        AssistantMessage([ToolCall("call-1", "calc", {"expression": "1 + 1"})], stop_reason="tool_calls")
    )
    durable.branch(root)
    durable.append(UserMessage("complete branch"))
    complete_leaf = durable.append(AssistantMessage([TextBlock("complete")]))

    async def final_stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("continued")]))

    agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    assert session._durable_session.leaf_id == complete_leaf
    session.branch(incomplete_leaf)
    with pytest.raises(SessionIncompleteError, match="incomplete"):
        await session.prompt("do not run")

    session.branch(complete_leaf)
    await session.prompt("continue")


def test_branch_unknown_entry_is_a_session_branch_error(tmp_path):
    agent = Agent(Model("mock"), "", [], two_turn_calc_stream)
    session = AgentSession.create(agent, session_root=tmp_path)

    with pytest.raises(SessionBranchError, match="unknown branch entry"):
        session.branch("missing")


def test_load_and_branch_project_paths_through_context_builder(tmp_path, monkeypatch):
    durable = JsonlSessionStore(tmp_path).create()
    root = durable.append(UserMessage("root"))
    old_leaf = durable.append(AssistantMessage([TextBlock("old")]))
    durable.branch(root)
    new_leaf = durable.append(AssistantMessage([TextBlock("new")]))
    calls = []

    def recording_builder(entries):
        calls.append([entry.entry_id for entry in entries])
        return [entry.message for entry in entries]

    monkeypatch.setattr(agent_session_module, "build_session_messages", recording_builder, raising=False)
    agent = Agent(Model("mock"), "", [], two_turn_calc_stream)
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path, leaf_id=old_leaf)
    session.branch(new_leaf)

    assert calls == [[root, old_leaf], [root, new_leaf]]


def test_load_and_branch_set_cursor_from_built_messages_not_path_length(tmp_path, monkeypatch):
    durable = JsonlSessionStore(tmp_path).create()
    root = durable.append(UserMessage("root"))
    old_leaf = durable.append(AssistantMessage([TextBlock("old")]))
    durable.branch(root)
    new_leaf = durable.append(AssistantMessage([TextBlock("new")]))

    def shortened_builder(entries):
        return [entries[0].message]

    monkeypatch.setattr(agent_session_module, "build_session_messages", shortened_builder, raising=False)
    agent = Agent(Model("mock"), "", [], two_turn_calc_stream)
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path, leaf_id=old_leaf)

    assert len(session._durable_session.path_to_leaf()) == 2
    assert agent.messages == [UserMessage("root")]
    assert session.persisted_message_count == 1

    session.branch(new_leaf)
    assert session.persisted_message_count == 1


@pytest.mark.asyncio
async def test_previous_provider_context_snapshot_is_unchanged_by_later_branch_projection(tmp_path):
    contexts = []

    async def recording_stream(model, context, options):
        contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model("mock"), "", [], recording_stream)
    session = AgentSession.create(agent, session_root=tmp_path)
    await session.prompt("first")
    root = session._durable_session.entries[0].entry_id
    await session.prompt("second")
    previous_context = contexts[1]

    session.branch(root)

    assert previous_context.messages == [
        UserMessage("first"),
        AssistantMessage([TextBlock("final")]),
        UserMessage("second"),
    ]
