import json

import pytest

from rova.agent_session.context_builder import (
    COMPACTION_SUMMARY_PREAMBLE,
    SessionProjectionError,
    build_session_messages,
    build_session_projection,
)
from rova.agent_session.entries import CompactionEntry, MessageEntry
from rova.agent_session.session_store import JsonlSessionStore, SessionCorruptionError, SessionStoreError
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.agent_session.agent_session import AgentSession


def _messages(entries):
    return [projected.message for projected in build_session_projection(entries)]


def test_message_entry_projection_preserves_raw_message_identity_and_provenance():
    raw = UserMessage("one")

    projection = build_session_projection([MessageEntry("e1", None, raw)])

    assert projection[0].message is raw
    assert projection[0].source_entry_id == "e1"
    assert build_session_messages([MessageEntry("e1", None, raw)]) == [raw]


def test_first_compaction_projects_synthetic_user_summary_before_retained_raw_entries():
    messages = [UserMessage("one"), UserMessage("two"), UserMessage("three"), UserMessage("four")]
    entries = [
        MessageEntry("e1", None, messages[0]),
        MessageEntry("e2", "e1", messages[1]),
        MessageEntry("e3", "e2", messages[2]),
        MessageEntry("e4", "e3", messages[3]),
        CompactionEntry("c1", "e4", "S1", "e3"),
    ]

    projection = build_session_projection(entries)

    assert [item.source_entry_id for item in projection] == [None, "e3", "e4"]
    assert isinstance(projection[0].message, UserMessage)
    assert projection[0].message.content == f"{COMPACTION_SUMMARY_PREAMBLE}\n<SUMMARY>\nS1\n</SUMMARY>"
    assert _messages(entries) == [projection[0].message, messages[2], messages[3]]


def test_none_first_kept_projects_only_summary_then_later_raw_entries():
    first = UserMessage("one")
    later = UserMessage("later")
    entries = [
        MessageEntry("e1", None, first),
        CompactionEntry("c1", "e1", "all history", None),
        MessageEntry("e2", "c1", later),
    ]

    projection = build_session_projection(entries)

    assert [item.source_entry_id for item in projection] == [None, "e2"]
    assert projection[1].message is later


def test_repeated_compaction_replaces_prior_summary_without_resurrecting_hidden_history():
    messages = [UserMessage(value) for value in ["M1", "M2", "M3", "M4", "M5", "M6"]]
    entries = [
        MessageEntry("e1", None, messages[0]),
        MessageEntry("e2", "e1", messages[1]),
        MessageEntry("e3", "e2", messages[2]),
        MessageEntry("e4", "e3", messages[3]),
        CompactionEntry("c1", "e4", "S1", "e3"),
        MessageEntry("e6", "c1", messages[4]),
        MessageEntry("e7", "e6", messages[5]),
        CompactionEntry("c2", "e7", "S2", "e6"),
    ]

    projection = build_session_projection(entries)

    assert [item.source_entry_id for item in projection] == [None, "e6", "e7"]
    assert projection[0].message.content.endswith("S2\n</SUMMARY>")
    assert [item.message for item in projection[1:]] == messages[4:]


def test_projection_rejects_first_kept_hidden_by_an_earlier_compaction():
    entries = [
        MessageEntry("e1", None, UserMessage("M1")),
        MessageEntry("e2", "e1", UserMessage("M2")),
        MessageEntry("e3", "e2", UserMessage("M3")),
        CompactionEntry("c1", "e3", "S1", "e3"),
        CompactionEntry("c2", "c1", "S2", "e1"),
    ]

    with pytest.raises(SessionProjectionError, match="not visible"):
        build_session_projection(entries)


def test_store_rejects_appending_compaction_with_physically_valid_but_hidden_first_kept(tmp_path):
    durable = JsonlSessionStore(tmp_path).create()
    first = durable.append(UserMessage("M1"))
    durable.append(UserMessage("M2"))
    kept = durable.append(UserMessage("M3"))
    durable.append_compaction("S1", kept)

    with pytest.raises(SessionStoreError, match="not visible"):
        durable.append_compaction("S2", first)


def test_loader_rejects_compaction_with_first_kept_hidden_by_an_earlier_compaction(tmp_path):
    session_id = "hiddenfirstkept"
    records = [
        {"type": "session", "version": 1, "session_id": session_id, "created_at": "2026-01-01T00:00:00+00:00"},
        {"type": "message", "entry_id": "e1", "parent_id": None, "message": {"role": "user", "content": "M1"}},
        {"type": "message", "entry_id": "e2", "parent_id": "e1", "message": {"role": "user", "content": "M2"}},
        {"type": "message", "entry_id": "e3", "parent_id": "e2", "message": {"role": "user", "content": "M3"}},
        {"type": "compaction", "entry_id": "c1", "parent_id": "e3", "summary": "S1", "first_kept_entry_id": "e3"},
        {"type": "compaction", "entry_id": "c2", "parent_id": "c1", "summary": "S2", "first_kept_entry_id": "e1"},
    ]
    (tmp_path / f"{session_id}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match="not visible"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_agent_session_load_and_branch_use_compacted_logical_projection(tmp_path):
    durable = JsonlSessionStore(tmp_path).create()
    e1 = durable.append(UserMessage("M1"))
    e2 = durable.append(UserMessage("M2"))
    e3 = durable.append(UserMessage("M3"))
    e4 = durable.append(UserMessage("M4"))
    durable.branch(e2)
    durable.append(UserMessage("sibling"))
    durable.branch(e4)
    c1 = durable.append_compaction("S1", e3)
    e6 = durable.append(UserMessage("M5"))

    async def final_stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path)

    summary = agent.messages[0]
    assert isinstance(summary, UserMessage)
    assert agent.messages[1:] == [UserMessage("M3"), UserMessage("M4"), UserMessage("M5")]
    assert UserMessage("sibling") not in agent.messages
    assert session.persisted_message_count == len(agent.messages) == 4
    assert len(durable.path_to_leaf(e6)) == 6

    session.branch(e4)
    assert agent.messages == [UserMessage("M1"), UserMessage("M2"), UserMessage("M3"), UserMessage("M4")]
    session.branch(c1.entry_id)
    assert agent.messages == [summary, UserMessage("M3"), UserMessage("M4")]
    session.branch(e6)
    assert agent.messages == [summary, UserMessage("M3"), UserMessage("M4"), UserMessage("M5")]


@pytest.mark.asyncio
async def test_compacted_session_persists_only_new_runtime_suffix_not_synthetic_summary(tmp_path):
    durable = JsonlSessionStore(tmp_path).create()
    durable.append(UserMessage("M1"))
    kept = durable.append(UserMessage("M2"))
    durable.append_compaction("S1", kept)

    async def final_stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("A3")]))

    agent = Agent(Model("mock"), "", [], final_stream)
    session = AgentSession.load(agent, durable.session_id, session_root=tmp_path)
    await session.prompt("U3")

    loaded = JsonlSessionStore(tmp_path).load(durable.session_id)
    assert [message.content for message in loaded.physical_messages if isinstance(message, UserMessage)] == ["M1", "M2", "U3"]
    assert loaded.physical_messages[-1] == AssistantMessage([TextBlock("A3")])
    assert all(
        not (isinstance(message, UserMessage) and message.content.startswith(COMPACTION_SUMMARY_PREAMBLE))
        for message in loaded.physical_messages
    )
    assert session.persisted_message_count == len(agent.messages)
