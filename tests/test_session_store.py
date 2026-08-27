import json

import pytest

from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.agent_session.serialization import MessageSerializationError, message_from_dict, message_to_dict
from rova.agent_session.session_store import JsonlSessionStore, SessionCorruptionError, SessionStoreError


@pytest.mark.parametrize(
    "message",
    [
        UserMessage("hello"),
        AssistantMessage([TextBlock("answer")], stop_reason="stop"),
        AssistantMessage([ToolCall("call-1", "calc", {"expression": "2 + 2"})], stop_reason="tool_calls"),
        ToolResultMessage("call-1", "calc", [TextBlock("4")]),
        ToolResultMessage("call-1", "calc", [TextBlock("bad arguments")], is_error=True),
    ],
)
def test_message_serialization_round_trips_each_committed_message_kind(message):
    restored = message_from_dict(message_to_dict(message))

    assert restored == message


def test_store_appends_one_linear_jsonl_entry_per_message_and_reloads_in_order(tmp_path):
    store = JsonlSessionStore(tmp_path)
    session = store.create()
    messages = [
        UserMessage("calculate 2 + 2"),
        AssistantMessage([ToolCall("call-1", "calc", {"expression": "2 + 2"})], stop_reason="tool_calls"),
        ToolResultMessage("call-1", "calc", [TextBlock("4")]),
        AssistantMessage([TextBlock("2 + 2 = 4")]),
    ]

    for message in messages:
        session.append(message)

    loaded = store.load(session.session_id)
    lines = (tmp_path / f"{session.session_id}.jsonl").read_text(encoding="utf-8").splitlines()

    assert loaded.messages == messages
    assert [json.loads(line)["type"] for line in lines] == ["session", "message", "message", "message", "message"]
    assert [json.loads(line)["parent_id"] for line in lines[1:]] == [None, loaded.entry_ids[0], loaded.entry_ids[1], loaded.entry_ids[2]]


def test_store_rejects_corrupt_jsonl_without_silently_skipping_it(tmp_path):
    session_id = "badsession"
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text(
        '{"type":"session","version":1,"session_id":"badsession","created_at":"2026-01-01T00:00:00+00:00"}\nnot json\n',
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match="line 2"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_partial_assistant_message_is_never_serialized_as_durable_history():
    with pytest.raises(MessageSerializationError, match="partial"):
        message_to_dict(AssistantMessage([TextBlock("partial")], partial=True))


def test_each_successful_append_flushes_and_fsyncs_before_returning(tmp_path, monkeypatch):
    fsync_calls = []
    monkeypatch.setattr("rova.agent_session.session_store.os.fsync", lambda file_descriptor: fsync_calls.append(file_descriptor))

    session = JsonlSessionStore(tmp_path).create()
    session.append(UserMessage("durable"))

    assert len(fsync_calls) == 2


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"])
def test_store_round_trips_unicode_line_separators_without_splitting_jsonl_records(tmp_path, separator):
    session = JsonlSessionStore(tmp_path).create()
    message = UserMessage(f"before{separator}after")

    session.append(message)

    assert JsonlSessionStore(tmp_path).load(session.session_id).messages == [message]


def test_store_rejects_duplicate_entry_id_even_when_parent_chain_matches(tmp_path):
    session_id = "duplicates"
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"type":"session","version":1,"session_id":"duplicates","created_at":"2026-01-01T00:00:00+00:00"}',
                '{"type":"message","entry_id":"same","parent_id":null,"message":{"role":"user","content":"first"}}',
                '{"type":"message","entry_id":"same","parent_id":"same","message":{"role":"user","content":"second"}}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match="duplicate entry_id"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_store_rejects_header_missing_required_created_at(tmp_path):
    session_id = "missingcreated"
    (tmp_path / f"{session_id}.jsonl").write_text(
        '{"type":"session","version":1,"session_id":"missingcreated"}\n', encoding="utf-8"
    )

    with pytest.raises(SessionCorruptionError, match="invalid session header"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_store_rejects_message_entry_missing_parent_id_even_at_linear_root(tmp_path):
    session_id = "missingparent"
    (tmp_path / f"{session_id}.jsonl").write_text(
        "\n".join(
            [
                '{"type":"session","version":1,"session_id":"missingparent","created_at":"2026-01-01T00:00:00+00:00"}',
                '{"type":"message","entry_id":"entry1","message":{"role":"user","content":"first"}}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match="missing parent_id"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_store_loads_append_only_tree_and_restores_each_leaf_path(tmp_path):
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    root = durable.append(UserMessage("root"))
    old_leaf = durable.append(AssistantMessage([TextBlock("old branch")]))

    durable.branch(root)
    new_leaf = durable.append(AssistantMessage([TextBlock("new branch")]))
    loaded = store.load(durable.session_id)

    assert [entry.parent_id for entry in loaded.entries] == [None, root, root]
    assert loaded.leaf_id == new_leaf
    assert [entry.entry_id for entry in loaded.path_to_leaf(old_leaf)] == [root, old_leaf]
    assert [entry.message for entry in loaded.path_to_leaf(new_leaf)] == [UserMessage("root"), AssistantMessage([TextBlock("new branch")])]


@pytest.mark.parametrize(
    "session_id, entries, reason",
    [
        (
            "secondroot",
            [
                '{"type":"message","entry_id":"entry1","parent_id":null,"message":{"role":"user","content":"one"}}',
                '{"type":"message","entry_id":"entry2","parent_id":null,"message":{"role":"user","content":"two"}}',
            ],
            "second root",
        ),
        (
            "dangling",
            [
                '{"type":"message","entry_id":"entry1","parent_id":null,"message":{"role":"user","content":"one"}}',
                '{"type":"message","entry_id":"entry2","parent_id":"future","message":{"role":"user","content":"two"}}',
            ],
            "dangling parent",
        ),
    ],
)
def test_store_rejects_non_tree_parent_relationships(tmp_path, session_id, entries, reason):
    header = f'{{"type":"session","version":1,"session_id":"{session_id}","created_at":"2026-01-01T00:00:00+00:00"}}'
    (tmp_path / f"{session_id}.jsonl").write_text("\n".join([header, *entries, ""]), encoding="utf-8")

    with pytest.raises(SessionCorruptionError, match=reason):
        JsonlSessionStore(tmp_path).load(session_id)


def test_store_rejects_child_that_references_a_later_entry(tmp_path):
    session_id = "futureparent"
    header = '{"type":"session","version":1,"session_id":"futureparent","created_at":"2026-01-01T00:00:00+00:00"}'
    entries = [
        '{"type":"message","entry_id":"root","parent_id":null,"message":{"role":"user","content":"root"}}',
        '{"type":"message","entry_id":"child","parent_id":"later","message":{"role":"user","content":"child"}}',
        '{"type":"message","entry_id":"later","parent_id":"root","message":{"role":"user","content":"later"}}',
    ]
    (tmp_path / f"{session_id}.jsonl").write_text("\n".join([header, *entries, ""]), encoding="utf-8")

    with pytest.raises(SessionCorruptionError, match="dangling parent"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_store_rejects_unknown_jsonl_entry_type(tmp_path):
    session_id = "unknowntype"
    (tmp_path / f"{session_id}.jsonl").write_text(
        "\n".join(
            [
                '{"type":"session","version":1,"session_id":"unknowntype","created_at":"2026-01-01T00:00:00+00:00"}',
                '{"type":"future_entry","entry_id":"entry1","parent_id":null}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match="expected message entry"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_failed_fsync_does_not_advance_durable_tree_memory_state(tmp_path, monkeypatch):
    durable = JsonlSessionStore(tmp_path).create()
    root = durable.append(UserMessage("root"))
    before_entries = list(durable.entries)
    before_by_id = dict(durable.by_id)
    monkeypatch.setattr("rova.agent_session.session_store.os.fsync", lambda _: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(SessionStoreError, match="could not append"):
        durable.append(UserMessage("not committed"))

    assert durable.entries == before_entries
    assert durable.by_id == before_by_id
    assert durable.leaf_id == root


def test_store_rejects_nonempty_jsonl_without_final_newline(tmp_path):
    session_id = "missingnewline"
    (tmp_path / f"{session_id}.jsonl").write_text(
        "\n".join(
            [
                '{"type":"session","version":1,"session_id":"missingnewline","created_at":"2026-01-01T00:00:00+00:00"}',
                '{"type":"message","entry_id":"entry1","parent_id":null,"message":{"role":"user","content":"complete json but incomplete record"}}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match="final newline"):
        JsonlSessionStore(tmp_path).load(session_id)


def test_store_lists_read_only_session_summaries_by_most_recent_update(tmp_path):
    store = JsonlSessionStore(tmp_path)
    older = store.create()
    older.append(UserMessage("first user request"))
    older.append(AssistantMessage([TextBlock("first answer")]))
    newer = store.create()
    newer.append(UserMessage("second user request"))

    summaries = store.list_sessions()

    assert [summary.session_id for summary in summaries] == [newer.session_id, older.session_id]
    assert summaries[0].first_user_preview == "second user request"
    assert summaries[1].first_user_preview == "first user request"
    assert summaries[0].created_at
    assert summaries[0].updated_at


def test_store_lists_an_empty_directory_without_creating_persistence(tmp_path):
    store = JsonlSessionStore(tmp_path / "missing")

    assert store.list_sessions() == []
    assert not store.root.exists()
