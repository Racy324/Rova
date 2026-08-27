import json

import pytest

from rova.agent_session.entries import CompactionEntry, MessageEntry
from rova.agent_session.session_store import JsonlSessionStore, SessionCorruptionError, SessionStoreError
from rova.ai.messages import UserMessage


def test_session_entry_data_structures_are_pure_entry_records():
    message_entry = MessageEntry("message-1", None, UserMessage("hello"))
    compaction_entry = CompactionEntry("compaction-1", "message-1", "summary", "message-1")

    assert message_entry.entry_id == "message-1"
    assert compaction_entry.first_kept_entry_id == "message-1"


def test_append_compaction_is_an_append_only_current_leaf_child_and_round_trips(tmp_path):
    store = JsonlSessionStore(tmp_path)
    session = store.create()
    root = session.append(UserMessage("one"))
    kept = session.append(UserMessage("two"))

    compaction = session.append_compaction("summary of one", kept)
    loaded = store.load(session.session_id)
    record = json.loads((tmp_path / f"{session.session_id}.jsonl").read_text(encoding="utf-8").splitlines()[-1])

    assert compaction.parent_id == kept
    assert session.leaf_id == compaction.entry_id
    assert record == {
        "type": "compaction",
        "entry_id": compaction.entry_id,
        "parent_id": kept,
        "summary": "summary of one",
        "first_kept_entry_id": kept,
    }
    assert isinstance(loaded.path_to_leaf()[-1], CompactionEntry)
    assert loaded.messages == [UserMessage("one"), UserMessage("two")]


def test_append_compaction_accepts_none_first_kept_and_excludes_compaction_from_physical_messages(tmp_path):
    session = JsonlSessionStore(tmp_path).create()
    session.append(UserMessage("one"))

    compaction = session.append_compaction("summary of all history", None)

    assert compaction.first_kept_entry_id is None
    assert session.physical_messages == [UserMessage("one")]
    assert session.messages == session.physical_messages


def test_compaction_appended_from_an_older_branch_leaf_reloads_successfully(tmp_path):
    store = JsonlSessionStore(tmp_path)
    session = store.create()
    root = session.append(UserMessage("one"))
    session.append(UserMessage("two"))
    session.append(UserMessage("three"))

    session.branch(root)
    compaction = session.append_compaction("summary of one", root)
    loaded = store.load(session.session_id)

    assert loaded.leaf_id == compaction.entry_id
    assert [entry.entry_id for entry in loaded.path_to_leaf()] == [root, compaction.entry_id]


def test_append_compaction_fsyncs_before_mutating_in_memory_tree(tmp_path, monkeypatch):
    store = JsonlSessionStore(tmp_path)
    session = store.create()
    root = session.append(UserMessage("one"))
    observed = []

    def observe_fsync(_):
        observed.append((list(session.entries), dict(session.by_id), session.leaf_id))

    monkeypatch.setattr("rova.agent_session.session_store.os.fsync", observe_fsync)
    compaction = session.append_compaction("summary", root)

    assert observed == [([session.entries[0]], {root: session.entries[0]}, root)]
    assert session.leaf_id == compaction.entry_id


def test_failed_compaction_append_leaves_tree_unchanged(tmp_path, monkeypatch):
    store = JsonlSessionStore(tmp_path)
    session = store.create()
    root = session.append(UserMessage("one"))
    before_entries = list(session.entries)
    before_by_id = dict(session.by_id)

    monkeypatch.setattr(
        "rova.agent_session.session_store.os.fsync",
        lambda _: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(SessionStoreError, match="could not append"):
        session.append_compaction("summary", root)

    assert session.entries == before_entries
    assert session.by_id == before_by_id
    assert session.leaf_id == root


@pytest.mark.parametrize(
    "summary, first_kept, reason",
    [
        ("", None, "summary"),
        ("summary", "missing", "first_kept"),
    ],
)
def test_append_compaction_validates_payload_before_writing(tmp_path, summary, first_kept, reason):
    session = JsonlSessionStore(tmp_path).create()
    session.append(UserMessage("one"))

    with pytest.raises(SessionStoreError, match=reason):
        session.append_compaction(summary, first_kept)


@pytest.mark.parametrize(
    "records, reason",
    [
        (
            [
                {"type": "message", "entry_id": "root", "parent_id": None, "message": {"role": "user", "content": "root"}},
                {"type": "message", "entry_id": "left", "parent_id": "root", "message": {"role": "user", "content": "left"}},
                {"type": "message", "entry_id": "right", "parent_id": "root", "message": {"role": "user", "content": "right"}},
                {"type": "compaction", "entry_id": "compact", "parent_id": "right", "summary": "summary", "first_kept_entry_id": "left"},
            ],
            "ancestor",
        ),
        (
            [
                {"type": "message", "entry_id": "root", "parent_id": None, "message": {"role": "user", "content": "root"}},
                {"type": "compaction", "entry_id": "first", "parent_id": "root", "summary": "summary", "first_kept_entry_id": "root"},
                {"type": "compaction", "entry_id": "second", "parent_id": "first", "summary": "summary", "first_kept_entry_id": "first"},
            ],
            "MessageEntry",
        ),
        (
            [
                {"type": "message", "entry_id": "root", "parent_id": None, "message": {"role": "user", "content": "root"}},
                {"type": "compaction", "entry_id": "compact", "parent_id": "root", "summary": "summary", "first_kept_entry_id": "future"},
                {"type": "message", "entry_id": "future", "parent_id": "root", "message": {"role": "user", "content": "future"}},
            ],
            "first_kept",
        ),
    ],
)
def test_loader_rejects_invalid_compaction_first_kept_targets(tmp_path, records, reason):
    session_id = "compactioninvalid"
    header = {"type": "session", "version": 1, "session_id": session_id, "created_at": "2026-01-01T00:00:00+00:00"}
    (tmp_path / f"{session_id}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in [header, *records]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SessionCorruptionError, match=reason):
        JsonlSessionStore(tmp_path).load(session_id)
