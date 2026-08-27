from __future__ import annotations

import json

import pytest

from rova.agent_core.tool_output import ArtifactStoreError
from rova.artifacts import FileArtifactStore


def test_file_store_writes_raw_utf8_envelope_without_path_in_reference(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")

    reference = store.write_text(
        "你好\nraw output",
        tool_call_id="call-1",
        tool_name="read",
        is_error=False,
        run_id="run-1",
        session_id="session-1",
    )

    files = list((tmp_path / "artifacts").glob("*.json"))
    assert len(files) == 1
    envelope = json.loads(files[0].read_text(encoding="utf-8"))
    assert envelope["raw_output"] == "你好\nraw output"
    assert envelope["tool_call_id"] == "call-1"
    assert envelope["run_id"] == "run-1"
    assert envelope["session_id"] == "session-1"
    assert reference.byte_count == len("你好\nraw output".encode("utf-8"))
    assert "path" not in reference.to_dict()


def test_file_store_uses_distinct_opaque_ids_for_repeated_call_ids(tmp_path):
    store = FileArtifactStore(tmp_path)

    first = store.write_text("one", tool_call_id="same", tool_name="read", is_error=False, run_id=None, session_id=None)
    second = store.write_text("two", tool_call_id="same", tool_name="read", is_error=False, run_id=None, session_id=None)

    assert first.artifact_id != second.artifact_id


def test_file_store_writes_user_requested_output_without_tool_identity(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")

    reference = store.write_text_artifact(
        "# Saved output\n\nA user-requested result.",
        artifact_kind="user_requested_output",
        media_type="text/markdown; charset=utf-8",
        run_id="run-1",
        session_id="session-1",
    )

    envelope = json.loads(next((tmp_path / "artifacts").glob("*.json")).read_text(encoding="utf-8"))
    assert envelope["artifact_kind"] == "user_requested_output"
    assert envelope["media_type"] == "text/markdown; charset=utf-8"
    assert envelope["raw_output"] == "# Saved output\n\nA user-requested result."
    assert envelope["run_id"] == "run-1"
    assert envelope["session_id"] == "session-1"
    assert "tool_call_id" not in envelope
    assert "tool_name" not in envelope
    assert "is_error" not in envelope
    assert reference.run_id == "run-1"


def test_file_store_surfaces_atomic_write_failure(monkeypatch, tmp_path):
    store = FileArtifactStore(tmp_path)
    monkeypatch.setattr("rova.artifacts.file_store.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk failure")))

    with pytest.raises(ArtifactStoreError, match="failed to persist tool output artifact"):
        store.write_text("secret", tool_call_id="call", tool_name="read", is_error=False, run_id=None, session_id=None)

    assert list(tmp_path.glob("*.json")) == []
