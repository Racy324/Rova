from __future__ import annotations

import json

import pytest

from rova.agent_core.events import AgentEvent
from rova.agent_session.execution_journal import ExecutionEnvironmentIdentity, ToolExecutionJournal


def _completed_event() -> AgentEvent:
    return AgentEvent(
        "tool_execution_state",
        tool_call_id="call-1",
        tool_name="read",
        batch_id="batch-1",
        call_index=0,
        batch_mode="parallel",
        execution_mode="parallel",
        execution_state="completed",
        outcome="success",
        result="model-visible preview",
        is_error=False,
        metadata={
            "outcome": "success",
            "tool_output": {
                "externalized": True,
                "artifact_ref": "artifact-1",
                "original_size_chars": 120_000,
                "preview_truncated": True,
            },
        },
    )


def test_completed_receipt_round_trips_canonical_tool_result(tmp_path) -> None:
    journal = ToolExecutionJournal(tmp_path, "session1")

    journal.append(_completed_event(), assistant_entry_id="assistant-entry-1")

    records = journal.load()
    assert len(records) == 1
    record = records[0]
    assert record.assistant_entry_id == "assistant-entry-1"
    assert record.state == "completed"
    assert record.receipt is not None
    assert record.receipt.content == "model-visible preview"
    assert record.receipt.is_error is False
    assert record.receipt.metadata["tool_output"]["artifact_ref"] == "artifact-1"
    assert "raw artifact content" not in json.dumps(record.receipt.metadata)


def test_legacy_state_line_loads_without_a_completion_receipt(tmp_path) -> None:
    journal = ToolExecutionJournal(tmp_path, "session1")
    journal.path.write_text(
        json.dumps(
            {
                "batch_id": "batch-1",
                "tool_call_id": "call-1",
                "call_index": 0,
                "tool_name": "read",
                "batch_mode": "parallel",
                "execution_mode": "parallel",
                "state": "completed",
                "outcome": "success",
                "timestamp": "2026-09-07T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = journal.load()[0]

    assert record.is_legacy is True
    assert record.receipt is None


def test_v2_record_without_environment_identity_remains_readable(tmp_path) -> None:
    journal = ToolExecutionJournal(tmp_path, "session1")
    journal.append(_completed_event(), assistant_entry_id="assistant-entry-1")

    record = journal.load()[0]
    assert record.is_legacy is False
    assert record.environment_kind is None
    assert record.sandbox_id is None


def test_completed_receipt_preserves_an_empty_canonical_result(tmp_path) -> None:
    journal = ToolExecutionJournal(tmp_path, "session1")
    event = _completed_event()
    event.result = ""

    journal.append(event, assistant_entry_id="assistant-entry-1")

    record = journal.load()[0]
    assert record.receipt is not None
    assert record.receipt.content == ""


def test_v3_records_persist_the_execution_environment_identity(tmp_path) -> None:
    journal = ToolExecutionJournal(tmp_path, "session1")

    journal.append(
        _completed_event(),
        assistant_entry_id="assistant-entry-1",
        environment_identity=ExecutionEnvironmentIdentity("docker_sandbox", "sandbox-1"),
    )

    record = journal.load()[0]
    assert record.environment_kind == "docker_sandbox"
    assert record.sandbox_id == "sandbox-1"
    assert record.is_legacy is False
    assert "container" not in journal.path.read_text(encoding="utf-8")
