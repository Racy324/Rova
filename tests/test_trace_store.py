from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from rova.ai.messages import AssistantMessage, TextBlock, Usage
from rova.trace import (
    CompactionStatus,
    CompactionTrace,
    CompactionTrigger,
    JsonlTraceStore,
    RunStatus,
    RunTrace,
    StepTrace,
    StepUsageTrace,
    TerminationReason,
    ToolCallTrace,
    ToolOutcome,
    ToolResultTrace,
    TraceCorruptionError,
    TraceStoreError,
    run_trace_to_dict,
)


def make_trace(run_id: str = "run1") -> RunTrace:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return RunTrace(
        run_id,
        timestamp,
        ended_at=timestamp,
        duration_ms=1.5,
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.FINAL_RESPONSE,
        input_message="hello",
        steps=[
            StepTrace(
                1,
                timestamp,
                ended_at=timestamp,
                duration_ms=1.5,
                assistant_message=AssistantMessage([TextBlock("你好")]),
                usage=StepUsageTrace(estimated_input_tokens=1, actual_usage=Usage(1, 2, 3)),
                tool_calls=[
                    ToolCallTrace(
                        "tool-1", "shell", {"command": "echo hi"}, "batch-1", 0, "sequential", "sequential",
                        executed=True, committed=True, started_at=timestamp, ended_at=timestamp, duration_ms=0.1,
                        outcome=ToolOutcome.SUCCESS,
                        result=ToolResultTrace("ok", True, "artifact-1", 2, False, 2),
                        diagnostics={"command": "echo hi", "nested": [1, "two"]},
                    )
                ],
            )
        ],
        actual_usage_complete=True,
        actual_usage=Usage(1, 2, 3),
        compactions=[
            CompactionTrace(
                timestamp, CompactionTrigger.MANUAL, "entry-1", ended_at=timestamp,
                duration_ms=0.2, status=CompactionStatus.COMPLETED,
            )
        ],
    )


def test_jsonl_trace_store_round_trips_finalized_trace_and_rejects_duplicates(tmp_path):
    store = JsonlTraceStore(tmp_path / "trace.jsonl")
    trace = make_trace()
    store.append(trace)
    assert store.load_all() == [trace]
    with pytest.raises(TraceStoreError, match="duplicate"):
        store.append(trace)


def test_jsonl_trace_store_writes_v3_step_schema_without_legacy_turn_fields(tmp_path):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trace = RunTrace(
        "v3",
        timestamp,
        ended_at=timestamp,
        duration_ms=1,
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.FINAL_RESPONSE,
        input_message="hello",
        steps=[
            StepTrace(
                1,
                timestamp,
                ended_at=timestamp,
                duration_ms=1,
                assistant_message=AssistantMessage([TextBlock("done")]),
                usage=StepUsageTrace(estimated_input_tokens=7, actual_usage=Usage(7, 1, 8)),
            )
        ],
        actual_usage_complete=True,
        actual_usage=Usage(7, 1, 8),
    )
    store = JsonlTraceStore(tmp_path / "trace.jsonl")

    store.append(trace)

    record = json.loads(store.path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 3
    assert set(record["trace"]) >= {"input_message", "steps", "actual_usage_complete", "actual_usage"}
    assert "turns" not in record["trace"]
    assert "tool_executions" not in record["trace"]
    assert store.load_all()[0].steps[0].usage.estimated_input_tokens == 7


def test_jsonl_trace_store_keeps_multiple_runs_in_append_order(tmp_path):
    store = JsonlTraceStore(tmp_path / "trace.jsonl")
    store.append(make_trace("first"))
    store.append(make_trace("second"))

    assert [trace.run_id for trace in store.load_all()] == ["first", "second"]


def test_jsonl_trace_store_rejects_malformed_record(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"schema_version":1,"trace":{}}\nnot json\n', encoding="utf-8")
    with pytest.raises(TraceCorruptionError, match="line 1|line 2"):
        JsonlTraceStore(path).load_all()


def test_jsonl_trace_store_rejects_an_unsupported_schema_version(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"schema_version":99,"trace":{}}\n', encoding="utf-8")

    with pytest.raises(TraceCorruptionError, match="schema_version"):
        JsonlTraceStore(path).load_all()


def test_jsonl_trace_store_reads_v1_records_without_memory_events(tmp_path):
    from rova.trace.store import _legacy_trace_to_dict

    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    legacy = RunTrace(
        "legacy", timestamp, ended_at=timestamp, duration_ms=1,
        status=RunStatus.COMPLETED, termination_reason=TerminationReason.FINAL_RESPONSE,
    )
    payload = _legacy_trace_to_dict(legacy)
    payload.pop("memory_events")
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"schema_version": 1, "trace": payload}) + "\n", encoding="utf-8")

    traces = JsonlTraceStore(path).load_all()

    assert traces[0].memory_events == []


def test_jsonl_trace_store_rejects_unfinalized_trace_and_does_not_mutate_it(tmp_path):
    trace = RunTrace("active", datetime(2026, 1, 1, tzinfo=timezone.utc))
    store = JsonlTraceStore(tmp_path / "trace.jsonl")

    with pytest.raises(TraceStoreError, match="finalized"):
        store.append(trace)

    assert trace.status is RunStatus.RUNNING
    assert trace.ended_at is None
    assert store.load_all() == []


def test_jsonl_trace_store_rejects_nonfinite_values_and_normalizes_timestamps_to_utc(tmp_path):
    store = JsonlTraceStore(tmp_path / "trace.jsonl")
    invalid = make_trace("invalid")
    invalid.duration_ms = float("nan")
    with pytest.raises(TraceStoreError, match="finite"):
        store.append(invalid)

    offset = timezone(timedelta(hours=8))
    trace = make_trace("offset")
    trace.started_at = datetime(2026, 1, 1, 8, tzinfo=offset)
    trace.ended_at = datetime(2026, 1, 1, 8, tzinfo=offset)
    store.append(trace)
    assert '"started_at":"2026-01-01T00:00:00+00:00"' in store.path.read_text(encoding="utf-8")
