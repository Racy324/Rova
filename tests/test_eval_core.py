from datetime import datetime, timezone

import pytest

from rova.ai.messages import AssistantMessage, TextBlock, Usage
from rova.eval.models import (
    BehaviorMetrics,
    CheckResult,
    EvalCase,
    EvalExecution,
    EvalResult,
    EvalStatus,
    EvalSuiteResult,
    FailureReason,
)
from rova.trace.models import (
    CompactionStatus,
    CompactionTrace,
    CompactionTrigger,
    RunStatus,
    RunTrace,
    TerminationReason,
    ToolExecutionTrace,
    ToolOutcome,
    TurnTrace,
)


def _run_trace() -> RunTrace:
    started_at = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    return RunTrace(
        run_id="run-1",
        started_at=started_at,
        ended_at=datetime(2026, 8, 19, 12, 0, 1, tzinfo=timezone.utc),
        duration_ms=1_000.0,
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.FINAL_RESPONSE,
        turns=[
            TurnTrace(
                turn_index=0,
                started_at=started_at,
                usage=Usage(3, 2, 5),
            ),
            TurnTrace(
                turn_index=1,
                started_at=started_at,
                usage=Usage(7, 4, 11),
            ),
        ],
        tool_executions=[
            ToolExecutionTrace(
                tool_call_id="tool-1",
                tool_name="edit_file",
                arguments={},
                turn_index=0,
                started_at=started_at,
                is_error=True,
                outcome=ToolOutcome.POLICY_DENIED,
            ),
            ToolExecutionTrace(
                tool_call_id="tool-2",
                tool_name="edit_file",
                arguments={},
                turn_index=0,
                started_at=started_at,
                is_error=True,
                outcome=ToolOutcome.APPROVAL_DENIED,
            ),
            ToolExecutionTrace(
                tool_call_id="tool-3",
                tool_name="shell",
                arguments={},
                turn_index=1,
                started_at=started_at,
                is_error=False,
                outcome=ToolOutcome.COMMAND_NONZERO_EXIT,
            ),
            ToolExecutionTrace(
                tool_call_id="tool-4",
                tool_name="shell",
                arguments={},
                turn_index=1,
                started_at=started_at,
                is_error=True,
                outcome=ToolOutcome.COMMAND_TIMEOUT,
            ),
        ],
        compactions=[
            CompactionTrace(
                started_at=started_at,
                trigger=CompactionTrigger.AUTOMATIC,
                first_kept_entry_id="entry-1",
                status=CompactionStatus.COMPLETED,
            )
        ],
        usage=Usage(10, 6, 16),
    )


def test_eval_case_and_execution_keep_task_declaration_separate_from_runtime() -> None:
    case = EvalCase(
        case_id="calculator-fix",
        name="修复除法错误",
        prompt="修复 calculator.py 中的除法错误。",
        description="确定性的单文件修复任务。",
        tags=["bugfix", "single-file"],
        metadata={"fixture": "calculator"},
    )
    execution = EvalExecution(
        case_id=case.case_id,
        run_trace=_run_trace(),
        final_assistant=AssistantMessage(content=[TextBlock("已完成")]),
        artifacts={"workspace": "/tmp/eval"},
    )

    assert case.tags == ["bugfix", "single-file"]
    assert execution.case_id == "calculator-fix"
    assert execution.run_trace is not None
    assert execution.final_assistant is not None


def test_behavior_metrics_are_derived_from_trace_without_task_success_judgment() -> None:
    metrics = BehaviorMetrics.from_run_trace(_run_trace())

    assert metrics.turn_count == 2
    assert metrics.tool_call_count == 4
    assert metrics.tool_error_count == 3
    assert metrics.policy_denied_count == 1
    assert metrics.approval_denied_count == 1
    assert metrics.shell_nonzero_count == 1
    assert metrics.shell_timeout_count == 1
    assert metrics.compaction_count == 1
    assert (metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (10, 6, 16)
    assert metrics.run_duration_ms == 1_000.0
    assert not hasattr(metrics, "task_success")


def test_eval_result_serialization_references_run_id_without_copying_full_trace() -> None:
    started_at = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    result = EvalResult(
        eval_run_id="eval-run-1",
        case_id="calculator-fix",
        status=EvalStatus.FAIL,
        started_at=started_at,
        ended_at=datetime(2026, 8, 19, 12, 0, 2, tzinfo=timezone.utc),
        duration_ms=2_000.0,
        run_id="run-1",
        checks=[CheckResult("pytest", passed=False, message="1 failed")],
        task_success=False,
        runtime_termination_reason=TerminationReason.MAX_TURNS,
        failure_reason=FailureReason.TASK_VALIDATION_FAILED,
        metrics=BehaviorMetrics.from_run_trace(_run_trace()),
    )

    serialized = result.to_dict()
    restored = EvalResult.from_dict(serialized)

    assert serialized["run_id"] == "run-1"
    assert "run_trace" not in serialized
    assert restored == result


def test_suite_result_counts_statuses_and_task_success_rate() -> None:
    now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    metric = BehaviorMetrics.from_run_trace(_run_trace())
    suite = EvalSuiteResult(
        suite_id="core",
        case_results=[
            EvalResult("one", "case-1", EvalStatus.PASS, now, now, 0.0, None, [], True, None, None, metric),
            EvalResult("two", "case-2", EvalStatus.FAIL, now, now, 0.0, None, [], False, None, FailureReason.TASK_VALIDATION_FAILED, metric),
            EvalResult("three", "case-3", EvalStatus.ERROR, now, now, 0.0, None, [], None, None, FailureReason.EVALUATOR_ERROR, metric),
        ],
    )

    assert (suite.total, suite.passed, suite.failed, suite.errors) == (3, 1, 1, 1)
    assert suite.task_success_rate == 0.5


def test_eval_result_enforces_status_task_success_and_failure_semantics() -> None:
    now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    metrics = BehaviorMetrics.from_run_trace(_run_trace())

    with pytest.raises(ValueError, match="PASS"):
        EvalResult(
            "bad-pass", "case", EvalStatus.PASS, now, now, 0.0, None, [], False,
            None, None, metrics,
        )
    with pytest.raises(ValueError, match="FAIL"):
        EvalResult(
            "bad-fail", "case", EvalStatus.FAIL, now, now, 0.0, None, [], False,
            None, FailureReason.EVALUATOR_ERROR, metrics,
        )
    with pytest.raises(ValueError, match="ERROR"):
        EvalResult(
            "bad-error", "case", EvalStatus.ERROR, now, now, 0.0, None, [], None,
            None, None, metrics,
        )

    result = EvalResult(
        "max-turn-pass", "case", EvalStatus.PASS, now, now, 0.0, "run", [], True,
        TerminationReason.MAX_TURNS, None, metrics,
    )
    assert result.task_success is True


def test_check_result_metadata_is_json_safe_and_defensively_copied() -> None:
    metadata = {"nested": {"values": ["before"]}}
    check = CheckResult("workspace", passed=True, metadata=metadata)
    metadata["nested"]["values"].append("after")

    serialized = check.to_dict()
    serialized["metadata"]["nested"]["values"].append("external")

    assert check.metadata == {"nested": {"values": ["before"]}}
    with pytest.raises(ValueError, match="finite"):
        CheckResult("bad", passed=False, metadata={"value": float("nan")})
    with pytest.raises(ValueError, match="JSON"):
        CheckResult("bad", passed=False, metadata={"value": object()})


def test_eval_result_from_dict_rejects_non_dict_unknown_fields_and_non_finite_numbers() -> None:
    now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    result = EvalResult(
        "eval", "case", EvalStatus.FAIL, now, now, 1.0, "run", [], False,
        None, FailureReason.TASK_VALIDATION_FAILED,
        BehaviorMetrics.from_run_trace(_run_trace()),
    )
    serialized = result.to_dict()

    with pytest.raises(ValueError, match="object"):
        EvalResult.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown"):
        EvalResult.from_dict({**serialized, "unexpected": True})
    with pytest.raises(ValueError, match="finite"):
        EvalResult.from_dict({**serialized, "duration_ms": float("inf")})
    with pytest.raises(ValueError, match="finite"):
        EvalResult.from_dict(
            {
                **serialized,
                "metrics": {**serialized["metrics"], "run_duration_ms": float("nan")},
            }
        )


def test_behavior_metrics_aggregates_turn_usage_when_run_usage_is_missing() -> None:
    trace = _run_trace()
    trace.usage = None

    metrics = BehaviorMetrics.from_run_trace(trace)

    assert (metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (10, 6, 16)
