from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from rova.eval.models import (
    BehaviorMetrics,
    CheckResult,
    EvalCase,
    EvalExecution,
    EvalStatus,
    FailureReason,
)
from rova.eval.runner import EvalRunner
from rova.trace.models import RunStatus, RunTrace, TerminationReason


def _case(case_id: str) -> EvalCase:
    return EvalCase(case_id=case_id, name=case_id, prompt="complete the task")


def _execution(
    case_id: str,
    termination: TerminationReason | None = TerminationReason.FINAL_RESPONSE,
) -> EvalExecution:
    now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    trace = RunTrace(
        run_id=f"run-{case_id}",
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        status=RunStatus.COMPLETED,
        termination_reason=termination,
    )
    return EvalExecution(case_id=case_id, run_trace=trace)


class _Executor:
    def __init__(self, executions: dict[str, EvalExecution]) -> None:
        self.executions = executions
        self.executed: list[str] = []

    async def execute(self, case: EvalCase) -> EvalExecution:
        self.executed.append(case.case_id)
        return self.executions[case.case_id]


class _Check:
    def __init__(self, name: str, passed: bool) -> None:
        self.name = name
        self.passed = passed
        self.calls: list[str] = []

    async def evaluate(self, case: EvalCase, execution: EvalExecution) -> CheckResult:
        self.calls.append(case.case_id)
        return CheckResult(self.name, self.passed)


@pytest.mark.asyncio
async def test_runner_marks_failed_check_as_fail_and_continues_suite() -> None:
    failing_case = _case("failing")
    passing_case = _case("passing")
    executor = _Executor(
        {
            failing_case.case_id: _execution(failing_case.case_id),
            passing_case.case_id: _execution(passing_case.case_id),
        }
    )
    failed_check = _Check("workspace", False)
    passed_check = _Check("workspace", True)

    suite = await EvalRunner(
        executor,
        lambda case: [failed_check] if case.case_id == "failing" else [passed_check],
        suite_id="suite",
    ).run([failing_case, passing_case])

    assert [result.status for result in suite.case_results] == [
        EvalStatus.FAIL,
        EvalStatus.PASS,
    ]
    assert suite.case_results[0].task_success is False
    assert suite.case_results[0].failure_reason is FailureReason.TASK_VALIDATION_FAILED
    assert suite.case_results[1].task_success is True
    assert executor.executed == ["failing", "passing"]


@pytest.mark.asyncio
async def test_runner_runs_evaluators_after_runtime_failure_and_maps_reason() -> None:
    case = _case("provider-failure")
    evaluator = _Check("workspace", False)
    execution = _execution(case.case_id, TerminationReason.PROVIDER_ERROR)

    result = (
        await EvalRunner(_Executor({case.case_id: execution}), lambda _: [evaluator]).run([case])
    ).case_results[0]

    assert evaluator.calls == [case.case_id]
    assert result.status is EvalStatus.FAIL
    assert result.task_success is False
    assert result.failure_reason is FailureReason.RUNTIME_PROVIDER_ERROR
    assert result.runtime_termination_reason is TerminationReason.PROVIDER_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("termination", "failure_reason"),
    [
        (TerminationReason.HARNESS_ERROR, FailureReason.RUNTIME_HARNESS_ERROR),
        (TerminationReason.ABORTED, FailureReason.RUNTIME_ABORTED),
        (TerminationReason.MAX_TURNS, FailureReason.MAX_TURNS),
        (
            TerminationReason.SESSION_PERSISTENCE_ERROR,
            FailureReason.SESSION_PERSISTENCE_ERROR,
        ),
    ],
)
async def test_runner_maps_nonfinal_runtime_termination_when_check_fails(
    termination: TerminationReason, failure_reason: FailureReason
) -> None:
    case = _case(termination.value)

    result = (
        await EvalRunner(
            _Executor({case.case_id: _execution(case.case_id, termination)}),
            lambda _: [_Check("workspace", False)],
        ).run([case])
    ).case_results[0]

    assert result.status is EvalStatus.FAIL
    assert result.failure_reason is failure_reason


@pytest.mark.asyncio
async def test_all_passing_checks_are_pass_even_when_runtime_reached_max_turns() -> None:
    case = _case("fixed-before-limit")

    result = (
        await EvalRunner(
            _Executor(
                {
                    case.case_id: _execution(
                        case.case_id, TerminationReason.MAX_TURNS
                    )
                }
            ),
            lambda _: [_Check("pytest", True), _Check("file", True)],
        ).run([case])
    ).case_results[0]

    assert result.status is EvalStatus.PASS
    assert result.task_success is True
    assert result.failure_reason is None
    assert result.runtime_termination_reason is TerminationReason.MAX_TURNS


class _RaisingEvaluator:
    async def evaluate(self, case: EvalCase, execution: EvalExecution) -> CheckResult:
        raise RuntimeError("evaluator bug")


class _RaisingExecutor:
    async def execute(self, case: EvalCase) -> EvalExecution:
        raise RuntimeError("executor bug")


@pytest.mark.asyncio
async def test_evaluator_or_executor_exception_marks_only_that_case_as_error() -> None:
    error_case = _case("error")
    later_case = _case("later")
    executor = _Executor({error_case.case_id: _execution(error_case.case_id), later_case.case_id: _execution(later_case.case_id)})
    suite = await EvalRunner(
        executor,
        lambda case: [_RaisingEvaluator()] if case.case_id == "error" else [_Check("ok", True)],
    ).run([error_case, later_case])

    assert [result.status for result in suite.case_results] == [
        EvalStatus.ERROR,
        EvalStatus.PASS,
    ]
    assert suite.case_results[0].task_success is None
    assert suite.case_results[0].failure_reason is FailureReason.EVALUATOR_ERROR

    executor_error = (
        await EvalRunner(_RaisingExecutor(), lambda _: [_Check("unused", True)]).run([error_case])
    ).case_results[0]
    assert executor_error.status is EvalStatus.ERROR
    assert executor_error.failure_reason is FailureReason.EVALUATOR_ERROR


class _CancelledExecutor:
    async def execute(self, case: EvalCase) -> EvalExecution:
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_runner_propagates_cancellation() -> None:
    with pytest.raises(asyncio.CancelledError):
        await EvalRunner(_CancelledExecutor(), lambda _: []).run([_case("cancel")])
