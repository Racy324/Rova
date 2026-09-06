from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from rova.trace.models import TerminationReason, TraceError

from .evaluators import Evaluator
from .models import (
    BehaviorMetrics,
    CheckResult,
    EvalCase,
    EvalExecution,
    EvalResult,
    EvalStatus,
    EvalSuiteResult,
    FailureReason,
)


class EvalExecutor(Protocol):
    async def execute(self, case: EvalCase) -> EvalExecution: ...


EvaluatorResolver = Callable[[EvalCase], Iterable[Evaluator]]


class EvalRunner:
    def __init__(self, executor: EvalExecutor, evaluators_for_case: EvaluatorResolver, *, suite_id: str | None = None) -> None:
        self._executor = executor
        self._evaluators_for_case = evaluators_for_case
        self._suite_id = suite_id or uuid4().hex

    async def run(self, cases: Iterable[EvalCase]) -> EvalSuiteResult:
        results: list[EvalResult] = []
        for case in cases:
            results.append(await self._run_case(case))
        return EvalSuiteResult(self._suite_id, results)

    async def _run_case(self, case: EvalCase) -> EvalResult:
        started_at = _now()
        started_perf_counter = time.perf_counter()
        try:
            execution = await self._executor.execute(case)
            if execution.case_id != case.case_id:
                raise ValueError("executor returned an execution for a different case")
            checks = [await evaluator.evaluate(case, execution) for evaluator in self._evaluators_for_case(case)]
            return _result_from_execution(case, execution, checks, started_at, started_perf_counter)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _error_result(case, started_at, started_perf_counter, error)


def _result_from_execution(case: EvalCase, execution: EvalExecution, checks: list[CheckResult], started_at: datetime, started_perf_counter: float) -> EvalResult:
    trace = execution.run_trace
    metrics = BehaviorMetrics.from_run_trace(trace) if trace is not None else _empty_metrics()
    termination = trace.termination_reason if trace is not None else None
    passed = all(check.passed for check in checks)
    if passed:
        status = EvalStatus.PASS
        task_success = True
        failure_reason = None
    else:
        status = EvalStatus.FAIL
        task_success = False
        failure_reason = _runtime_failure_reason(termination) or FailureReason.TASK_VALIDATION_FAILED
    return EvalResult(
        eval_run_id=uuid4().hex,
        case_id=case.case_id,
        status=status,
        started_at=started_at,
        ended_at=_now(),
        duration_ms=_elapsed_ms(started_perf_counter),
        run_id=trace.run_id if trace is not None else None,
        checks=checks,
        task_success=task_success,
        runtime_termination_reason=termination,
        failure_reason=failure_reason,
        metrics=metrics,
        error=execution.execution_error,
    )


def _error_result(case: EvalCase, started_at: datetime, started_perf_counter: float, error: Exception) -> EvalResult:
    return EvalResult(
        eval_run_id=uuid4().hex,
        case_id=case.case_id,
        status=EvalStatus.ERROR,
        started_at=started_at,
        ended_at=_now(),
        duration_ms=_elapsed_ms(started_perf_counter),
        run_id=None,
        checks=[],
        task_success=None,
        runtime_termination_reason=None,
        failure_reason=FailureReason.EVALUATOR_ERROR,
        metrics=_empty_metrics(),
        error=TraceError(type(error).__name__, str(error)),
    )


def _runtime_failure_reason(reason: TerminationReason | None) -> FailureReason | None:
    return {
        TerminationReason.PROVIDER_ERROR: FailureReason.RUNTIME_PROVIDER_ERROR,
        TerminationReason.HARNESS_ERROR: FailureReason.RUNTIME_HARNESS_ERROR,
        TerminationReason.ABORTED: FailureReason.RUNTIME_ABORTED,
        TerminationReason.MAX_TURNS: FailureReason.MAX_TURNS,
        TerminationReason.SESSION_PERSISTENCE_ERROR: FailureReason.SESSION_PERSISTENCE_ERROR,
    }.get(reason)


def _empty_metrics() -> BehaviorMetrics:
    return BehaviorMetrics(
        turn_count=0,
        tool_call_count=0,
        tool_error_count=0,
        policy_denied_count=0,
        approval_denied_count=0,
        shell_nonzero_count=0,
        shell_timeout_count=0,
        compaction_count=0,
        actual_usage_available=False,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        run_duration_ms=None,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed_ms(started_perf_counter: float) -> float:
    return max(0.0, (time.perf_counter() - started_perf_counter) * 1000)
